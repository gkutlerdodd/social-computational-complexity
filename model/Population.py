# -*- coding: utf-8 -*-

import numpy as np
from numpy import sum as nsum
from numpy import add as nadd
from numpy import zeros as zeros
from numpy import nonzero
from numpy import int32 as nint32, uint32 as nuint32
from numpy.random import binomial
from numpy.random import multinomial
import math

import genome


class Population(object):
    """Represent a population within a metapopulation

    A population is a collection of individuals. Each individual is represented
    by a number. The binary representation of that number defines that
    individual's genotype. The state of the highest order bit determines whether
    (1) or not (0) that individual is a producer.

    * genome_length: the length of the genome. The production allele is added to
        this, so the number of genotypes is 2^(genome_length+1)
    * stress_survival_rate: the probability of an individual surviving a change
        of environment (stress)
    * mutation_rate_social: the probability of a mutation (bit flip) occuring at
        the social locus
    * mutation_rate_adaptation: the probability of a mutation (bit flip) at a
        non-social locus
    * capacity_min: the minimum size of a fully-grown population. This occurs
        when there are no producers
    * capacity_max: the maximum size of a fully-grown population. This occurs
        when a population consists entirely of producers
    * production_cost: the fitness cost of production. This manifests itself as
        a decrease in growth rate
    * initialize: How to initialize the population
        empty: the population will have no individuals


    """

    def __init__(self, metapopulation, config):
        """Initialize a Population object"""
        self.metapopulation = metapopulation
        self.config = config

        self.genome_length = config['Population']['genome_length']
        self.stress_survival_rate = config['Population']['stress_survival_rate']
        self.mutation_rate_social = config['Population']['mutation_rate_social']
        self.mutation_rate_adaptation = config['Population']['mutation_rate_adaptation']
        self.dilution_factor = config['Population']['dilution_factor']
        self.dilution_prob_min = config['Population']['dilution_prob_min']
        self.capacity_min = config['Population']['capacity_min']
        self.capacity_max = config['Population']['capacity_max']
        self.capacity_shape = config['Population']['capacity_shape']
        self.production_cost = config['Population']['production_cost']
        self.initialize = config['Population']['initialize']
        self.max_types = config['Simulation']['max_types']

        # Create an empty population
        if self.initialize.lower() == 'empty':
            self.empty()
        else:
            raise Exception("Must have population initialization as empty!")
        #elif self.initialize.lower() == 'random':
            #self.randomize()

        self.delta = zeros(self.abundances.size, dtype=nint32)
        self.diluted = True


    def __repr__(self):
        """Return a string representation of a Population object"""
        res = "Population: Size {s}, {p} producers".format(s=int(self.size()),
                                                               p=self.prop_producers())
        return res


    def empty(self):
        """Empty a population"""
        #GKD: NEW - abundances is now of length max_types
        self.abundances = zeros(self.max_types, dtype=nuint32)
        
        #self.abundances = zeros(2**(self.genome_length + 1), dtype=nuint32)


    #Deprecated
    #def randomize(self):
        #"""Create a random population"""
        #self.abundances = np.random.random_integers(low=0,
                                                    #high=self.capacity_min,
                                                    #size=2**(self.genome_length+1))

    def select_mutant(self, parent_genotype):
        """Select a new mutant

        Generate the probability for a given genotype to mutate into any other genotype, then remove those already
        already seen (i.e. non-novel mutants) and select one of the remaining genotypes

        """
        #Calculate Hamming distances
        hamming_distances = np.array([genome.hamming_distance(parent_genotype-1, i) for i in range(2**(self.genome_length+1))])

        if parent_genotype > int("1"+"0"*self.genome_length, 2):
            social_hd = np.append(np.ones(2**self.genome_length), np.zeros(2**self.genome_length))
        else:
            social_hd = np.append(np.zeros(2**self.genome_length), np.ones(2**self.genome_length))

        nonsocial_hd = hamming_distances - social_hd

        #Calculate mutation probabilities
        npower = np.power
        mr = npower(1 - self.mutation_rate_adaptation, self.genome_length - nonsocial_hd) * \
             npower(self.mutation_rate_adaptation, nonsocial_hd) * \
             npower(1 - self.mutation_rate_social, 1 - social_hd) * \
             npower(self.mutation_rate_social, social_hd)

        #Set probabilities of existing genotypes to 0 and normalize
        for i in self.metapopulation.type2geno[:self.metapopulation.current_type]:
            mr[i-1] = 0

        mr = mr/sum(mr)


        #Step 7: select new genotype and return it
        return np.random.choice(np.arange(1, 2**(self.genome_length+1)+1), p=mr)


    def dilute(self):
        """Dilute a population
        
        dilute dilutes the population by the dilution factor, which is specified
        in the Population section of the configuration as dilution_factor.

        """
        if self.is_empty():
            return

        self.diluted = False
        prob_dilute = 1

        if self.dilution_prob_min < 1:
            prob_dilute = self.dilution_prob_min + (1.0 - self.dilution_prob_min) * self.prop_producers()

        if self.dilution_prob_min == 1 or binomial(n=1, p=prob_dilute, size=1)[0]:
            self.abundances = binomial(self.abundances, self.dilution_factor)
            self.diluted = True


    def grow(self):
        """Grow the population to carrying capacity
        
        The final population size is determined based on the proportion of
        producers present. This population is determined by drawing from a
        multinomial with the probability of each genotype proportional to its
        abundance times its fitness.
        """

        mode = self.config['Simulation']['mode']

        if self.is_empty() or not self.diluted:
            return

        landscape = self.metapopulation.fitness_landscape

        if mode == 'cooperation':
            final_size = self.capacity_min + \
                    (self.capacity_max - self.capacity_min) * \
                    (self.prop_producers()**self.capacity_shape)

        #For simulations of costly learning, patch size does not depend on the proportion of learners
        else:
            final_size = self.capacity_min

        #NEW!! scaling patch size with fitness
        #final_size = self.capacity_min + \
                #(self.capacity_max - self.capacity_min) * \
                #(self.prop_producers()**self.capacity_shape) * \
                #math.log(self.average_fitness() + 1)

        grow_probs = self.abundances * (landscape/nsum(landscape))

        if nsum(grow_probs) > 0:
            norm_grow_probs = grow_probs/nsum(grow_probs)
            self.abundances = multinomial(final_size, norm_grow_probs, 1)[0]

        self.metapopulation.num_births += self.size()


    def mutate(self):
        """Mutate a Population
        
        Each genotype mutates to another with probability inversely proportional
        to the Hamming distance (# different bits in binary representation)
        between them.
        
        """

        if self.is_empty():
            return

        if not self.diluted:
            return

        mutated_population = zeros(self.abundances.size, dtype=nuint32)
        new_genos = []
        current_type = self.metapopulation.current_type

        #GKD - new
        #If some individuals of a particularly genotype will mutate to a different genotype not in type2geno,
        #draw new genotype(s)
        for i in nonzero(self.abundances)[0]:
            new_mutants = multinomial(self.abundances[i], self.metapopulation.mutation_probs[i], size=1)[0]
            mutated_population = nadd(mutated_population, new_mutants[:-1])
            if new_mutants[-1] != 0:
                #For each new mutation, draw a new genotype
                for j in range(new_mutants[-1]):
                    new_genotype = self.select_mutant(self.metapopulation.type2geno[i])
                    if new_genotype not in new_genos:
                        if current_type == self.max_types:
                            raise Exception("Maximum number of genotypes exceeded - increase max_types")

                        mutated_population[current_type] = 1
                        new_genos.append(new_genotype)
                        current_type += 1
                    else:
                        #Back-index to figure out which position to put in
                        mutated_population[self.metapopulation.current_type - len(new_genos) + new_genos.index(new_genotype)] += 1

        self.abundances = mutated_population
        if new_genos:
            return new_genos


    def select_migrants(self, migration_rate):
        """Select individuals to migrate
                                    
        Select genotypes to migrate. The amount of each genotype that migrates
        is chosen in proportion to that genotype's abundance.
                                    
        """

        assert migration_rate >= 0 and migration_rate <= 1

        return binomial(self.abundances, migration_rate)


    def remove_emigrants(self, emigrants):
        """Remove emigrants from the population
                                    
        remove_emigrants removes the given emigrants from the population. The
        genotypes are not immediately removed to the population, but their
        counts are placed in a temporary area until census() is called.

        """
        self.delta -= emigrants


    def add_immigrants(self, immigrants):
        """Add immigrants to the population
                                    
        add_immigrants adds the given immigrants to the population. The new
        genotypes are not immediately added to the population, but placed in
        a temporary area until census() is called.

        """
        self.delta += immigrants


    def census(self):
        """Update the population's abundances after migration
        
        When migration occurs, the immigrants and emigrants are not directly
        accounted for in the list of genotype abundances. This function adds
        immigrants and removes emigrants to/from the abundances.
                                    
        """

        self.abundances = nadd(self.abundances, self.delta)
        self.delta = zeros(self.abundances.size, dtype=nint32)


    def reset_loci(self):
        """Reset the loci of the population to all zeros

        When an environment changes, the population is not yet adapted to it.
        This function captures this change by resetting num_loci fitness-encoding
        loci to zero.
        """
        if self.is_empty():
            return

        #GKD: new
        num_producers = self.num_producers()
        num_nonproducers = self.num_nonproducers()

        self.abundances = zeros(self.max_types, dtype=nuint32)
        self.abundances[0] = num_nonproducers
        self.abundances[1] = num_producers

        #new_abundances = zeros(self.abundances.size, dtype=np.int)
        #gs = np.right_shift(np.arange(start=0, stop=2**self.genome_length), num_loci)
        #genotypes_shifted = np.append(gs, gs + (2**self.genome_length))

        #for i in range(2**(self.genome_length+1)):
            #new_abundances[genotypes_shifted[i]] += self.abundances[i]

        #self.abundances = new_abundances

    def bottleneck(self, survival_rate):
        """ Pass the population through a bottleneck

        This function passes the population through a bottleneck. The
        probability of survival is specified as the survival_rate parameter
        [0,1]. 
        """

        assert survival_rate >= 0
        assert survival_rate <= 1

        self.abundances = binomial(self.abundances, survival_rate)


    def size(self):
        """Get the size of the population"""
        return self.abundances.sum()


    def __len__(self):
        return self.abundances.sum()


    def is_empty(self):
        """Return whether or not the population is empty"""
        return self.abundances.sum() == 0


    def num_producers(self):
        """Get the number of producers"""
        #GKD - new

        return nsum(self.abundances[self.metapopulation.social_types == 1])

        #return self.abundances[2**self.genome_length:].sum()


    def num_nonproducers(self):
        """Get the number of non-producers"""
        #GKD - new

        return nsum(self.abundances[self.metapopulation.social_types == 0])

        #return self.abundances[:2**self.genome_length].sum()


    def prop_producers(self):
        """Get the proportion of producers"""
        popsize = self.abundances.sum()
        
        if popsize == 0:
            return 'NA'
        else:
            return 1.0 * self.num_producers() / popsize


    def average_fitness(self):
        """Get the average fitness in the population"""

        popsize = self.size()
        landscape = self.metapopulation.fitness_landscape

        if popsize == 0:
            return 'NA'
        else:
            return nsum(self.abundances * landscape)/popsize

    def average_fitness_producers(self):
        """Get the average fitness of producers"""
        popsize = self.size()
        landscape = self.metapopulation.fitness_landscape
        num_producers = self.num_producers()

        if popsize == 0 or num_producers == 0:
            return np.nan
        else:
            return nsum(self.abundances[self.metapopulation.social_types == 1] * landscape[self.metapopulation.social_types == 1])/num_producers

    def average_fitness_nonproducers(self):
        """Get the average fitness of nonproducers"""

        popsize = self.size()
        landscape = self.metapopulation.fitness_landscape
        num_nonproducers = self.num_nonproducers()

        if popsize == 0 or num_nonproducers == 0:
            return np.nan
        else:
            return nsum(self.abundances[self.metapopulation.social_types == 0] * landscape[self.metapopulation.social_types == 0])/num_producers



    def max_fitnesses(self):
        """Get the maximum fitness among producers and non-producers"""

        popsize = self.size()

        if popsize == 0:
            return (0,0)

        # Get the fitnesses of genotypes present in the population
        fitnesses = np.array(self.abundances > 0, dtype=int) * self.metapopulation.fitness_landscape

        #GKD - check type2geno to find which abundances are producers and which are nonproducers
        max_producer = np.max(fitnesses[self.metapopulation.social_types == 1])
        max_nonproducer = np.max(fitnesses[self.metapopulation.social_types == 0])

        #max_producer = fitnesses[2**self.genome_length:].max()
        #max_nonproducer = fitnesses[:2**self.genome_length].max()

        return (max_producer, max_nonproducer)
        
    def get_genomes_present(self):
        """Get genomes in population and their frequency"""

        #GKD - new
        genome_dict = dict()
        type2geno = self.metapopulation.type2geno
        for i in range(self.metapopulation.current_type):
            if self.abundances[i] > 0:
                genome_dict[type2geno[i]] = self.abundances[i]

        #genome_dict = dict()
        #indices = np.array(range(2**(self.genome_length+1)))
       
        #for i,g in zip(indices[self.abundances >0], self.abundances[self.abundances>0]):
            #genome_dict[i]=g

        return genome_dict
