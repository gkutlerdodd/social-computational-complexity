# -*- coding: utf-8 -*-

import os

import networkx as nx
import numpy as np
from numpy import zeros as zeros
from numpy import int32 as nint32, uint32 as nuint32
from numpy.random import binomial, choice as nchoice, exponential, random_integers

import genome
from Population import Population
import topology as topology
from PopulationOutput import PopulationOutput
from MetapopulationOutput import MetapopulationOutput
from GenotypesOutput import GenotypesOutput
from FitnessOutput import FitnessOutput
from EnvChangeOutput import EnvChangeOutput
from datetime import datetime


class Metapopulation(object):

    def __init__(self, config):
        """Initialize a Metapopulation object"""

        self.config = config
        self.time = 0
        self.num_births = 0

        if self.config['Simulation']['mode'] != 'cooperation' and self.config['Simulation']['mode'] != 'learning':
            raise Exception("Simulation mode in config['Simulation']['mode'] must be either 'cooperation' or 'learning'")

        self.migration_rate = self.config['Metapopulation']['migration_rate']
        self.migration_dest = self.config['Metapopulation']['migration_dest']
        self.migration_p_far = self.config['Metapopulation']['migration_p_far']
        topology_type = self.config['Metapopulation']['topology']

        if topology_type.lower() == 'moore':
            width = int(self.config['MooreTopology']['width'])
            height = self.config['MooreTopology']['height']
            periodic = self.config['MooreTopology']['periodic']
            radius = self.config['MooreTopology']['radius']

            self.topology = topology.moore_lattice(rows=height, columns=width,
                                                   radius=radius,
                                                   periodic=periodic)

        elif topology_type.lower() == 'vonneumann':
            width = self.config['VonNeumannTopology']['width']
            height = self.config['VonNeumannTopology']['height']
            periodic = self.config['VonNeumannTopology']['periodic']

            self.topology = topology.vonneumann_lattice(rows=height,
                                                        columns=width,
                                                        periodic=periodic)

        elif topology_type.lower() == 'smallworld':
            size = self.config['SmallWorldTopology']['size']
            neighbors = self.config['SmallWorldTopology']['neighbors']
            edgeprob = self.config['SmallWorldTopology']['edgeprob']

            if self.config['SmallWorldTopology']['seed'] == 0:
                self.config['SmallWorldTopology']['seed'] = self.config['Simulation']['seed']

            seed = self.config['SmallWorldTopology']['seed']

            self.topology = topology.smallworld(size=size, neighbors=neighbors,
                                                edgeprob=edgeprob, seed=seed)


        elif topology_type.lower() == 'complete':
            self.topology = nx.complete_graph(n=self.config['CompleteTopology']['size'])


        elif topology_type.lower() == 'regular':
            size = self.config['RegularTopology']['size']
            degree = self.config['RegularTopology']['degree']

            if self.config['RegularTopology']['seed'] == 0:
                self.config['RegularTopology']['seed'] = self.config['Simulation']['seed']

            seed = self.config['RegularTopology']['seed']

            self.topology = topology.regular(size=size, degree=degree,
                                             seed=seed)


        # Export the structure of the topology, allowing the topology to be
        # re-created. This is especially useful for randomly-generated
        # topologies.
        if self.config['Metapopulation']['export_topology']:
            nx.write_gml(self.topology, os.path.join(self.config['Simulation']['data_dir'], 'topology.gml'))

        #GKD: Build the type2geno list and starting fitnesses
        #The first two entries of type2geno are L+1 zeros (defector with 0 adaptations)
        #and 1 followed by L zeros (cooperator with 0 adaptations)
        #Note 1 is added to all base 10 numbers
        self.max_types = config['Simulation']['max_types']
        self.genome_length = self.config['Population']['genome_length']
        self.type2geno = zeros(self.max_types, dtype=nint32)
        self.type2geno[0] = 1
        self.type2geno[1] = 1+int("1"+"0"*self.genome_length, 2)
        self.social_types = zeros(self.max_types, dtype=nint32)
        self.social_types[1] = 1
        self.fitness_landscape = zeros(self.max_types)
        self.fitness_landscape[0] = self.calculate_fitness(self.type2geno[0])
        self.fitness_landscape[1] = self.calculate_fitness(self.type2geno[1])
        self.current_type = 2


        #GKD: initialize mutation_probs
        self.mutation_rate_social = self.config['Population']['mutation_rate_social']
        self.mutation_rate_adaptation = self.config['Population']['mutation_rate_adaptation']

        hd = zeros((self.max_types,self.max_types), dtype=nint32)
        social_hd = zeros((self.max_types,self.max_types), dtype=nint32)
        nonsocial_hd = zeros((self.max_types,self.max_types), dtype=nint32)
        hd[0,1] = 1
        hd[1,0] = 1
        social_hd[0,1] = 1
        social_hd[1,0] = 1

        self.mutate_to_self_prob = ((1 - self.mutation_rate_adaptation) ** self.genome_length) * (1 - self.mutation_rate_social)

        npower = np.power
        mr = npower(1-self.mutation_rate_adaptation, self.genome_length-nonsocial_hd[:self.current_type,:self.current_type]) *\
                npower(self.mutation_rate_adaptation, nonsocial_hd[:self.current_type,:self.current_type]) *\
                npower(1-self.mutation_rate_social, social_hd[:self.current_type,:self.current_type]==0) *\
                npower(self.mutation_rate_social, social_hd[:self.current_type,:self.current_type])

        self.mutation_probs = np.zeros((self.max_types,self.max_types+1))
        self.mutation_probs[:self.current_type, :self.current_type] = mr

        # Store the probabilities of mutations between all pairs of genotypes
        #self.mutation_probs = self.get_mutation_probabilities()

        # Create the fitness landscape
        #self.fitness_landscape = self.build_fitness_landscape()

        initial_state = self.config['Metapopulation']['initial_state']
        max_cap = self.config['Population']['capacity_max']
        min_cap = self.config['Population']['capacity_min']
        initial_producer_proportion = self.config['Population']['initial_producer_proportion']
        mode = self.config['Simulation']['mode']


        # Create each of the populations

        for n, d in list(self.topology.nodes(data=True)):
            d['population'] = Population(metapopulation=self, config=config)

            if initial_state == 'corners':
                # Place all producers in one corner and all non-producers in
                # the other
                if n == 0:
                    d['population'].abundances[1] = max_cap if mode == 'cooperation' else min_cap
                    d['population'].dilute()
                elif n == len(self.topology)-1:
                    d['population'].abundances[0] = min_cap
                    d['population'].dilute()

            elif initial_state == 'stress':
                cap = int(min_cap + ( (max_cap - min_cap) * initial_producer_proportion)) if mode == 'cooperation' else min_cap
                num_producers = int(cap * initial_producer_proportion)
                num_nonproducers = cap - num_producers

                d['population'].abundances[0] = num_nonproducers
                d['population'].abundances[1] = num_producers
                d['population'].bottleneck(survival_rate=self.config['Population']['stress_survival_rate'])
                
            elif initial_state == 'even_split':
                if n < len(self.topology)*initial_producer_proportion:
                    d['population'].abundances[0] = 1
                else:
                    d['population'].abundances[1] = 1


        # How frequently should the metapopulation be mixed?
        self.metapopulation_mixing = self.config['MetapopulationMixing']['enabled']
        self.mix_frequency = self.config['MetapopulationMixing']['frequency']


        # Does the environment change? If so, how...
        self.environment_changes = self.config['EnvironmentalChange']['enabled']
        self.env_change_frequency = self.config['EnvironmentalChange']['frequency']
        self.environment_changed = False

        if self.environment_changes:
            self.set_next_environment_change()


        data_dir = self.config['Simulation']['data_dir']

        # log_objects is a list of any logging objects used by this simulation
        self.log_objects = []

        if self.config['MetapopulationLog']['enabled']:
            fname = self.config['MetapopulationLog']['filename']
            freq = self.config['MetapopulationLog']['frequency']
            compress = self.config['MetapopulationLog']['compress']
            self.log_objects.append((freq, MetapopulationOutput(metapopulation=self,
                                                                filename=os.path.join(data_dir, fname),
                                                                header=True,
                                                                include_uuid=self.config['MetapopulationLog']['include_uuid'],
                                                                compress=compress)))

        if self.config['PopulationLog']['enabled']:
            fname = self.config['PopulationLog']['filename']
            freq = self.config['PopulationLog']['frequency']
            compress = self.config['PopulationLog']['compress']
            self.log_objects.append((freq, PopulationOutput(metapopulation=self,
                                                            filename=os.path.join(data_dir, fname),
                                                            header=True,
                                                            include_uuid=self.config['PopulationLog']['include_uuid'],
                                                            compress=compress)))

        if self.config['GenotypeLog']['enabled']:
            fname = self.config['GenotypeLog']['filename']
            freq = self.config['GenotypeLog']['frequency']
            compress = self.config['GenotypeLog']['compress']
            self.log_objects.append((freq, GenotypesOutput(metapopulation=self,
                                                           filename=os.path.join(data_dir, fname),
                                                           header=True,
                                                           include_uuid=self.config['GenotypeLog']['include_uuid'],
                                                           compress=compress)))

        if self.config['FitnessLog']['enabled'] :
            fname = self.config['FitnessLog']['filename']
            freq = self.config['FitnessLog']['frequency']
            compress = self.config['FitnessLog']['compress']
            self.log_objects.append((freq, FitnessOutput(metapopulation=self,
                                                         filename=os.path.join(data_dir, fname),
                                                         header=True,
                                                         include_uuid=self.config['FitnessLog']['include_uuid'],
                                                         compress=compress)))

        if self.config['EnvChangeLog']['enabled']:
            fname = self.config['EnvChangeLog']['filename']
            freq = self.config['EnvChangeLog']['frequency']
            compress = self.config['EnvChangeLog']['compress']
            self.log_objects.append((freq, EnvChangeOutput(metapopulation=self,
                                                           filename=os.path.join(data_dir, fname),
                                                           header=True,
                                                           include_uuid=self.config['EnvChangeLog']['include_uuid'],
                                                           compress=compress)))


    def __repr__(self):
        """Return a string representation of the Metapopulation object"""
        prop_producers = self.prop_producers()

        if prop_producers == 'NA':
            res = "Metapopulation: Size {s}, NA% producers".format(s=self.size())
        else:
            maxfit = self.max_fitnesses()
            maxfit_p = max(maxfit[0]) / max(self.fitness_landscape)
            maxfit_np = max(maxfit[1]) / max(self.fitness_landscape)
            #res = "Metapopulation: Size {s}, {p:.1%} producers".format(s=self.size(),
            #                                                        p=self.prop_producers())

            if maxfit_p > maxfit_np:
                symbol = '>'
            elif maxfit_p < maxfit_np:
                symbol = '<'
            else:
                symbol = '='

            res = "Metapopulation: Size {s}, {p:.1%} producers. w(P): {mp:.2} "\
                  "{sym} w(Np): {mnp:.2}.".format(s=self.size(), p=self.prop_producers(),
                                                 mp=maxfit_p, mnp=maxfit_np, sym=symbol)

        return res


#    def build_fitness_landscape(self):
#        """Build a fitness landscape
#
#        """
#
#        genome_length = self.config['Population']['genome_length']
#        base_fitness = self.config['Population']['base_fitness']
#        production_cost = self.config['Population']['production_cost']
#        benefit_nonzero = self.config['Population']['benefit_nonzero']
#        fitness_shape = self.config['Population']['fitness_shape']

#        landscape = zeros(2**(genome_length))

#        for i in range(2**(genome_length)):
#            num_ones = sum(genome.base10_as_bitarray(i))
#            landscape[i] = base_fitness + (benefit_nonzero * (num_ones**fitness_shape))

#        return np.append(landscape, landscape - production_cost)

    def calculate_fitness(self, genotype_base10):
        """Calculate the fitness of an individual from its genotype in base 2
        Allows for a custom fitness function as specified in config, with the
        default fitness function being a single-peaked landscape with exponential steps

        """
        fitness_function = self.config['Population']['fitness_function']
        genome_length = self.config['Population']['genome_length']
        #base_fitness = self.config['Population']['base_fitness']
        production_cost = self.config['Population']['production_cost']
        benefit_nonzero = self.config['Population']['benefit_nonzero']
        #fitness_shape = self.config['Population']['fitness_shape']
        smooth_scaling = self.config['Population']['smooth_scaling']
        mode = self.config['Simulation']['mode']

        if fitness_function != None and not callable(fitness_function):
            raise Exception("fitness_function in config['Population']['fitness_function'] must be callable")

        genotype = np.binary_repr(genotype_base10 - 1, genome_length + 1)

        if mode == 'cooperation':
        #Default fitness function (single peaked landscape)
            if fitness_function == None:
                nonzero_loci = np.nonzero([int(i) for i in [*genotype[1:]]])[0]
                fitness = benefit_nonzero**((np.sum(nonzero_loci)+len(nonzero_loci))*smooth_scaling)
                if genotype[0] == "0":
                    return(fitness)
                    #return((base_fitness + (benefit_nonzero * (num_ones**fitness_shape))))
                else:
                    return(fitness*(1-production_cost))
                    #return((base_fitness + (benefit_nonzero * (num_ones**fitness_shape))) - production_cost)

            #Custom fitness function
            else:
                if genotype[0] == "0":
                    return(fitness_function(genotype_base10 - 1))
                else:
                    return(fitness_function(genotype_base10 - (2**genome_length) - 1)*(1-production_cost)) #Only compute on adaptive loci

        #NEW: adding learning
        elif mode == 'learning':
            if fitness_function == None:
                nonzero_loci = np.nonzero([int(i) for i in [*genotype[1:]]])[0]
                fitness_current_genotype = benefit_nonzero**((np.sum(nonzero_loci)+len(nonzero_loci))*smooth_scaling)
                if genotype[0] == "0":
                    return(fitness_current_genotype)
                else:
                    # Iterate through all bit flips
                    max_fitness = fitness_current_genotype
                    for i in range(len(genotype) - 1):
                        flipped_genotype = genotype[0:i+1] + "1" + genotype[i+2:] if genotype[i+1] == "0" else genotype[0:i+1] + "0" + genotype[i+2:]
                        nonzero_loci = np.nonzero([int(i) for i in [*flipped_genotype[1:]]])[0]
                        fitness_flipped_bit = benefit_nonzero**((np.sum(nonzero_loci)+len(nonzero_loci))*smooth_scaling)

                        # If the sampled mutant has the highest seen fitness, copy the fitness of that genotype
                        if fitness_flipped_bit > max_fitness:
                            max_fitness = fitness_flipped_bit

                    return max_fitness * (1 - production_cost)

            # Custom fitness function
            else:
                if genotype[0] == "0":
                    return (fitness_function(genotype_base10 - 1))
                else:
                    fitness_current_genotype = fitness_function(genotype_base10 - (2**genome_length) - 1)
                    max_fitness = fitness_current_genotype
                    for i in range(len(genotype) - 1):
                        flipped_genotype = genotype[0:i+1] + "1" + genotype[i+2:] if genotype[i+1] == "0" else genotype[0:i+1] + "0" + genotype[i+2:]
                        genotype_base10_flipped = int(flipped_genotype, 2)
                        fitness_flipped_bit = fitness_function(genotype_base10_flipped)

                        if fitness_flipped_bit > max_fitness:
                            max_fitness = fitness_flipped_bit

                    return max_fitness * (1 - production_cost)

    def insert_types(self, new_genotypes):
        """Add a new (set of) genotype(s) to the type-genotype mapping and fitness landscape,
        and get the new mutation probabilities to/from this genotype

        """
        for i in new_genotypes:
            self.type2geno[self.current_type] = i
            self.fitness_landscape[self.current_type] = self.calculate_fitness(i)

            #Calculate the new Hamming distances
            new_dist = np.array([genome.hamming_distance(j-1, i-1) for j in self.type2geno[:self.current_type]])

            cooperator_threshold = int("1"+"0"*self.genome_length, 2)

            self.social_types[self.current_type] = 1 if i > cooperator_threshold else 0
            
            new_socdist = np.array([1 if (j > cooperator_threshold and i <= cooperator_threshold) or (j <= cooperator_threshold and i > cooperator_threshold) else 0 for j in self.type2geno[:self.current_type]])
            new_nonsocdist = new_dist - new_socdist
            npower = np.power

            mr = npower(1 - self.mutation_rate_adaptation, self.genome_length - new_nonsocdist) * \
                  npower(self.mutation_rate_adaptation, new_nonsocdist) * \
                  npower(1 - self.mutation_rate_social, 1 - new_socdist) * \
                  npower(self.mutation_rate_social, new_socdist)

            self.mutation_probs[self.current_type, :self.current_type] = mr
            self.mutation_probs[:self.current_type, self.current_type] = mr.T
            self.mutation_probs[self.current_type, self.current_type] = self.mutate_to_self_prob

            self.current_type += 1

    def cleanup_types(self):
        """Remove genotypes from type2geno that are not currently present in the population.

        """

        #Find all genotypes still present in the population
        total_abundances = zeros(self.max_types, dtype=nuint32)
        for n, d in list(self.topology.nodes(data=True)):
            total_abundances = np.add(total_abundances, d['population'].abundances)

        nonzero_indices = np.nonzero(total_abundances)[0]

        #Ensure that both all-0s genotypes are kept (for resetting the environment)
        if not np.in1d(0, nonzero_indices) and not np.in1d(1, nonzero_indices):
            nonzero_indices = np.insert(nonzero_indices, 0, [0, 1])

        elif not np.in1d(0, nonzero_indices):
            nonzero_indices = np.insert(nonzero_indices, 0, 0)

        elif not np.in1d(1, nonzero_indices):
            nonzero_indices = np.insert(nonzero_indices, 1, 1)

        #Restructure population abundances
        self.current_type = len(nonzero_indices)
        for n, d in list(self.topology.nodes(data=True)):
            new_abundances = np.zeros(self.max_types, dtype=nuint32)
            new_abundances[:self.current_type] = d['population'].abundances[nonzero_indices]
            d['population'].abundances = new_abundances

        #Restructure fitness landscape, type2geno, and mutation_probs
        temp_fitness_landscape = zeros(self.max_types)
        temp_fitness_landscape[:self.current_type] = self.fitness_landscape[nonzero_indices]
        self.fitness_landscape = temp_fitness_landscape

        temp_type2geno = zeros(self.max_types, dtype=nint32)
        temp_type2geno[:self.current_type] = self.type2geno[nonzero_indices]
        self.type2geno = temp_type2geno

        temp_mutation_probs = np.zeros((self.max_types,self.max_types+1))
        x,y = np.meshgrid(nonzero_indices, nonzero_indices, indexing='ij')
        temp_mutation_probs[:self.current_type, :self.current_type] = self.mutation_probs[x,y]
        self.mutation_probs = temp_mutation_probs





#    def get_mutation_probabilities(self):
#        """Get a table of probabilities among all pairs of genotypes
#
#        This works by first generating the Hamming distances between all of the
#        possible genotypes. These distances are then used to calculate the
#        probabilities of mutating by:
#
#            (1-mu)^(#matching bits) * mu^(#different bits)
#
#        Where #matching bits is the genome length - Hamming distance and
#        #different bits is the Hamming distance.
#
#        """
#
#        genome_length = self.config['Population']['genome_length']
#        mutation_rate_social = self.config['Population']['mutation_rate_social']
#        mutation_rate_adaptation = self.config['Population']['mutation_rate_adaptation']
#
#        S = np.vstack((np.array([[0]*2**genome_length + [1]*2**genome_length]).repeat(repeats=2**genome_length, axis=0),
#                       np.array([[1]*2**genome_length + [0]*2**genome_length]).repeat(repeats=2**genome_length, axis=0)))
#
#
#        # TODO: to handle things like P->NP but not NP->P (or vice versa), just
#        # manipulate the mr vector.
#
#        # Get the pairwise Hamming distance for all genotypes
#        hamming_v = np.vectorize(genome.hamming_distance)
#        genotypes = np.arange(start=0, stop=2**(genome_length+1))
#        xx, yy = np.meshgrid(genotypes, genotypes)
#        hamming_distances = hamming_v(xx, yy)
#
#        # nonsocial_hd is a matrix containing the pairwise Hamming distances
#        # between all genomes considering only the non-social loci
#        nonsocial_hd = hamming_distances - S
#
#        # mr is a matrix where each element contains the probability of mutating
#        # from one genome to the other.
#        # mr = npower(1-m1, L-NS) * npower(m1, NS) * npower(1-m2, S2) * npower(m2, S)
#
#        npower = np.power
#        mr = npower(1-mutation_rate_adaptation, genome_length-nonsocial_hd) *\
#                npower(mutation_rate_adaptation, nonsocial_hd) *\
#                npower(1-mutation_rate_social, S==0) *\
#                npower(mutation_rate_social, S)
#
#        return mr


    def dilute(self):
        """Dilute the metapopulation

        Dilute the metapopulation by diluting each population by the dilution
        factor specified with the dilution_factor option in the Population
        section of the configuration file.

        """
        for n, d in list(self.topology.nodes(data=True)):
            d['population'].dilute()


    def mix(self):
        """Mix the population

        Mix the population. The abundances at all populations are combined and
        re-distributed.
        """

        abundances = zeros(self.max_types, dtype=np.int)

        for n, d in list(self.topology.nodes(data=True)):
            abundances += d['population'].abundances

        for n, d in list(self.topology.nodes(data=True)):
            d['population'].abundances = binomial(abundances, 1.0/len(self.topology))


    def grow(self):
        """Grow the metapopulation ...."""
        for n, d in list(self.topology.nodes(data=True)):
            d['population'].grow()


    def mutate(self):
        """Mutate the metapopulation ...."""
        for n, d in list(self.topology.nodes(data=True)):
            #GKD: mutate each population, and if there are novel mutations not currently in type2geno, add them
            new_genos = d['population'].mutate()
            if new_genos != None:
                self.insert_types(new_genos)

    def migrate(self, single_dest=True):
        """Migrate individuals among the populations
        
        * single_dest: if True (default), all migrants will go to a single
            neighbor population. Otherwise, migrants will be distributed among
            all neighbors. 
        
        """
        if self.migration_rate == 0 or self.topology.number_of_nodes() < 2:
            return

        for n, d in list(self.topology.nodes(data=True)):
            pop = d['population']

            if self.topology.degree(n) == 0:
                return

            # Migrate everything to one neighboring population
            if self.migration_dest.lower() == 'single':
                migrants = pop.select_migrants(migration_rate=self.migration_rate)

                if binomial(n=1, p=self.migration_p_far, size=None) == 1:
                    neighbor_index = random_integers(low=0,high=self.topology.number_of_nodes()-1)
                else:
                    neighbor_index = nchoice(list(self.topology.neighbors(n)))

                neighbor = self.topology.nodes[neighbor_index]['population']
                neighbor.add_immigrants(migrants)
                pop.remove_emigrants(migrants)

            # Distribute the migrants among the neighboring populations
            elif self.migration_dest.lower() == 'neighbors':
                num_neighbors = self.topology.degree(n)
                for neighbor_node in self.topology.neighbors(n):
                    if binomial(n=1, p=self.migration_p_far, size=None) == 1: 
                        neighbor_node = random_integers(low=0,high=self.topology.number_of_nodes()-1)
                    migrants = pop.select_migrants(migration_rate=self.migration_rate/num_neighbors)
                    neighbor = self.topology.nodes[neighbor_node]['population']
                    neighbor.add_immigrants(migrants)
                    pop.remove_emigrants(migrants)


    def census(self):
        """Update each population's abundance to account for migration"""
        for n, d in list(self.topology.nodes(data=True)):
            d['population'].census()


    def cycle(self):
        """Cycle the metapopulation

        In each cycle, the metapopulation cycles its state by diluting each
        population, allowing each population to grow to capacity, mutate each
        population, and then migrating among populations.

        """
        #start_time = datetime.now()
        self.write_logfiles()
        #end_time = datetime.now()
        #print("Log time: ", str(end_time - start_time))

        self.grow()

        #start_time = datetime.now()
        self.mutate()
        #end_time = datetime.now()
        #print("Mutate time: ", str(end_time - start_time))

        self.migrate()
        self.census()

        #If type2geno is at least 95% full, remove unused types
        if self.current_type >= 0.95*self.max_types:
            old_type = self.current_type
            self.cleanup_types()
            print("Cleaned types! Removed", str(old_type-self.current_type),"unused types.")

        if self.metapopulation_mixing and self.time > 0 and \
                (self.time % self.mix_frequency == 0):
            self.mix()

        if self.environment_changes and self.time == self.next_env_change_cycle:
            self.change_environment()
            self.environment_changed = True
            self.set_next_environment_change()
        else:
            self.dilute()
            self.environment_changed = False

        self.time += 1


    def change_environment(self):
        """Change the environment

        The change_environment function changes the environment for the
        metapopulation. This re-generates the fitness landscape and zeros out
        all fitness-encoding loci. This is meant to represent the metapopulation
        being subjected to different selective pressures. The number of
        individuals of each genotype that survive this event are proportional to
        the abundance of that genotype times the mutation rate (representing
        individuals that acquired the mutation that allows them to persist).
        """

        #self.fitness_landscape = self.build_fitness_landscape()

        for n, d in list(self.topology.nodes(data=True)):
            #d['population'].bottleneck(survival_rate=self.config['Population']['stress_survival_rate'])
            d['population'].reset_loci()


    def set_next_environment_change(self):
        """Set the cycle at which the next environmental change will occur
        """
        if self.config['EnvironmentalChange']['type'] == 'regular':
            self.next_env_change_cycle = self.time + self.env_change_frequency
        elif self.config['EnvironmentalChange']['type'] == 'exponential':
            self.next_env_change_cycle = self.time + np.round(exponential(scale=self.env_change_frequency, size=1)[0]).astype(int)


    def size(self):
        """Return the size of the metapopulation

        The size of the metapopulation is the sum of the sizes of the
        subpopulations
        """
        return sum(len(d['population']) for n, d in list(self.topology.nodes(data=True)))


    def __len__(self):
        """Return the length of a Metapopulation

        We'll define the length of a metapopulation as its size, so len(metapop)
        returns the number of individuals in all populations of Metapopulation
        metapop
        """
        return self.size()


    def num_producers(self):
        """Return the number of producers in the metapopulation"""
        return sum(d['population'].num_producers() for n, d in list(self.topology.nodes(data=True)))

    def num_nonproducers(self):
        """Return the number of producers in the metapopulation"""
        return sum(d['population'].num_nonproducers() for n, d in list(self.topology.nodes(data=True)))


    def prop_producers(self):
        """Get the proportion of producers in the metapopulation"""
        metapopsize = self.size()

        if metapopsize == 0:
            return 'NA'
        else:
            return 1.0 * self.num_producers() / self.size()


    def max_fitnesses(self):
        """Get the maximum fitness among producers and non-producers"""

        prod_max = [d['population'].max_fitnesses()[0] for n, d in list(self.topology.nodes(data=True))]
        nonprod_max = [d['population'].max_fitnesses()[1] for n, d in list(self.topology.nodes(data=True))]

        return (prod_max, nonprod_max)
        
    def average_fitness(self):
        """Get the average fitness of the metapopulation"""
        landscape = self.fitness_landscape
    
        popsize = self.size()
        fitnesses = np.sum([d['population'].abundances*landscape for n, d in list(self.topology.nodes(data=True))])

        if popsize == 0:
            return 'NA'
        else:
            return fitnesses/popsize
        
    def average_delta_fitness(self):
        """Get the average fitness of cooperators divided by the average fitness of defectors"""
        landscape = self.fitness_landscape

        avg_fitnesses_prod = np.sum([d['population'].abundances[self.social_types == 1] * landscape[self.social_types == 1] for n, d in list(self.topology.nodes(data=True))])/self.num_producers()
        avg_fitnesses_nonprod = np.sum([d['population'].abundances[self.social_types == 0] * landscape[self.social_types == 0] for n, d in list(self.topology.nodes(data=True))])/self.num_nonproducers()

        return avg_fitnesses_prod/avg_fitnesses_nonprod

    def write_logfiles(self):
        """Write any log files"""

        for (freq, l) in self.log_objects:
            if self.time % freq == 0:
                l.update(time=self.time)


    def cleanup(self):
        for (freq, l) in self.log_objects:
            l.close()

