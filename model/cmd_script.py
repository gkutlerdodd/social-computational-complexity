#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# Script Name: cmd_script.py
# Description: X
#
# Output: X
#
# Author: Griffin Kutler Dodd, adapted from Hammarlund et al. (2016)
#
# Date Created: 2025-02-03
# Last Modified: 2026-06-22
#
# Requirements:
#   - This script is designed to be run using Python 3.13.5
#
# Usage: X
#
# ------------------------------------------------------------------------------


import numpy as np
import argparse
import os
import sys
from datetime import datetime, timedelta
from time import time
from uuid import uuid4
from warnings import warn

###################
##Fitness functions
###################

def fitness_function_hard(x, genome_size=0, c=2):
    """Hard fitness landscape from Kaznatcheev (2019)."""

    s_plus = 2
    s_minus = 1

    if genome_size < 1:
        raise ValueError("genome_size must be at least 1.")

    #Base cases
    if genome_size == 1:
        return 2 * x + c

    if genome_size == 2:
        return (4 + c) if x == 3 else (x + c)

    #Recursive landscape construction
    a = (x & 2) >> 1 #Second least significant bit
    b = x & 1 #Least significant bit
    z = x >> 2
    z_star = 1 if (genome_size - 2 == 1) else 3 #Peak of the sublandscape
    f_z = fitness_function_hard(z, genome_size - 2)
    f_z_star = fitness_function_hard(z_star, genome_size - 2)

    if (a == b) and (b == 0):
        return f_z
    elif (a != b) and (z != z_star):
        return f_z + s_minus
    elif (a == 0) and (b == 1) and (z == z_star):
        return f_z_star + s_minus
    elif (a == 1) and (b == 0) and (z == z_star):
        return f_z_star + s_plus
    elif (a == b) and (b == 1):
        return fitness_function_hard(z ^ z_star, genome_size - 2, c=0) + f_z_star + 2 * s_plus
    else:
        raise RuntimeError("Error when computing fitness_function_hard.")

def fitness_function_hard_exp(x, genome_size=0, m=2):
    """Exponentially-scaled hard fitness function from Kaznatcheev (2019), acting on m bits."""

    s_plus = 1.5
    s_minus = 0.5

    if genome_size < 1:
        raise ValueError("genome_size must be at least 1.")
    if m < 2:
        raise ValueError("m (buffer_bits) must be at least 2.")

    #Base case
    if genome_size <= m:
        return s_plus ** bin(x).count("1")

    #Recursive construction
    y = np.binary_repr(x)[-m:] #m least significant bits
    num_ones = y.count("1")
    z = x >> m
    z_star = int("1" * m, 2) if (genome_size - m >= m) else int("1" * (genome_size - m), 2) #Peak of the sublandscape
    f_z = fitness_function_hard_exp(z, genome_size - m, m)
    f_z_star = fitness_function_hard_exp(z_star, genome_size - m, m)

    if num_ones == 0:
        return f_z
    elif 0 < num_ones < m and z != z_star:
        return s_minus * f_z
    elif 0 < num_ones < m and z == z_star:
        return (s_plus ** num_ones) * f_z_star
    elif num_ones == m:
        return (s_plus ** m) * fitness_function_hard_exp(z ^ z_star, genome_size - m, m) * f_z_star
    else:
        raise RuntimeError("Error when computing fitness_function_hard_exp.")

###################
##Config helpers
###################
def _compute_smooth_scaling(genome_length, buffer_bits):
    """Compute the scaling factor for the maximal fitness on a smooth landscape to
    match that of the size-equivalent hard landscape."""

    num_steps = buffer_bits * (2 ** (genome_length / buffer_bits) - 1)
    return (num_steps * 2) / (genome_length * (genome_length + 1))

def _resolve_fitness_function(landscape, genome_length, buffer_bits):
    """Return the callable fitness function for a named landscape (None for smooth)."""

    name = (landscape or "").lower()
    if name in ("smooth", "none", ""):
        return None
    elif name == "hard-additive":
        return lambda x: fitness_function_hard(x, genome_length)
    elif name == "hard-exponential":
        return lambda x: fitness_function_hard_exp(x, genome_length, buffer_bits)
    else:
        raise ValueError("Unknown landscape '{}'. Choose: smooth, hard-additive, or hard-exponential".format(landscape))

def _make_config(args, seed, uuid, data_dir):
    """Build a config dictionary from parsed CLI args."""

    genome_length = args.genome_length
    buffer_bits = args.buffer_bits
    smooth_scaling = _compute_smooth_scaling(genome_length, buffer_bits)
    fitness_fn = _resolve_fitness_function(args.landscape, genome_length, buffer_bits)
    prefix = getattr(args, 'filename_prefix', '')

    return {
        'Simulation': {
            'cycles':          args.cycles,
            'max_types':       args.max_types,
            'stop_when_empty': not args.no_stop_when_empty,
            'data_dir':        data_dir,
            'mode':            args.mode,
            'seed':            seed,
            'UUID':            uuid,
        },
        'Metapopulation': {
            'migration_rate':  args.migration_rate,
            'migration_dest':  args.migration_dest,
            'migration_p_far': args.migration_p_far,
            'topology':        args.topology,
            'export_topology': args.export_topology,
            'initial_state':   args.initial_state,
        },
        'CompleteTopology': {
            'size': args.topology_size,
        },
        'MooreTopology': {
            'width':    args.topology_width,
            'height':   args.topology_height,
            'radius':   args.topology_radius,
            'periodic': args.periodic,
        },
        'VonNeumannTopology': {
            'width':    args.topology_width,
            'height':   args.topology_height,
            'periodic': args.periodic,
        },
        'SmallWorldTopology': {
            'size':      args.topology_size,
            'neighbors': args.sw_neighbors,
            'edgeprob':  args.sw_edgeprob,
            'seed':      seed,
        },
        'RegularTopology': {
            'size':   args.topology_size,
            'degree': args.reg_degree,
            'seed':   seed,
        },
        'Population': {
            'genome_length':               genome_length,
            'mutation_rate_social':        args.mutation_rate_social,
            'mutation_rate_adaptation':    args.mutation_rate_adaptation,
            'stress_survival_rate':        args.stress_survival_rate,
            'dilution_factor':             args.dilution_factor,
            'dilution_prob_min':           args.dilution_prob_min,
            'capacity_min':                args.capacity_min,
            'capacity_max':                args.capacity_max,
            'capacity_shape':              args.capacity_shape,
            'initial_producer_proportion': args.initial_producer_proportion,
            'production_cost':             args.production_cost,
            'initialize':                  'empty',
            'base_fitness':                1.0,
            'benefit_nonzero':             args.benefit_nonzero,
            'fitness_shape':               1.0,
            'fitness_function':            fitness_fn,
            'buffer_bits':                 buffer_bits,
            'smooth_scaling':              smooth_scaling,
        },
        'EnvironmentalChange': {
            'enabled':   args.env_change,
            'type':      args.env_change_type,
            'frequency': args.env_change_frequency,
        },
        'MetapopulationMixing': {
            'enabled':   args.mixing,
            'frequency': args.mixing_frequency,
        },
        'MetapopulationLog': {
            'enabled':      True,
            'frequency':    args.metapop_log_freq,
            'filename':     prefix + 'metapopulation.csv',
            'compress':     False,
            'include_uuid': True,
        },
        'PopulationLog': {
            'enabled':      args.population_log,
            'frequency':    args.population_log_freq,
            'filename':     prefix + 'population.csv',
            'compress':     False,
            'include_uuid': True,
        },
        'FitnessLog': {
            'enabled':      args.fitness_log,
            'frequency':    1,
            'filename':     prefix + 'fitness.csv',
            'compress':     False,
            'include_uuid': False,
        },
        'GenotypeLog': {
            'enabled':      args.genotype_log,
            'frequency':    1,
            'filename':     prefix + 'genotypes.csv',
            'compress':     False,
            'include_uuid': False,
        },
        'EnvChangeLog': {
            'enabled':      args.env_change,
            'frequency':    1,
            'filename':     prefix + 'environmental_change.csv',
            'compress':     False,
            'include_uuid': False,
        },
    }

def _prepare_data_dir(data_dir):

    """Create data_dir, renaming with a timestamp if it already exists."""
    if os.path.exists(data_dir):
        newname = '{}-{}'.format(data_dir, datetime.now().strftime('%Y%m%d%H%M%S'))
        warn('{} already exists. Using {} instead.'.format(data_dir, newname))
        data_dir = newname
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def _write_info(data_dir, config, seed, mode='w'):
    """Write a run-info file."""
    path = os.path.join(data_dir, 'info.txt')
    with open(path, mode) as f:
        f.write('Generated at {}\n'.format(datetime.now().isoformat()))
        f.write('Random Seed: {}\n'.format(seed))
        f.write('UUID: {}\n'.format(config['Simulation']['UUID']))
        f.write('Mode: {}\n'.format(config['Simulation']['mode']))
        f.write('Landscape: {}\n'.format('smooth' if config['Population']['fitness_function'] is None else 'custom'))
        f.write('Command: {}\n'.format(' '.join(sys.argv)))
    return path

def _run_simulation(config, quiet):
    from Metapopulation import Metapopulation
    m = Metapopulation(config=config)
    run_start = time()

    for t in range(config['Simulation']['cycles']):
        m.cycle()
        if not quiet:
            print("[{t}] {m}".format(t=t, m=m))
        if config['Simulation']['stop_when_empty'] and m.size() == 0:
            break

    m.write_logfiles()
    m.cleanup()

    return timedelta(seconds=time() - run_start)


def parse_arguments():
    p = argparse.ArgumentParser(
        prog='hankshaw',
        description='Run a Hankshaw/Baldwin effect simulation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #Simulation parameters
    sim = p.add_argument_group('Simulation')
    sim.add_argument('--cycles', '-n', type=int, default=2500,
                     help='Number of simulation cycles.')
    sim.add_argument('--max-types', type=int, default=1000, dest='max_types',
                     help='Maximum number of distinct genotypes tracked simultaneously.')
    sim.add_argument('--mode', choices=['cooperation', 'learning'],
                     default='cooperation', help='Simulation mode.')
    sim.add_argument('--seed', '-s', type=int, default=None,
                     help='RNG seed.')
    sim.add_argument('--data-dir', '-d', metavar='DIR', dest='data_dir',
                     default=None, help='Output directory. Default: data/<UUID>.')
    sim.add_argument('--no-stop-when-empty', action='store_true', default=False,
                     dest='no_stop_when_empty', help='Continue running even if the metapopulation goes extinct.')
    sim.add_argument('--quiet', '-q', action='store_true', default=False,
                     help='Suppress per-cycle status messages.')

    #Paired runs
    pair = p.add_argument_group('Paired runs')
    pair.add_argument('--paired', action='store_true', default=False,
                      help='Run smooth then hard landscapes back-to-back, '
                           'sharing one UUID and one output directory. '
                           'Output files are prefixed smooth- / hard- automatically.')

    #Landscape parameters
    land = p.add_argument_group('Fitness landscape')
    land.add_argument('--landscape', default='smooth',
                      choices=['smooth', 'hard-additive', 'hard-exponential'],
                      help='Choice of fitness function. Defaults to a smooth landscape,'
                           'and can choose from hard_additive and hard_exponential for'
                           'computionally complex landscapes.')
    land.add_argument('--genome-length', type=int, default=15,
                      dest='genome_length', help='Number of adaptive (non-social) loci.')
    land.add_argument('--buffer-bits', type=int, default=3,
                      dest='buffer_bits', help='Block size m used by hard landscapes.')
    land.add_argument('--benefit-nonzero', type=float, default=1.5,
                      dest='benefit_nonzero', help='Base of the exponential fitness function (smooth landscape).')

    #Population parameters
    pop = p.add_argument_group('Population')
    pop.add_argument('--production-cost', type=float, default=0.1, dest='production_cost',
                     help='Fitness cost of being a cooperator/learner.')
    pop.add_argument('--mutation-rate-social', type=float, default=1e-5,
                     dest='mutation_rate_social', help='Mutation rate at the social locus.')
    pop.add_argument('--mutation-rate-adaptation', type=float, default=1e-5,
                     dest='mutation_rate_adaptation', help='Mutation rate at adaptive loci.')
    pop.add_argument('--stress-survival-rate', type=float, default=1e-5,
                     dest='stress_survival_rate',
                     help='Fraction surviving an environmental change event.')
    pop.add_argument('--dilution-factor', type=float, default=0.1,
                     dest='dilution_factor',
                     help='Fraction of the population surviving dilution each cycle.')
    pop.add_argument('--dilution-prob-min', type=float, default=1.0,
                     dest='dilution_prob_min',
                     help='Minimum probability that a patch is diluted (cooperation mode).')
    pop.add_argument('--capacity-min', type=int, default=800,
                     dest='capacity_min', help='Carrying capacity with no producers (Hankshaw).')
    pop.add_argument('--capacity-max', type=int, default=2000,
                     dest='capacity_max', help='Carrying capacity with all producers (Hankshaw).')
    pop.add_argument('--capacity-shape', type=float, default=1.0,
                     dest='capacity_shape', help='Shape exponent for producer-proportion  to capacity curve.')
    pop.add_argument('--initial-producer-proportion', type=float, default=0.5,
                     dest='initial_producer_proportion', help='Starting fraction of producers (stress / even_split states).')

    #Topology parameters
    topo = p.add_argument_group('Topology')
    topo.add_argument('--topology', default='complete', choices=['complete', 'moore', 'vonneumann', 'smallworld', 'regular'],
                      help='Population structure.')
    topo.add_argument('--topology-size', type=int, default=100,
                      dest='topology_size',  help='Number of nodes (complete / smallworld / regular).')
    topo.add_argument('--topology-width', type=int, default=25,
                      dest='topology_width', help='Grid width (moore / vonneumann).')
    topo.add_argument('--topology-height', type=int, default=25,
                      dest='topology_height', help='Grid height (moore / vonneumann).')
    topo.add_argument('--topology-radius', type=int, default=1,
                      dest='topology_radius', help='Interaction radius (moore only).')
    topo.add_argument('--periodic', action='store_true', default=False,
                      help='Periodic boundary conditions (moore / vonneumann).')
    topo.add_argument('--sw-neighbors', type=int, default=4,
                      dest='sw_neighbors', help='k nearest neighbours (smallworld).')
    topo.add_argument('--sw-edgeprob', type=float, default=0.1,
                      dest='sw_edgeprob', help='Edge rewiring probability (smallworld).')
    topo.add_argument('--reg-degree', type=int, default=4,
                      dest='reg_degree', help='Node degree (regular).')
    topo.add_argument('--export-topology', action='store_true', default=False,
                      dest='export_topology', help='Save the topology graph as topology.gml.')
    topo.add_argument('--initial-state',
                      choices=['even_split', 'stress', 'corners'],
                      default='even_split', dest='initial_state',
                      help='How initial individuals are distributed across patches.')

    #Migration parameters
    mig = p.add_argument_group('Migration')
    mig.add_argument('--migration-rate', type=float, default=0.05,
                     dest='migration_rate',
                     help='Fraction of each population that migrates per cycle.')
    mig.add_argument('--migration-dest', choices=['single', 'neighbors'],
                     default='single', dest='migration_dest',
                     help='Whether migrants go to one neighbour or spread across all.')
    mig.add_argument('--migration-p-far', type=float, default=0.0,
                     dest='migration_p_far', help='Probability of a long-range (random destination) migration.')

    #Environmental change parameters
    env = p.add_argument_group('Environmental change')
    env.add_argument('--env-change', action='store_true', default=False,
                     dest='env_change', help='Enable periodic environmental change events.')
    env.add_argument('--env-change-type', choices=['regular', 'exponential'],
                     default='regular', dest='env_change_type',
                     help='Whether change intervals are fixed or exponentially distributed.')
    env.add_argument('--env-change-frequency', type=int, default=1,
                     dest='env_change_frequency',
                     help='Mean number of cycles between environmental changes.')

    #Mixing parameters
    mix = p.add_argument_group('Metapopulation mixing')
    mix.add_argument('--mixing', action='store_true', default=False,
                     help='Enable periodic full mixing of the metapopulation.')
    mix.add_argument('--mixing-frequency', type=int, default=100,
                     dest='mixing_frequency',
                     help='Cycles between mixing events.')

    #Logging
    log = p.add_argument_group('Logging')
    log.add_argument('--metapop-log-freq', type=int, default=1,
                     dest='metapop_log_freq',
                     help='Write the metapopulation log every N cycles.')
    log.add_argument('--population-log', action='store_true', default=False,
                     dest='population_log',
                     help='Enable per-patch population log.')
    log.add_argument('--population-log-freq', type=int, default=10,
                     dest='population_log_freq',
                     help='Write the population log every N cycles.')
    log.add_argument('--fitness-log', action='store_true', default=False,
                     dest='fitness_log',
                     help='Enable max-fitness log.')
    log.add_argument('--genotype-log', action='store_true', default=False,
                     dest='genotype_log',
                     help='Enable genotype-abundance log.')
    log.add_argument('--filename-prefix', default='', dest='filename_prefix',
                     metavar='PREFIX',
                     help='Prefix prepended to all output CSV filenames.')

    return p.parse_args()

def main():
    args = parse_arguments()

    seed = args.seed if args.seed is not None else int(np.random.randint(low=0, high=np.iinfo(np.uint32).max))

    #If in paired mode, run smooth and hard landscape simulations
    if args.paired:
        shared_uuid = str(uuid4())
        base_dir = args.data_dir if args.data_dir else os.path.join('data', shared_uuid)
        base_dir = _prepare_data_dir(base_dir)

        for landscape, prefix in [('smooth', 'smooth-'), ('hard-exponential', 'hard-')]:
            args.landscape = landscape
            args.filename_prefix = prefix
            config = _make_config(args, seed=seed, uuid=shared_uuid, data_dir=base_dir)

            np.random.seed(seed)
            info_path = _write_info(base_dir, config, seed, mode='a')

            if not args.quiet:
                print('=== Running {} landscape (seed={}) ==='.format(landscape, seed))

            elapsed = _run_simulation(config, args.quiet)

            with open(info_path, 'a') as f:
                f.write('{} run time: {}\n'.format(landscape.capitalize(), elapsed))

            if not args.quiet:
                print('{} landscape done. Time: {}\n'.format(landscape.capitalize(), elapsed))

    else:
        uuid = str(uuid4())
        data_dir = args.data_dir if args.data_dir else os.path.join('data', uuid)
        data_dir = _prepare_data_dir(data_dir)

        config = _make_config(args, seed=seed, uuid=uuid, data_dir=data_dir)
        np.random.seed(seed)

        info_path = _write_info(data_dir, config, seed, mode='w')

        if not args.quiet:
            print('=== Running {} landscape (seed={}) ==='.format(args.landscape, seed))

        elapsed = _run_simulation(config, args.quiet)

        with open(info_path, 'a') as f:
            f.write('Run Time: {}\n'.format(elapsed))

        if not args.quiet:
            print('Done. Time: {}'.format(elapsed))

if __name__ == '__main__':
    main()