# Computational complexity maintains costly learning and cooperation*

## Contents

## Use guide
Simulations can be run using `cmd_script.py`. To run paired simulations on an easy and hard landscape, use:

```bash
python cmd_script.py --paired
```

## All options

### Simulation

| Flag | Default | Description |
|---|---|---|
| `--cycles`, `-n` | `2500` | Number of simulation cycles to run. |
| `--mode` | `cooperation` | Simulation mode: `cooperation` or `learning`. |
| `--seed`, `-s` | random | RNG seed. |
| `--data-dir`, `-d` | `data/<UUID>` | Directory for all output files. |
| `--no-stop-when-empty` | off | By default the run stops early if the population goes extinct; this flag disables that. |
| `--quiet`, `-q` | off | Suppress per-cycle status messages. |
| `--max-types` | `1000` | Maximum number of genotypes tracked simultaneously. |

### Paired runs

| Flag | Default | Description |
|---|---|---|
| `--paired` | off | Run smooth then hard landscapes back-to-back. Output files are prefixed `smooth-` / `hard-`.

### Fitness landscape

| Flag | Default | Description |
|---|---|---|
| `--landscape` | `smooth` | `smooth`, `hard-additive`, `hard-exponential`. |
| `--genome-length` | `15` | Number of adaptive (non-social) loci. |
| `--buffer-bits` | `3` | Block size used by `hard` and  landscapes. |
| `--benefit-nonzero` | `1.5` | Base fitness. |

### Population

| Flag | Default | Description |
|---|---|---|
| `--production-cost` | `0.1` | Fitness cost of being a producer (cooperation) or learner (learning). |
| `--mutation-rate-social` | `1e-5` | Mutation rate at the social locus. |
| `--mutation-rate-adaptation` | `1e-5` | Mutation rate at adaptive loci. |
| `--stress-survival-rate` | `1e-5` | Fraction of individuals surviving an environmental change event. |
| `--dilution-factor` | `0.1` | Fraction of each patch surviving dilution each cycle. |
| `--dilution-prob-min` | `1.0` | Minimum probability that a patch is diluted (cooperation mode only). |
| `--capacity-min` | `800` | Carrying capacity of a patch with no producer (cooperation mode). This is the patch size in learning mode. |
| `--capacity-max` | `2000` | Carrying capacity of a patch consisting entirely of producers. Ignored in learning mode. |
| `--capacity-shape` | `1.0` | Shape exponent of the producer-proportion to capacity curve. |
| `--initial-producer-proportion` | `0.5` | Starting fraction of producers, used by the `stress` and `even_split` initial states. |

### Topology

| Flag | Default | Description |
|---|---|---|
| `--topology` | `complete` | Population structure: `complete`, `moore`, `vonneumann`, `smallworld`, or `regular`. |
| `--topology-size` | `100` | Number of nodes — used by `complete`, `smallworld`, and `regular`. |
| `--topology-width` | `25` | Grid width — used by `moore` and `vonneumann`. |
| `--topology-height` | `25` | Grid height — used by `moore` and `vonneumann`. |
| `--topology-radius` | `1` | Interaction radius — used by `moore` only. |
| `--periodic` | off | Use periodic boundary conditions (`moore` / `vonneumann`). |
| `--sw-neighbors` | `4` | Number of nearest neighbours k (`smallworld`). |
| `--sw-edgeprob` | `0.1` | Edge rewiring probability (`smallworld`). |
| `--reg-degree` | `4` | Node degree (`regular`). |
| `--initial-state` | `even_split` | How individuals are seeded at the start: `even_split` — one individual per patch alternating producer/non-producer. `stress` — producers and non-producers mixed then bottlenecked. `corners` — all producers in one corner, non-producers in the other. |
| `--export-topology` | off | Save the graph structure as `topology.gml` in the output directory. |

### Migration

| Flag | Default | Description |
|---|---|---|
| `--migration-rate` | `0.05` | Fraction of each patch that migrates per cycle. |
| `--migration-dest` | `single` | `single` — all migrants go to one randomly chosen neighbour. `neighbors` — migrants are distributed across all neighbours. |
| `--migration-p-far` | `0.0` | Probability that a migration event goes to a random (non-neighbouring) patch instead. |

### Environmental change

| Flag | Default | Description |
|---|---|---|
| `--env-change` | off | Enable periodic environmental change events. When an event occurs, all adaptive loci are reset to zero. |
| `--env-change-type` | `regular` | `regular` — changes occur at a fixed interval. `exponential` — interval is drawn from an exponential distribution. |
| `--env-change-frequency` | `1` | Mean number of cycles between environmental changes. |

### Metapopulation mixing

| Flag | Default | Description |
|---|---|---|
| `--mixing` | off | Enable periodic full mixing: all patches are pooled and individuals redistributed. |
| `--mixing-frequency` | `100` | Cycles between mixing events. |

### Logging

All runs always write `metapopulation.csv` and `info.txt`.

| Flag | Default | Description |
|---|---|---|
| `--metapop-log-freq` | `1` | Write the metapopulation log every N cycles. |
| `--population-log` | off | Enable per-patch population log (`population.csv`). |
| `--population-log-freq` | `10` | Write the population log every N cycles. |
| `--fitness-log` | off | Enable max-fitness log (`fitness.csv`). |
| `--genotype-log` | off | Enable genotype-abundance log (`genotypes.csv`). |
| `--filename-prefix` | `""` | Prefix prepended to all CSV filenames (e.g. `--filename-prefix hard-` produces `hard-metapopulation.csv`). Set automatically in `--paired` mode. |

---

## Output files

All output is written to the directory specified by `--data-dir` (or
`data/<UUID>/` by default).

| File | Always written | Description |
|---|---|---|
| `info.txt` | ✓ | Run metadata: seed, UUID, platform, elapsed time. |
| `[prefix]metapopulation.csv` | ✓ | Per-cycle population size, cooperator proportion, average fitness, max producer/non-producer fitness, and average fitness ratio. |
| `[prefix]population.csv` | optional | Per-cycle, per-patch breakdown of size, producers, and genotypes present. |
| `[prefix]fitness.csv` | optional | Per-cycle maximum fitness among producers and non-producers. |
| `[prefix]genotypes.csv` | optional | Per-cycle average abundance of every genotype across all patches. |
| `[prefix]environmental_change.csv` | when `--env-change` | Flags which cycles experienced an environmental change event. |
| `topology.gml` | when `--export-topology` | Graph structure of the metapopulation. |

---


## License

This work is licensed under a [Creative Commons](http://creativecommons.org) [Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

