# small-network

Evolving spiking neural networks for ambient music generation.

## Quick Start

```bash
# Fresh run (50 generations)
python evolve.py --generations 50

# Resume from checkpoint
python evolve.py --resume evolve_midi/<run>/checkpoint.pkl --generations 20
```

## Evolution Parameters

| Parameter                  | Default | Description                                   |
| -------------------------- | ------- | --------------------------------------------- |
| `mu`                       | 20      | Number of parents (survivors each generation) |
| `lambda_`                  | 100     | Number of offspring per generation            |
| `generations`              | 50      | Total generations to run                      |
| `save_every_n_generations` | 5       | Save MIDI + checkpoint every N generations    |

### Mutation Rates

| Parameter                  | Default | Description                                |
| -------------------------- | ------- | ------------------------------------------ |
| `weight_mutation_rate`     | 0.1     | Probability of mutating each weight        |
| `weight_mutation_scale`    | 0.1     | Std dev of weight perturbations            |
| `threshold_mutation_rate`  | 0.1     | Probability of mutating neuron thresholds  |
| `threshold_mutation_scale` | 0.1     | Std dev of threshold perturbations         |
| `refraction_mutation_rate` | 0.05    | Probability of mutating refractory periods |

### Simulation

| Parameter    | Default       | Description                     |
| ------------ | ------------- | ------------------------------- |
| `sim_steps`  | 256           | Timesteps per simulation        |
| `tempo`      | 60            | BPM for MIDI output             |
| `base_notes` | [48,60,72,84] | Root notes for 4 voices (C3-C6) |

## Output

Each run creates `evolve_midi/<timestamp>/` containing:

- `gen###_best_*.mid` - Best MIDI every 5 generations
- `checkpoint.pkl` - Resumable state
- `final_best_*.mid` - Best overall
- `evolution_history.png` - Fitness plots
