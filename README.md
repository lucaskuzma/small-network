# small-network

Evolving spiking neural networks for ambient music generation.

## Quick Start

```bash
# Fresh run with pitch encoding (default, 12 chromatic outputs per voice)
python evolve.py --generations 50

# Fresh run with motion encoding (8 outputs per voice)
python evolve.py --encoding motion --generations 50

# Resume from checkpoint
python evolve.py --resume evolve_midi/<run>/checkpoint.pkl --generations 20
```

## Output Encodings

### Pitch Encoding (default)

- **12 outputs per voice** — chromatic pitch classes (C, C#, D, ..., B)
- Notes triggered when output exceeds percentile-based threshold
- Direct pitch selection: highest output wins

### Motion Encoding

- **8 outputs per voice** — `[u1, u4, u7, d1, d3, d8, v1, v2]`
- **Motion** = `(u1×1 + u4×4 + u7×7) - (d1×1 + d3×3 + d8×8)` — competing forces
- **Velocity** = `v1 × v2` — soft AND, both must be active
- Pitch wraps modulo 12 within each voice's octave
- All voices start at unison (C in their respective octaves)
- Favors holds (~9%) and third intervals over stepwise motion

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

| Parameter   | Default | Description              |
| ----------- | ------- | ------------------------ |
| `sim_steps` | 256     | Timesteps per simulation |
| `tempo`     | 60      | BPM for MIDI output      |

## Output

Each run creates `evolve_midi/<timestamp>_<encoding>/` containing:

- `gen###_best_*.mid` - Best MIDI every 5 generations
- `checkpoint.pkl` - Resumable state
- `final_best_*.mid` - Best overall
- `evolution_history.png` - Fitness plots

## Experiments

- `exp_outputs.py` - Test pitch encoding with visualization
- `exp_motion.py` - Test motion encoding with visualization
