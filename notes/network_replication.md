# Replicating the Sonic Mandala Networks

Settings needed to rebuild the specific networks behind the album tracks, for a standalone (e.g. JS) player with visualization.

**No re-evolution needed.** Every run folder contains `best_genotype.pkl`, the final evolved network. Simulating it with the current code reproduces the run's `final_best_*.mid` exactly (verified 2026-11-09 for both submitted movements: every note-on matches in voice, step, pitch, and velocity).

---

## Track → run mapping

From `notes/sonic_mandala.txt` (WAV filenames contain the run timestamp) and the run folders.

| # | Track | WAV | Run folder | Scale | Config | Seed | Final MIDI |
| - | ----- | --- | ---------- | ----- | ------ | ---- | ---------- |
| 1 | Whispering Bamboo | `Trio_04-193615_in-sen` | `evolve_midi_penta/20260312_193615_in-sen` | In-Sen | unified | 1234 | `final_best_0.9031.mid` |
| 2 | Koto's Resonance | `Trio_05-194507_kumoi` | `evolve_midi_penta/20260312_194507_kumoi` | Kumoi | unified | 1235 | `final_best_0.9096.mid` |
| 3 | **Echo Chamber** | `Trio_06_192932_iwato` | `evolve_midi_penta/20260312_192932_iwato` | Iwato | unified | 1677 | `final_best_0.9065.mid` |
| 4 | Silk Road Dreaming | `Trio_07-195151_more_kumoi_4s` | `evolve_midi_penta/20260312_195151_kumoi` | Kumoi | unified | 1952 | `final_best_0.8990.mid` |
| 5 | Zen Garden Starfield | `ModularTrio_06-171628_iwato_4s` | `evolve_midi/20260320_171628_iwato` | Iwato | modular | 1677 | `final_best_0.7582.mid` |
| 6 | **The Enduring Enso** | `ModularTrio_05_173617-kumoi_4s` | `evolve_midi/20260320_173617_kumoi` | Kumoi | modular | 1873 | `final_best_0.7885.mid` |
| 7 | Cosmic Journey | `ModularTrio_04-172455_in-sen_4s` | `evolve_midi/20260320_172455_in-sen` | In-Sen | modular | 1678 | `final_best_0.8425.mid` |

- Side A (tracks 1–4): Mar 12 batch, code at commit `dba56d0`.
- Side B (tracks 5–7): Mar 20 evening batch, code at commit `a045c40` (identical to current `network.py` simulation).
- Bold = the two movements submitted to AIMG 2026.
- Only tracks 3 and 6 have been verified against their MIDI so far. The others should reproduce the same way, but run the check before relying on them.

---

## What the pickle contains

`NetworkGenotype` (class in `network.py`; unpickling needs that module importable).

| Field | Echo Chamber (unified) | Enduring Enso (modular) |
| ----- | ---------------------- | ----------------------- |
| `num_neurons` | 120 | 255 |
| `num_readouts` × `n_outputs_per_readout` | 3 × 5 | 3 × 5 |
| `network_weights` | 120 × 120 float64, range −1..1, 99% nonzero | 255 × 255 float64, range −1..1, 99.5% nonzero |
| `output_weights` | 120 × 15 float64, ~100% nonzero | 255 × 15 float64, 33% nonzero (each readout masked to its module) |
| `thresholds` | 120 float64, 0..1 | 255 float64, 0..1 |
| `refraction_period` | 120 int, 4..33 | 255 int, 4..33 |
| `num_modules` / `inter_module_factor` / `readout_to_module` | absent (older class; patch in `1` / `None` / `[0,0,0]` before calling `to_network()`) | 3 / 0.5 / `[0,1,2]` |

Notes:

- The weight matrices are nearly dense. Initial sparsity (0.2) only applied at generation 0; mutation can touch any weight, and the weight threshold is 0, so almost every connection is live. A visualization of "the connectome" should plan for ~14k and ~65k edges, or filter by magnitude for display only.
- Weight convention: `network_weights[j, i]` is the connection **from** neuron j **to** neuron i (incoming = `W.T @ firing`).
- Output column `k` maps to readout `k // 5`, scale degree `k % 5`.
- Module blocks (Enso): neurons 0–84, 85–169, 170–254; readout r reads only module r.
- Refractory periods in the pickle (4..33) differ from the initial draw (2 + U[0, 62)) because they were mutated. Use the pickle values.

---

## Settings not in the pickle

Applied by `NetworkGenotype.to_network()` and the evaluation loop. Same for both movements.

| Setting | Value |
| ------- | ----- |
| Activation leak (multiplied every step) | 0.98 |
| Refraction leak (applied to activation while refractory, and to output decay) | 0.75 |
| Weight threshold | 0 (disabled) |
| Activation update | `clip(a + incoming, 0, 1)` (no tanh) |
| Threshold oscillation | none (`threshold_variation_periods` = 0) |
| Initial state | activations, firing, outputs, refractory counters all zero |
| Stimulus | add 1.0 to activation of `argmax(W.sum(axis=1))` (largest signed sum of outgoing weights), before step 0 |
| Stimulus neuron | Echo Chamber: 20 (threshold 0.0); Enso: 89 (threshold 0.494) |
| Steps | 288 |
| Step | one sixteenth note |

Tick order (important for exact replication; see `NeuralNetwork.tick`):

1. `incoming = W.T @ firing_prev`
2. `ok = refractory == 0`
3. `a[ok] = clip(a + incoming, 0, 1)[ok]`
4. `firing = ok & (a >= thresholds)`
5. `r_new = where(firing, refraction_period, r)`
6. `a[r > 0] *= refraction_leak` (uses the **old** counter)
7. `outputs = clip(outputs * refraction_leak + W_out.T @ firing, 0, 1)`
8. `a *= activation_leak`
9. `r = max(0, r_new − 1)`

---

## Output → MIDI mapping

Per voice, per step (from `save_argmax_outputs_as_midi` in `utils_sonic.py`):

- `best = argmax(outputs[voice])` (first index wins ties)
- If `outputs[best] > 0.3`: target note = `base_note[voice] + scale[best]`, else silence
- If target differs from the current note: end the current note, start the target. If the same, sustain.
- Velocity = `int(clip(outputs[best] * 127, 1, 127))` at note start
- Base notes: voice 1 = 48 (C3), voice 2 = 60 (C4), voice 3 = 72 (C5)
- Scales: In-Sen `[0, 1, 5, 7, 10]`, Iwato `[0, 1, 5, 6, 10]`, Kumoi `[0, 2, 3, 7, 9]`
- MIDI: 480 ticks/beat, 120 ticks/step, channel = voice index, file tempo 60 BPM

---

## Playback as heard on the album

From the Live sets (screenshots in `notes/*.png`). Not part of the network; include only if the player should sound like the record.

| | Echo Chamber | Enduring Enso |
| - | ------------ | ------------- |
| Tempo | 20 BPM (score = 3:36) | 20 BPM (score = 3:36) |
| Voice 1 | koto | koto |
| Voice 2 | koto + shamisen | koto + shamisen |
| Voice 3 | Indian flute + clay flute | Indian flute + clay flute |
| Shamisen note-skip | drop 1 in 9 | drop 1 in 12 |
| Timing drift | voices 1, 2 (Note Echo, input muted, LFO on delay time, few ms) | voices 1, 2 |
| Extra | reversed audio of flute voice | — |
| Effects | reverb, delay, saturation, filtering, echo | same |

The shamisen patch is strongly velocity-sensitive, so quiet notes nearly vanish.

---

## Porting checklist

- Export each pickle to JSON (weights, output weights, thresholds, refraction periods, scale, module info) with a small Python script that imports `network.py`.
- Reproduce in JS with float64 (`Float64Array`) and the exact tick order above.
- Test: the JS note list must match the run's `final_best_*.mid` note-for-note. If it does, the port is exact; small floating-point differences would show up as divergent notes, since the dynamics are chaotic.
- Useful visualization signals per step: `activations`, `firing`, `refractory_counters`, `outputs` (3 × 5), and the note events.
