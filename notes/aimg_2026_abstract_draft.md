# A Small Nervous System as Sequencer: Neuroevolved Spiking Networks for Three-Voice Pentatonic Music

*Submission to the AI Music Composition Competition, AIMG 2026. Anonymized for review.*

**Keywords:** spiking neural networks, neuroevolution, reservoir computing, algorithmic composition, symbolic music evaluation, pentatonic scales, human–AI co-creation

## Abstract

We describe a compositional system in which a small spiking neural network is run as an autonomous dynamical system and shaped by evolutionary search until its spike activity, read out as notes, forms three-voice music in a Japanese pentatonic scale. The network receives no input beyond an initial stimulus and is not trained on musical data. Each of three readout layers selects one of five scale degrees per time step, so that every note the network can produce lies in the scale by construction. A (μ+λ) evolution strategy with Gaussian and differential mutation optimizes connection weights, firing thresholds, and refractory periods against a fitness function composed of four multiplicative terms: note density, agreement with a target distribution of melodic transitions, an n-gram repetition penalty, and the placement of stable tones on metrically strong positions. The system produced twelve pieces across the In-Sen, Iwato, and Kumoi scales in two configurations, one with a single shared pool of neurons and one with three loosely coupled modules, each driving one voice. The submitted work consists of two movements, one from each configuration. The evolved MIDI was not edited; instrumentation, timbre, effects, spatialization, mix, and playback tempo were decided by a human in a digital audio workstation. We describe the network, the encoding, the fitness function and the failure modes that shaped it, and the division of labor between the algorithm and the interpreter.

## 1. Introduction

Most generative music systems based on neural networks are trained to approximate a mapping from an input to an output, and composition is performed by sampling from the trained model. This work takes a different position. The network is treated as a physical system with its own dynamics, and the question is whether evolutionary pressure can shape those dynamics into something a listener would accept as music. The composition is encoded in the weights and topology of the network rather than in any input or dataset.

The architecture belongs to the family of recurrent networks studied in reservoir computing. Maass, Natschläger, and Markram [1] proposed that a recurrent circuit of spiking neurons, the "liquid," can serve as a generic high-dimensional dynamical system from which readout units extract useful signals. Jaeger's echo state networks [2] apply the same separation to rate-based units. Both frameworks assume an input stream and a trained readout. Here the reservoir runs freely after a single initial stimulus, and evolution, rather than regression, determines both the reservoir and the readout. Kiefer [3] examined echo state networks as mapping tools for musical instruments and treated the spectral radius of the reservoir as a controllable gain; our observations about the role of inhibitory connections and refractory periods are in the same spirit. The Continuous Thought Machine of Darlow et al. [4] motivated our early experiments with neuron synchronization as a readout signal.

The search procedure is a conventional evolution strategy [5] augmented with differential mutation [6]. The decision not to use crossover follows the neuroevolution literature [7], and our own experiments confirmed that recombining two networks with coherent dynamics rarely produces a third.

The relation between algorithm and performer in this project follows the model of John Cage's chance-determined scores as described by Pritchett [8]: the procedure determines what happens and when, and the performer determines what it sounds like. We apply this division literally. The network's MIDI output is the score; a human interprets it.

The contributions of this work are modest and practical. First, a spiking reservoir is used as an autonomous sequencer, with evolution replacing readout training. Second, the pitch scale is built into the readout layer rather than imposed as an objective, which removes a conflict between tonal consistency and pitch diversity that otherwise stalls the search. Third, a compact fitness function compares each voice's full distribution of melodic transitions against a hand-specified target for the scale, which resists exploits that defeated simpler metrics. Fourth, a modular topology with a single coupling parameter yields voices with distinct rhythmic behavior while the network remains one organism under one fitness function.

## 2. System

### 2.1 Network

The network consists of simple threshold neurons connected by a sparse random weight matrix with both excitatory and inhibitory weights. At each discrete time step, a neuron accumulates weighted input from the neurons that fired on the previous step, its accumulated activation decays by a fixed leak factor, and it fires when activation reaches its threshold. After firing, the neuron enters a refractory period during which it accepts no input and its activation decays more rapidly. Refractory periods are assigned per neuron from a uniform distribution and are themselves subject to mutation. The complete update rule is given in Appendix A, and all parameter values are listed in Appendix B.

Two parameters dominate the network's temporal behavior. The activation leak determines whether activity can persist across silences; a leak factor of 0.98 per step allows activity to survive gaps of tens of steps, whereas a factor of 0.90 does not. The distribution of refractory periods determines the rhythmic texture; a wide spread of periods across neurons produces activity on many time scales. Inhibitory connections proved more effective than tuning of the spectral radius for producing varied, non-saturating dynamics.

The simulation is run for 288 steps, interpreted as sixteenth notes, from a single initial stimulus applied to the most strongly connected neuron in each module. No further input is provided.

### 2.2 Readout and encoding

Three readout layers are attached to the network, one per voice. Each readout has five output units, one per degree of the chosen pentatonic scale, connected to the reservoir through a sparse random weight pattern that is also evolved. Output units integrate the firing of their connected neurons with the same decay used for refractory activation. At each step, the output unit with the highest value in a readout triggers a note on its scale degree, provided its value exceeds a fixed threshold of 0.3. Consecutive triggers of the same degree extend the note rather than restrike it. Each voice is confined to its own octave.

Two properties of this encoding mattered. Because the output units are scale degrees, every note is in the scale by construction, and tonal consistency need not appear in the fitness function. Because the trigger threshold is fixed rather than adaptive, the network controls its own note density. Earlier versions used percentile-based or peak-relative thresholds, which guaranteed a certain density regardless of network behavior and prevented evolution from discovering sparse textures.

The scales used are In-Sen (C, D♭, F, G, B♭), Iwato (C, D♭, F, G♭, B♭), and Kumoi (C, D, E♭, G, A). Changing the scale requires changing one line of the configuration.

### 2.3 Modular configuration

In the first configuration, all three readouts draw from the same pool of 120 neurons. The resulting voices are coherent but tend to move together. In the second configuration, the network is partitioned into three contiguous blocks of 85 neurons, each voice's readout is restricted to one block, and a single parameter sets the ratio of inter-block to intra-block connection density. The network is still evolved as a single genome under a single fitness value, and the connections between blocks are shaped by selection like any others. As a diagnostic, we track the mean pairwise absolute correlation of the blocks' firing rates; it is determined largely by the topology and is not part of the fitness.

## 3. Evolution and fitness

### 3.1 Search

The search is a (μ+λ) evolution strategy with μ = 24 parents and λ = 120 offspring per generation, run for 88 generations. The genome comprises the reservoir weights, the readout weights, the firing thresholds, and the integer refractory periods. Three fifths of the offspring are produced by Gaussian mutation of a single parent, divided among five groups that anneal the mutation rate or scale upward or downward as a function of how many generations the parent has survived without improvement. The remaining two fifths are produced by differential mutation: the base parent is moved by a scaled difference between two other parents of unequal fitness. The differential step is large while the population is diverse and shrinks as it converges. No crossover and no random immigrants are used.

### 3.2 Fitness

The fitness of a MIDI file is the product of four terms, each in [0, 1]. Each term is computed per voice, and the minimum across voices is taken, so that a single deficient voice determines the score. Definitions are given in Appendix C.

*Activity* measures the total note count against a target equal to the number of simulation steps, decreasing linearly with the absolute log-ratio between the two. Silence and saturation are penalized symmetrically.

*Tonal gravity* compares the distribution of a voice's melodic transitions, over the twenty ordered pairs of distinct scale degrees, with a target distribution written by hand for each scale. The targets encode the characteristic motion of the scale: the half-step between C and D♭ in In-Sen, the two half-step pairs in Iwato, and the D–E♭ color against the C–G fifth in Kumoi. Agreement is measured as the exponential of the negative Kullback–Leibler divergence from target to observed, with additive smoothing. Because the target assigns probability to all twenty transition types, a voice that uses only a few is heavily penalized; the term therefore also enforces pitch variety.

*Repetition* penalizes a voice whose most frequent two- or three-note sequence accounts for a large share of all its sequences, with two occurrences permitted before the penalty applies.

*Metric gravity*, used only in the modular configuration, scores a voice's tendency to place the root or fifth (the fourth, for Iwato) on metrically strong sixteenth-note positions, weighted by a binary-subdivision pattern over a four-beat cycle. The score peaks when about half of the strong-position notes are stable tones, so that neither ignoring the meter nor playing the root on every downbeat is rewarded.

The product form means each term acts as a gate. A network with lively rhythm and no tonal center scores near zero, as does one that lands on the root at every downbeat but repeats a two-note figure.

### 3.3 Design history

The fitness function and the encoding reached this form through a series of failures, which we report because they are more informative than the final design.

The first fitness function combined eleven metrics from the computational musicology literature with hand-tuned weights. Within 37 generations the best network outscored a set of reference ambient recordings, including Brian Eno's *Music for Airports*, on this function. The function was discarded as over-specified and replaced by two or three terms.

With twelve chromatic output units per voice, a tonal-consistency term and a pitch-diversity term were in direct conflict: every additional pitch class in use was another opportunity to leave the scale. Evolution settled on a trade-off that satisfied neither. Reducing the output layer to the five scale degrees removed the conflict and the tonal-consistency term together.

A global diversity measure was satisfied by networks in which each voice repeated a single note, since the voices chose different notes. Computing the measure per voice and taking the minimum closed this exploit.

An earlier tonal-gravity term scored each transition individually against the target row for its origin. Networks converged on two-note oscillations between C and D♭, scoring 0.94 on the term while producing music no one would want to hear. The term could not see that eighteen of twenty transition types never occurred. Comparing the full joint distribution corrected this.

Finally, lineage tracking showed that one ancestral line typically dominated the parent pool within five generations. Speciation and fitness sharing were tried and did not change the music. The limiting factor was the fitness landscape, not the diversity of the population.

## 4. The submitted work

### 4.1 Two movements

The submitted piece consists of two movements, eight minutes in total. The first, *The Enduring Enso*, comes from the modular configuration in the [TODO: scale] scale. The second, *Echo Chamber*, comes from the unified configuration in the [TODO: scale] scale. Presenting the modular network first and the unified network second is an artistic decision rather than a chronological one.

Once each configuration was fixed, twelve evolution runs were executed, four per scale, and the best-of-run MIDI file of each was taken as a composition. The pieces were not selected from a larger pool. The MIDI output is sparse and rhythmically uneven, with bursts of activity separated by silences, which suits a tradition in which silence carries structural weight.

### 4.2 Interpretation

The MIDI files are treated as scores. In a digital audio workstation, each voice was assigned an instrument and processed through effects—reverb, delay, saturation, and filtering—and the three voices were placed in a stereo field and mixed. The playback tempo was [TODO: tempo], which extends the 288-step score to four minutes. [TODO: one sentence on instruments, if desired.] No notes were added, removed, moved, revoiced, or repeated; the note data in the recording is identical to the network's output. The fitness function evaluated symbolic MIDI only and was never applied to audio, and the production was not iterated against any metric.

We consider this division of labor part of the method rather than a departure from it. The compositions have an internal logic—repetition with variation, voices converging and diverging—but, as with Cage's chance-determined scores, the choice of weight, color, and space for each voice is what makes them legible as music. The premise of the larger project is that the notes and the sound can be separated, with computation responsible for the former and a human interpreter for the latter.

### 4.3 Listening notes

In the first movement, the three voices maintain distinct rhythmic identities and are anchored by stable tones on strong beats; the inter-module coupling can be heard as occasional entrainment between voices. In the second movement, the voices share a single pool of neurons and tend to move together, converging and diverging as a group. In both, the half-step gestures that define the scale recur as points of tension and release, and the silences between bursts are the primary formal device.

## 5. Discussion

Several conclusions held across the project. The network's physical parameters—leak, refractory period distribution, sparsity, weight scale—determined the reachable musical space more than any property of the search; a change in the leak factor from 0.90 to 0.98 had a greater effect than any mutation scheme. The encoding of network state as notes had a greater effect than the fitness function; building the scale into the readout was the single most consequential change. Evolution of a chaotic spiking system is difficult because most mutations are destructive; mutation scales had to be set two orders of magnitude below typical weight values, and differential mutation helped by following the structure of the population rather than sampling noise. Every addition of complexity—more metrics, more output units, adaptive thresholds, speciation—degraded the results.

The limitations are plain. The pieces have local coherence but no large-scale form; there is no development or arc, which is acceptable for a static, meditative idiom and would not be elsewhere. The transition targets encode the author's understanding of the scales and are a form of curation, not learned style. The fitness terms are proxies, and no listening study was conducted.

## References

[1] W. Maass, T. Natschläger, and H. Markram, "Real-time computing without stable states: A new framework for neural computation based on perturbations," *Neural Computation*, vol. 14, no. 11, pp. 2531–2560, 2002.

[2] H. Jaeger, "The 'echo state' approach to analysing and training recurrent neural networks," GMD Report 148, German National Research Center for Information Technology, 2001. [verify]

[3] C. Kiefer, "Musical instrument mapping design with echo state networks," in *Proc. Int. Conf. New Interfaces for Musical Expression (NIME)*, London, 2014, pp. 293–298.

[4] L. Darlow, C. Regan, S. Risi, J. Seely, and L. Jones, "Continuous thought machines," arXiv:2505.05522, 2025; also in *Advances in Neural Information Processing Systems*, 2025.

[5] H.-G. Beyer and H.-P. Schwefel, "Evolution strategies: A comprehensive introduction," *Natural Computing*, vol. 1, no. 1, pp. 3–52, 2002. [verify]

[6] R. Storn and K. Price, "Differential evolution: A simple and efficient heuristic for global optimization over continuous spaces," *Journal of Global Optimization*, vol. 11, no. 4, pp. 341–359, 1997. [verify]

[7] K. O. Stanley and R. Miikkulainen, "Evolving neural networks through augmenting topologies," *Evolutionary Computation*, vol. 10, no. 2, pp. 99–127, 2002. [verify]

[8] J. Pritchett, *The Music of John Cage*. Cambridge, U.K.: Cambridge University Press, 1993. [verify]

---

## Appendix A. Update rule

One simulation step, as implemented. `W` is the reservoir weight matrix, `W_out` the readout weight matrix, `a` the activation vector, `f` the Boolean firing vector, `r` the refractory counter vector, `p` the per-neuron refractory period vector, and `o` the output vector.

```
incoming     = W.T @ f
can_receive  = (r == 0)
a            = where(can_receive, clip(a + incoming, 0, 1), a)
f            = can_receive & (a >= thresholds)
r_new        = where(f, p, r)
a            = where(r > 0, a * refraction_leak, a)
o            = o * refraction_leak + W_out.T @ f
a            = a * activation_leak
r            = max(0, r_new - 1)
```

Per readout, the note at this step is `argmax(o[readout])` if `max(o[readout]) > 0.3`, else none.

## Appendix B. Hyperparameters

| Parameter | Unified (second movement) | Modular (first movement) |
| --- | --- | --- |
| Neurons | 120 | 255 (3 × 85) |
| Modules | 1 | 3 |
| Inter-/intra-module connection density ratio | — | [TODO: 0.1 or 0.5] |
| Readouts (voices) | 3 | 3 |
| Outputs per readout | 5 | 5 |
| Reservoir sparsity | 0.2 | 0.2 |
| Reservoir weight scale (std.) | 0.3 | 0.3 |
| Readout sparsity | 0.05 | 0.05 |
| Readout weight scale (std.) | 0.3 | 0.3 |
| Activation leak | 0.98 | 0.98 |
| Refraction leak | 0.75 | 0.75 |
| Refractory period | 2 + U{0, …, 61} | 2 + U{0, …, 61} |
| Weight threshold | 0 (disabled) | 0 (disabled) |
| Argmax trigger threshold | 0.3 | 0.3 |
| Simulation steps | 288 | 288 |
| Step duration | sixteenth note | sixteenth note |
| Evaluation tempo | 60 BPM | 60 BPM |
| Playback tempo (DAW) | [TODO] | [TODO] |
| μ / λ | 24 / 120 | 24 / 120 |
| Generations | 88 | 88 |
| Weight mutation rate / scale | 0.1 / 0.01 | 0.1 / 0.01 |
| Threshold mutation rate / scale | 0.1 / 0.01 | 0.1 / 0.01 |
| Refractory period mutation rate | 0.1 | 0.1 |
| Annealing factor / maximum multiplier | 0.1 / 10 | 0.1 / 10 |
| Offspring split (Gaussian : differential) | 3 : 2 | 3 : 2 |
| Differential factor F | 0.7 | 0.7 |
| Fitness | act × ton × rep | act × ton × rep × met |

## Appendix C. Fitness definitions

All terms are computed per voice and aggregated by the minimum across voices. A voice with fewer than four notes scores 0 on tonal and metric gravity.

**Activity.** With `n` the total note count and `T = 288` the target,

```
activity = max(0, 1 − |log2(n / T)| / 2)
```

so that a count of `T/2` or `2T` scores 0.5 and `T/4` or `4T` scores 0.

**Tonal gravity.** Let `q` be the target distribution over the 20 ordered pairs of distinct scale degrees, and `p` the observed distribution of a voice's non-repeating transitions after additive smoothing with α = 1. Voices with fewer than 8 transitions score 0. Otherwise

```
tonal_gravity = exp(−KL(q ‖ p)) = exp(−Σ q_i log(q_i / p_i))
```

Target matrices (rows: from; columns: to; diagonal excluded, rows sum to 1):

In-Sen (C, D♭, F, G, B♭):

```
         C     Db    F     G     Bb
C      0.00  0.40  0.25  0.20  0.15
Db     0.45  0.00  0.25  0.15  0.15
F      0.20  0.15  0.00  0.35  0.30
G      0.20  0.10  0.30  0.00  0.40
Bb     0.35  0.20  0.20  0.25  0.00
```

Iwato (C, D♭, F, G♭, B♭):

```
         C     Db    F     Gb    Bb
C      0.00  0.40  0.20  0.15  0.25
Db     0.45  0.00  0.25  0.10  0.20
F      0.15  0.10  0.00  0.45  0.30
Gb     0.15  0.10  0.45  0.00  0.30
Bb     0.35  0.15  0.25  0.25  0.00
```

Kumoi (C, D, E♭, G, A):

```
         C     D     Eb    G     A
C      0.00  0.30  0.15  0.35  0.20
D      0.25  0.00  0.40  0.20  0.15
Eb     0.20  0.45  0.00  0.20  0.15
G      0.30  0.15  0.10  0.00  0.45
A      0.25  0.15  0.15  0.45  0.00
```

[Keep only the two matrices for the scales actually used in the submitted movements.]

**Repetition.** For n ∈ {2, 3}, let `c_max` be the count of the most frequent n-gram of scale degrees in the voice and `N` the number of n-grams. With two occurrences allowed,

```
excess   = max(0, c_max − 2)
penalty  = (excess / (N − 2))^2
score_n  = 1 − penalty
repetition = mean(score_2, score_3)
```

**Metric gravity** (modular configuration only). Sixteenth-note positions within a four-beat cycle receive weights

```
w = [4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0]
```

Let `ρ` be the weighted fraction of notes on positions with `w > 0` whose pitch class is a stable tone (C and G for In-Sen and Kumoi; C and F for Iwato). Then

```
metric_gravity = ρ / 0.5                     if ρ ≤ 0.5
               = 1 − 0.5 (ρ − 0.5) / 0.5     otherwise
```

**Composite.**

```
fitness = activity × tonal_gravity × repetition × metric_gravity
```

with `metric_gravity = 1` for the unified configuration.
