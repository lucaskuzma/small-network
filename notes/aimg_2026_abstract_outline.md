# AIMG 2026 — Composition Abstract Outline

**Target:** AI Music Composition Competition, AIMG 2026 (Cross-AI Symposium 2026 / IEEE Big Data 2026)
**AoE Deadline**: End of day on November 9, 2026, in the UTC-12 time zone.
**Berlin Deadline:** Berlin Time (CET, UTC+1): November 10, 2026, at 1:00 PM. [note, corrected deadline]
**Sources reviewed:** repo + `notes/`, `Codex 2601.md`, the.strange.agency project page and "Modular Network" post, Bandcamp album page, AIMG 2026 call (current page + ACL CFP mirror + AIMG 2025 guidelines)

---

## 0. What they are actually asking for

- **It is a short paper, not a 250-word abstract**
    - 1–2 pages in **IEEE conference (two-column) format**, PDF
    - Optional **appendix up to 2 pages** (counts toward total), optional sheet music up to 9 pages; everything combined ≤ 10 pages
    - Budget: ~1,400–1,800 words for 2 pages with 1–2 small figures; appendix absorbs tables and extra plots
- **Review is double-blind**
    - No author names, no links to the.strange.agency, GitHub, or the Bandcamp page in the body
    - Do not use "The Psychic Geometries Trio" or "Sonic Mandala" as identifiers (both resolve to you in one search); the "three members, one nervous system" hook is fine if de-branded
    - Audio must be a shareable MP3/MP4 link ≤ 10 min: use an anonymous host (unlisted Drive/Dropbox/SoundCloud) rather than Bandcamp
- **Eligibility clause**
    - "Not created using AI or algorithmic methods" → ineligible; "human-enhanced AI-generated scores" explicitly allowed
    - So: the human production pass is permitted, but the abstract must make clear where the algorithm stops and the human begins
- **Audio selection** (4:00 per track → two tracks = 8:00, under cap)
    - Recommended: one Side A track (unified network) + one Side B track (modular), ideally different scales, so the pair illustrates the architectural change described in the text [yes, i will combine two tracks into a piece with two movements]
    - Alternative: a single Side B track if the competition expects one composition per entry (check the AirBalloon form; if unclear, submit the pair as one two-movement entry and say so)
    - **TODO (you):** map Bandcamp track titles → side (A/B), scale (In-Sen / Iwato / Kumoi), and run folder in `evolve_midi_modular/` / `evolve_midi/` [this will be hard, not sure it's necessary to match the run]

---

## 1. Title + one-line summary

- **Working title options**
    - _Grown, Not Written: Evolving a 256-Neuron Spiking Network to Compose in Japanese Pentatonic Modes_
    - _A Small Nervous System as Sequencer: Neuroevolved Spiking Networks for Three-Voice Pentatonic Music_ [this is best]
    - _Reservoir as Composer: Autonomous Spiking Dynamics, Evolutionary Fitness, and Human Interpretation_
- **One-sentence pitch (for the first paragraph and the AirBalloon summary field)**
    - A tiny, untrained spiking neural network is run as an autonomous dynamical system; evolution shapes its weights so its spike readouts form three-voice pentatonic music; a human then interprets the resulting MIDI score in the studio
- **Keywords**
    - spiking neural networks, neuroevolution, reservoir computing, algorithmic composition, symbolic music evaluation, pentatonic scales, human–AI co-creation

---

## 2. Abstract paragraph (~150 words)

- **Problem / stance**
    - Most AI music is a trained model mapping input → output; here the network is treated as a physical system with its own dynamics, with no training data, no gradients, no input
- **Method in one breath**
    - 255–256 [use what the code says, notes were probably power of 2 shorthand] leaky threshold neurons, excitatory + inhibitory weights, refractory periods; three readout "voices" map argmax activation to a 5-note scale; (μ+λ) evolution with Gaussian and differential mutation; multiplicative fitness: activity × tonal gravity × repetition × metric gravity
- **Result**
    - Twelve evolved pieces across In-Sen, Iwato, Kumoi; two submitted, one from a unified network and one from a modular (three-cluster) network [first movement unified, second clustered]
- **Human role, stated plainly**
    - MIDI is the composition; instrumentation, timbre, space, and mix are a human interpretation (no notes added, removed, or moved — **confirm this is true**) [confirmed, midi was untouched]

---

## 3. Introduction / Motivation (~250 words)

- **Framing question**
    - What happens when a neural network is an agent in its own right, not a model of a function?
    - "Flow Control": computation handles the notes, human attention goes to everything else
- **Lineage to cite (short, one sentence each)** [double check references, do not hallucinate here!]
    - Liquid state machines (Maass 2002) and Echo State Networks in generative/free-running mode
    - Kiefer's ESN instrument mapping (spectral radius as gain knob)
    - Sakana's Continuous Thought Machines (neuron-level timing + synchronization as representation)
    - Neuroevolution (Stanley & Miikkulainen / NEAT lineage) for gradient-free shaping of dynamics
    - Cage / chance-determined scores: the performer's sensitivity makes the result musical
- **Contribution bullets (what is new here, modestly stated)**
    - Spiking reservoir used as an _autonomous sequencer_, with evolution replacing readout training
    - Encoding insight: bake the scale into the output layer rather than evolve scale adherence
    - A compact, idiom-aware fitness (transition-distribution KL + fractal metric weights) that resists the exploits simpler metrics fell to
    - Modular architecture for voice independence with a single coupling parameter

---

## 4. System (~500 words, 1 figure)

[this section seems like it should go into an appendix, maybe brief version in human language goes here, details in appendix]

- **4.1 Network physics**
    - Update rule (4 lines of code, quote it): leaky accumulation, threshold firing, refractory reset, excitatory + inhibitory weights
    - Default physics: 255 neurons, leak 0.98, refractory 4 (+ variation up to 30), sparsity ~5%, weight scale 0.3, functional weight threshold 0.05
    - Why these matter: leak × refractory interplay determines whether activity survives silences (0.90/4 steps → 66% survives; 0.98 lets tension hold across gaps); inhibition beat spectral-radius tuning for dynamic variety
- **4.2 Readout / encoding**
    - Three readouts = three voices, each a random sparse weight pattern over the reservoir (readouts are implicit synchrony detectors)
    - Per voice: 5 outputs → one per scale degree; argmax wins if above a fixed threshold; fixed (not adaptive) threshold so the network controls its own density
    - Scales as pitch-class sets: In-Sen {0,1,5,7,10}, Iwato {0,1,5,6,10}, Kumoi {0,2,3,7,9}; one line changes the scale
    - Briefly note rejected encodings: direct neuron→pitch (arpeggio soup), sync-clustering (all parameters correlated → staccato), 12-chromatic argmax (modality vs diversity conflict), motion encoding (relative intervals; viable but not used in final pieces)
- **4.3 Modular variant (Side B)**
    - Three contiguous blocks of 85 neurons; each voice's readout hard-masked to its module
    - `inter_module_factor` = 0.1 sets inter/intra connection density ratio; whole network still one genome under one fitness
    - Diagnostic: mean pairwise |Pearson| of module firing rates ≈ 0.5 at factor 0.5 (loosely coupled); coupling floor is structural, not evolved
- **Figure 1 candidate** [not sure we need figures if they don't yield information, no figures just for the sake of having some]
    - Schematic: reservoir (3 modules) → 3 readouts × 5 outputs → MIDI piano roll; or reuse a `module_graphs/` plot

---

## 5. Evolution and fitness (~450 words, 1 figure or table)

[same as system, seems like a some of this is appendix]

- **5.1 Search**
    - (μ+λ) ES, μ=24, λ=120, 88 generations, 288 steps at 60 BPM per evaluation; no crossover (wrecks the weight balance)
    - Offspring split: 3/5 Gaussian mutation under five annealing strategies (none / scale↑ / scale↓ / rate↑ / rate↓, driven by parent age), 2/5 differential (`child = parent + F·(better − worse)`, F=0.7)
    - Differential is self-scaling: large early steps when population diverse, fine-tuning after convergence; fitness ~0.8 in 2 generations vs ~48 with Gaussian only
    - Topology evolvable (zero weights can mutate in), functional threshold prunes weak links
- **5.2 Fitness = activity × tonal gravity × repetition × metric gravity (all per-voice, min across voices)**
    - _Activity_: log-space Gaussian around a target note density — penalizes both silence and spam
    - _Tonal gravity_: KL divergence between each voice's pitch-class transition distribution and a hand-written 5×5 target matrix per scale (encodes the C↔Db pull in In-Sen, dual semitone poles in Iwato, D↔Eb color over the C–G fifth in Kumoi); subsumes pitch-entropy
    - _Repetition_: n-gram repetition penalty for sequential monotony that distributional metrics cannot see
    - _Metric gravity_: fractal weights over a 16-step cycle `[4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0]`; peaked score so root/fifth on strong beats ~50% of the time is ideal, all-root penalized
    - Multiplicative so each factor gates: great rhythm with no tonal center ≈ 0
- **5.3 Failure modes that shaped the design (this is the interesting part for reviewers — keep 4–5 crisp bullets)**
    - Percentile/peak-relative thresholds forced density → evolution couldn't learn sparsity
    - Global diversity gamed by four voices each repeating one note → per-voice min
    - Per-transition tonal scoring gamed by C↔Db oscillators (0.94 score, sounded terrible) → joint-distribution KL
    - Refractory minimum 2 → every-other-step spike trains as sole attractor → raised to 4, leak to 0.98
    - Premature convergence in ~5 generations; speciation/fitness sharing didn't help — landscape, not diversity, was the problem
    - Eleven-metric musicology fitness surpassed _Music for Airports_ in 37 generations → fitness too complex, stripped down
- **Figure 2 / Table candidate**
    - `evolution_history.png` from the submitted runs (fitness + components), or a params table (move to appendix if tight)

---

## 6. The pieces and the human interpretation (~300 words)

- **6.1 What the network delivered**
    - Twelve best-of-run MIDI files (4 per scale); sparse, rhythmically uneven, bursts separated by lulls — a good fit for a tradition that values silence [once the system was decided, we simply did n runs, there was no cherry-picking of midi]
    - Side A (unified reservoir): voices entrained, tend to move together
    - Side B (modular): distinct rhythmic identities per voice, grounded by metric gravity; pitch histograms lean toward root/fifth without collapsing
    - Submitted tracks: **[Track X] — scale, side, seed/run**; **[Track Y] — scale, side, seed/run** (fill in)
    - [i don't think we need to match seed and run]
    - [will submit: "The Enduring Enso / Echo Chamber" which is a B-side section followed by an A-side section (yes, an artistic decision to do B then A)]
- **6.2 Interpretation in the studio (recommendation: highlight, don't hide — see §9)**
    - Composition vs interpretation framing: the MIDI is a score; instrument choice, timbre, effects (reverb, delay, saturation, filtering), spatialization, and mix are performance decisions
    - State the boundary exactly: notes, pitches, onsets, durations, and voice assignments unchanged from the evolved MIDI (**verify**; if anything was edited — tempo, transposition, trimmed length to 4:00, velocity — say so explicitly) [no midi was changed, nothing repeated, etc. midi is truth]
    - Note that no automatic evaluation covered the audio stage; the fitness function only ever saw symbolic MIDI
    - Cage analogy, one or two sentences: chance-determined scores became music through the performers' sensitivity [yes this is critical]
- **6.3 Listening guide (3–4 bullets reviewers can verify against the audio)**
    - Voice independence vs entrainment (A vs B)
    - Semitone tension/resolution gestures characteristic of each scale
    - Downbeat anchoring on root/fifth; long silences as structure

---

## 7. Discussion / Lessons (~250 words)

- **Network physics > evolutionary cleverness**
    - Leak, refractory, sparsity, weight scale define the reachable space; a 0.90→0.98 leak change outperformed any mutation scheme
- **Encoding is everything**
    - Baking the scale in removed a Pareto conflict and made the fitness honest
- **Evolution is a blunt instrument for chaotic systems**
    - Most mutations destructive; mutation scale had to drop to 0.001–0.01; weight-threshold dead zone freezes topology after convergence
- **Simplicity won**
    - Final system: 255 neurons, 3 voices × 5 outputs, 4-term fitness; every added complexity (11 metrics, adaptive thresholds, speciation) hurt
- **Limitations (be upfront)**
    - Local coherence, no large-scale form or arc — acceptable for ambient/meditative music, not for other idioms
    - Fitness encodes the author's idiom targets by hand; this is curation, not learned style
    - Metrics are proxies; no listening study
- **Future work (2–3 bullets)** [leave this out]
    - CMA-ES or similar covariance-aware search to cross the threshold dead zone
    - Nested networks or slow external modulation for long-range form
    - Evolve timbre/dynamics so the human stage shrinks — or deliberately keep it as the human's domain

---

## 8. Appendix (≤ 2 pages, optional but recommended)

[really? for each run? i don't think anyone cares and this is a ton of work to produce]

- **A. Hyperparameter table**
    - From `params.txt` of the two submitted runs: μ, λ, generations, sim_steps, tempo, mutation rates/scales, anneal factor, seed, neurons, modules, inter_module_factor
- **B. Fitness details**
    - The three 5×5 transition target matrices (compact), stable-tone sets, metric weight vector, activity target formula
- **C. Plots**
    - Evolution history for each submitted run; per-voice pitch histograms; module correlation plot (Side B)
- **D. Sheet music (optional)**
    - Piano-roll excerpt is probably more honest than engraved notation; if engraving, render the evolved MIDI via music21/MuseScore and label it "unedited network output"
- **E. Production notes**
    - Instrument per voice, effects chain summary, anything altered relative to MIDI

---

## 9. On the studio production question (recommendation)

- **Highlight it, framed as interpretation, with an explicit boundary**
    - The call allows "human-enhanced AI-generated scores" — disclosure is required for eligibility clarity, so silence is the riskier choice
    - Reviewers listen to audio; if it sounds better than typical AI music and the abstract pretends it's raw, that reads as misleading; if it's framed as score + interpretation, the quality becomes a feature of the methodology
    - The Cage/performer framing you already wrote is exactly the right rhetorical move and fits the workshop's "human-AI co-creation" and "AI music interpretation" topics
- **Be precise about what the human did not do**
    - No pitch, timing, or voice edits (if true); fitness never evaluated audio; production was not iterated against any metric
    - This keeps the "composition" claim clean while being honest about the "sound" [yes, this is part of my broader flow control investigation, where we divorce notes from final sound, a human is the interpreter of an ai score]
- **Keep it proportionate**
    - ~1 paragraph in §6.2, one bullet in the abstract, one row in the appendix — not a section of its own

---

## 10. Open items for you to fill in

- Track → side/scale/run mapping; confirm total audio ≤ 10:00 [two movement piece, 8 minutes]
- Confirm nothing in the MIDI was edited before production (tempo? transposition? trimming to 4:00? velocity?) [midi as is]
- Decide: two tracks as one entry vs two entries [one entry, will contact tracks]
- Anonymous audio link [where to stick it? make soundcloud account?]
- Which figures: architecture schematic (needs drawing) + one evolution plot, or two existing plots [i am not sure that figures add anything to the paper]
- Final 256 vs 255 neuron count — be consistent (final modular runs use 255 = 3×85) [code is truth]
- IEEE template (Overleaf) and author-blind check before export
- [the airballoon form asks for file uploads but also asks for an abstract. one gets the sense that this is meant to be an abstract of the abstract? though it does say latex supported and use $...$ for inline math, hard to say.]
