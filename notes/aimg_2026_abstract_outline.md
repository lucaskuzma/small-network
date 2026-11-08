# AIMG 2026 — Composition Abstract Outline (rev. 2)

**Target:** AI Music Composition Competition, AIMG 2026 (Cross-AI Symposium 2026 / IEEE Big Data 2026)
**Deadline (per your correction):** November 9, 2026 AoE = November 10, 2026, 13:00 Berlin. Note: the workshop page lists Nov 16 for compositions under the Cross-AI dates and Nov 9 for papers; trust the AirBalloon form.
**Draft:** `notes/aimg_2026_abstract_draft.md`
**Sources:** repo + `notes/`, `Codex 2601.md`, project page, "Modular Network" post, Bandcamp page, AIMG 2026 call (current page, ACL CFP mirror, AIMG 2025 guidelines), `params.txt` of all run folders, `network.py`, `eval_basic.py`, `utils_sonic.py`

---

## 0. Requirements and decisions

- **Format**
    - 1–2 pages IEEE two-column, PDF, double-blind; optional appendix ≤ 2 pages; total ≤ 10 pages
    - AirBalloon "abstract" field is a plain-text summary of the PDF; paste the abstract paragraph (~150 words); `$...$` math is supported but unnecessary
    - Body budget ~1,500 words; appendix ~1 page, built only from material that already exists as text in the repo
- **Anonymity**
    - No author names; no links to the.strange.agency, GitHub, or Bandcamp; no "Psychic Geometries Trio" or "Sonic Mandala"
    - Track titles are not identifying on their own; use "first movement" / "second movement" in the body and give titles once
    - Audio: upload the MP3 directly if the form allows; otherwise a private link from a neutral SoundCloud account (Drive/Dropbox expose the account name)
- **Eligibility**
    - "Human-enhanced AI-generated scores" explicitly allowed; disclose the production stage and the boundary (MIDI untouched)
- **Submission** (decided)
    - One entry, one piece in two movements, 8:00 total: "The Enduring Enso" (modular network) followed by "Echo Chamber" (unified network); order is an artistic decision and is stated as such
    - No track → run → seed matching; the paper describes the two system configurations, not two specific runs

---

## 1. Corrections from the code (supersede the project page)

- **Unified configuration (first-generation system, second movement)**
    - `evolve_midi_penta/2026-03-14` batch: 120 neurons, 3 readouts × 5 outputs, no modules
    - Fitness: activity × tonal gravity × repetition (metric gravity did not exist yet)
    - **TODO:** confirm Side A came from the Mar 14 batch and not the Mar 12 batch (which predates the tonal-gravity swap)
- **Modular configuration (first movement)**
    - `evolve_midi_modular/` (noon Mar 20): 255 neurons = 3 × 85, `inter_module_factor` 0.1
    - `evolve_midi/` (evening Mar 20): same, `inter_module_factor` 0.5 (after commit "stronger inter-module connects")
    - Fitness: activity × tonal gravity × repetition × metric gravity
    - **TODO:** which batch produced Side B? Blog text mentions both 0.1 and 0.5; the correlation figure says 0.5
- **Physics shared by both** (from `params.txt`, not the January notes)
    - activation leak 0.98; refraction leak 0.75 (applied to activation while refractory, and as output decay)
    - refractory period 2 + per-neuron integer drawn uniformly from [0, 62)
    - network sparsity 0.2, weight scale 0.3; output sparsity 0.05, output weight scale 0.3
    - weight threshold 0 (the functional threshold described on the project page was disabled for the final runs)
    - argmax trigger threshold 0.3; 288 steps (16th-note grid, 72 beats) at 60 BPM
- **Search shared by both**
    - μ=24, λ=120, 88 generations, no random injection, no speciation
    - weight/threshold mutation rate 0.1, scale 0.01; refraction mutation rate 0.1; anneal factor 0.1, max 10×
    - 3/5 Gaussian (five annealing groups), 2/5 differential, F=0.7
- **Activity metric**
    - `score = max(0, 1 − |log2(count / target)| / 2)`, target = number of simulation steps; linear in log2, not Gaussian
- **Other consequences**
    - The "refractory 4 / leak 0.98 / weight threshold 0.05 / 256 neurons" story belongs to the January development phase; the paper tells it as history in §3.3, not as the final configuration
    - Tempo: 288 steps at 60 BPM is 1:12 of MIDI; each movement is 4:00. The playback tempo in the DAW is therefore an interpretive choice and must be stated. **TODO:** what tempo was used?

---

## 2. Paper structure (body ≈ 1,500 words, no figures unless a piano roll is added)

- **Title:** *A Small Nervous System as Sequencer: Neuroevolved Spiking Networks for Three-Voice Pentatonic Music*
- **Abstract (~150 words)**
    - Stance: network treated as an autonomous dynamical system, no training data, no input
    - Method in one sentence; fitness as four multiplicative terms
    - Result: twelve pieces, three scales, two configurations; two movements submitted
    - Human role: MIDI is the score; instrumentation, timbre, space, mix, and tempo are interpretation; MIDI unaltered
- **1. Introduction (~250 words)**
    - Framing: computation handles the notes; human attention goes to everything else
    - Lineage: liquid state machines (Maass et al. 2002), ESNs (Jaeger 2001) in free-running mode, Kiefer 2014, CTM (Darlow et al. 2025), neuroevolution (Stanley & Miikkulainen 2002), differential evolution (Storn & Price 1997), Cage via Pritchett 1993
    - Contributions, stated modestly: autonomous spiking reservoir as sequencer; scale baked into the readout; compact idiom-aware fitness; modular coupling parameter for voice independence
- **2. System (~350 words, conceptual; numbers go to Appendix B)**
    - 2.1 Network: leaky threshold neurons, excitatory and inhibitory weights, per-neuron refractory periods; the update rule in words, listing in Appendix A
    - 2.2 Readout and encoding: three readouts = three voices; five outputs per voice = scale degrees; argmax above a fixed threshold; why fixed not adaptive
    - 2.3 Modular configuration: three blocks, readouts masked to blocks, one coupling parameter; whole network is one genome
- **3. Evolution and fitness (~400 words)**
    - 3.1 Search: (μ+λ), Gaussian with age-based annealing plus differential mutation; no crossover
    - 3.2 Fitness: four terms, each per voice with min aggregation, multiplicative; one sentence per term, definitions in Appendix C
    - 3.3 Design history: four or five failure modes that forced the design (adaptive thresholds, per-transition gravity exploit, global diversity exploit, over-specified eleven-metric fitness, scale as a chromatic objective)
- **4. The submitted work (~350 words)**
    - 4.1 Two movements: first from the modular configuration, second from the unified one; scales [TODO]; no cherry-picking across runs beyond taking each run's best-of-run file [confirm phrasing]
    - 4.2 Interpretation: one instrument per voice, effects, spatialization, mix, tempo; MIDI note data unaltered; fitness never evaluated audio; Cage/Pritchett paragraph
    - 4.3 Listening notes: voice independence vs entrainment; semitone gestures of the scale; downbeat anchoring (first movement only); silence as structure
- **5. Discussion (~200 words)**
    - Physics over search; encoding over fitness; evolution as a blunt instrument; simplicity
    - Limitations: no long-range form; hand-written idiom targets; no listening study
    - No future-work section
- **References**
- **Appendix A: update rule** (code listing from `network.py`, ~15 lines)
- **Appendix B: hyperparameters** (one table, two columns for unified/modular, differences marked)
- **Appendix C: fitness definitions** (activity formula, KL and smoothing, n-gram penalty, metric weight vector, stable-tone sets, transition targets for the two scales used)

---

## 3. Figures

- Default: none. Nothing in the evolution plots or schematics changes a jury's reading of the piece.
- Optional single figure: piano roll of the opening of one movement, exported from the DAW session (the actual submitted MIDI). Do not substitute a `midi_graphs/` image from an unmatched run.

---

## 4. Production disclosure (resolved)

- One paragraph in §4.2, one clause in the abstract, one row in Appendix B (tempo)
- State what was done: instrument per voice, reverb, delay, saturation, filtering, spatialization, mix, playback tempo
- State what was not done: no notes added, removed, moved, or revoiced; no repeats; fitness only ever saw MIDI; production was not iterated against any metric
- Framing: score and interpretation; the project's broader premise is that notes and sound are separable, with the human as interpreter of an algorithmic score

---

## 5. Open items

- Scale of each movement (needed for §4.3 and Appendix C)
- Side A batch: Mar 12 or Mar 14? (determines whether the second movement's fitness included tonal gravity)
- Side B batch: `inter_module_factor` 0.1 or 0.5?
- DAW playback tempo
- Instrument list per voice, brief effects summary (one sentence)
- Whether to include the piano-roll figure
- Verify references marked [verify] in the draft
- Deadline: Nov 9 vs Nov 16; check the form
