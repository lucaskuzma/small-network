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

- **Run reconstruction (resolved via `notes/sonic_mandala.txt` + git history)**
    - Album Side A = Mar 12 batch (`evolve_midi_penta/20260312_*`, 12 runs); Side B = Mar 20 evening batch (`evolve_midi/`, 9 runs, `inter_module_factor` 0.5). Mar 14 batch and Mar 20 noon modular batch were not used on the album
    - Both recorded fitness values reproduce exactly when the MIDI is re-scored with the evaluator at the commit in effect when the run started
- **Movement 1: *The Enduring Enso*** (`ModularTrio_05`)
    - Run `evolve_midi/20260320_173617_kumoi`, `final_best_0.7885.mid`; seed 1873; code at `a045c40` (= current code)
    - 255 neurons = 3 × 85, `inter_module_factor` 0.5, Kumoi
    - Fitness: activity × tonal gravity × repetition × metric gravity
    - Notes per voice: 84 / 89 / 106
- **Movement 2: *Echo Chamber*** (`Trio_06`)
    - Run `evolve_midi_penta/20260312_192932_iwato`, `final_best_0.9065.mid`; seed 1677; code at `dba56d0` (Mar 12 19:03, run started 19:29)
    - 120 neurons, no modules, Iwato
    - Fitness: **activity × diversity** (diversity = 0.5 · pitch-class entropy normalized by log2 7 + 0.5 · repetition). No tonal gravity: it was added Mar 14
    - Notes per voice: 64 / 108 / 116 (288 total, exactly on target)
- **Stimulus:** both runs stimulate a single most-connected neuron (not one per module)
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
    - Tempo: playback at 20 BPM, so the 72-beat score lasts 3:36 of each 4:00 track
- **Production (from the Live sets)**
    - Some voices doubled across two instruments; one instrument of a pair receives the voice through a Max for Live "note skip" device that drops 1 of every N notes
    - Note Echo with input muted and an LFO on its delay time, used as timing drift on voices 1 and 2; audio reverb, delay, saturation, filtering; LFO modulation of device parameters
    - The screenshot shown is the `ModularTrio_06` set (*Zen Garden Starfield*), which also has a reversed audio resample of the flute; confirm what the two submitted sets contain

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

- §4.2 now has four labeled categories: tempo, orchestration, real-time effects, mix; one clause in the abstract; tempo row in Appendix B
- State what was done: tempo 20 BPM; instruments per voice, with doubling; note-skip device on one instrument of a doubled pair; LFO timing drift on voices 1 and 2; audio reverb, delay, echo, saturation, filtering; reversed flute audio (Echo Chamber); mix and panning
- State the boundary precisely: the score (MIDI clip data) is unedited; real-time devices change what is heard (doublings, notes dropped within a doubling, millisecond timing offsets, audio echoes, reversed audio); fitness only ever saw MIDI; production was not iterated against any metric
- Framing: score and interpretation; the project's broader premise is that notes and sound are separable, with the human as interpreter of an algorithmic score

---

## 5. Open items

- Resolved: selection by listening (Live scenes show the auditioned runs; the chosen ones were not the top scorers); orchestration V1 koto, V2 koto + shamisen, V3 Indian flute + clay flute in both; reversed flute audio in Echo Chamber only; last 24 s is a fade-out
- Resolved: Note Echo (input muted, LFO on delay time) is a timing "humanization" drift on voices 1 and 2 in both movements, not an echo; voice 3 is on the grid. Shamisen note-skip is in both sets: N = 12 in Enso, N = 9 in Echo Chamber. The shamisen patch is strongly velocity-sensitive, so low-velocity notes nearly vanish; no velocity remapping, but the instrument choice acts as an implicit velocity mapping (stated in §4.2). Draft §4.2 has a separate "Timing drift" paragraph
- Whether to include the piano-roll figure
- Verify references marked [verify] in the draft
- Deadline: Nov 9 vs Nov 16; check the form
