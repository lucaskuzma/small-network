"""
Basic music evaluation - activity, diversity, and tonal gravity.
Walk before run: can we even get a network to play a scale?
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import mido

# Scale definitions as pitch class sets (0 = C)
SCALES = {
    "major": {0, 2, 4, 5, 7, 9, 11},
    "minor": {0, 2, 3, 5, 7, 8, 10},
    "dorian": {0, 2, 3, 5, 7, 9, 10},
    "phrygian": {0, 1, 3, 5, 7, 8, 10},
    "lydian": {0, 2, 4, 6, 7, 9, 11},
    "mixolydian": {0, 2, 4, 5, 7, 9, 10},
    "aeolian": {0, 2, 3, 5, 7, 8, 10},
    "locrian": {0, 1, 3, 5, 6, 8, 10},
    "pentatonic_major": {0, 2, 4, 7, 9},
    "pentatonic_minor": {0, 3, 5, 7, 10},
    "whole_tone": {0, 2, 4, 6, 8, 10},
}


# Target transition matrices for Japanese pentatonic scales.
# Each matrix is 5x5: target_matrix[i][j] = ideal probability of moving
# from scale degree i to scale degree j (given you move to a *different* note).
# Diagonal is 0, rows sum to 1.
# Encodes characteristic melodic motion: semitone pairs are the expressive core.
TRANSITION_TARGETS: Dict[str, Tuple[List[int], np.ndarray]] = {
    # In-Sen: C, Db, F, G, Bb — semitone at C<->Db
    # The C<->Db half-step is the defining gesture. Db->C resolution is strongest.
    "in-sen": (
        [0, 1, 5, 7, 10],
        np.array(
            [
                #    C     Db    F     G     Bb
                [0.00, 0.40, 0.25, 0.20, 0.15],  # FROM C:  C->Db semitone tension
                [0.45, 0.00, 0.25, 0.15, 0.15],  # FROM Db: Db->C resolution
                [0.20, 0.15, 0.00, 0.35, 0.30],  # FROM F:  F->G stepwise, F->Bb
                [0.20, 0.10, 0.30, 0.00, 0.40],  # FROM G:  G->Bb stepwise, G->F
                [0.35, 0.20, 0.20, 0.25, 0.00],  # FROM Bb: Bb->C resolution
            ]
        ),
    ),
    # Iwato: C, Db, F, Gb, Bb — semitones at C<->Db AND F<->Gb
    # Darkest scale: two semitone poles, tritone C-Gb.
    "iwato": (
        [0, 1, 5, 6, 10],
        np.array(
            [
                #    C     Db    F     Gb    Bb
                [0.00, 0.40, 0.20, 0.15, 0.25],  # FROM C:  C->Db semitone
                [0.45, 0.00, 0.25, 0.10, 0.20],  # FROM Db: Db->C resolution
                [0.15, 0.10, 0.00, 0.45, 0.30],  # FROM F:  F->Gb semitone
                [0.15, 0.10, 0.45, 0.00, 0.30],  # FROM Gb: Gb->F resolution
                [0.35, 0.15, 0.25, 0.25, 0.00],  # FROM Bb: Bb->C, connects both poles
            ]
        ),
    ),
    # Kumoi: C, D, Eb, G, A — semitone at D<->Eb
    # Warmer scale. C-G fifth is structural, D<->Eb is the color.
    "kumoi": (
        [0, 2, 3, 7, 9],
        np.array(
            [
                #    C     D     Eb    G     A
                [0.00, 0.30, 0.15, 0.35, 0.20],  # FROM C:  C->G structural, C->D
                [0.25, 0.00, 0.40, 0.20, 0.15],  # FROM D:  D->Eb semitone
                [0.20, 0.45, 0.00, 0.20, 0.15],  # FROM Eb: Eb->D resolution
                [0.30, 0.15, 0.10, 0.00, 0.45],  # FROM G:  G->A stepwise, G->C
                [0.25, 0.15, 0.15, 0.45, 0.00],  # FROM A:  A->G stepwise
            ]
        ),
    ),
}


# Stable tones per scale (pitch classes that serve as metric anchors).
# Root + fifth where possible; root + fourth for Iwato (no clean fifth).
STABLE_TONES: Dict[str, set] = {
    "in-sen": {0, 7},  # C + G (fifth)
    "iwato": {0, 5},  # C + F (fourth; Gb is a tritone, not stable)
    "kumoi": {0, 7},  # C + G (fifth)
}

# Fractal metric weight pattern for one 4-beat phrase (16 16th-notes).
# Each level of binary subdivision adds 1 to positions it hits.
# Position 0 (the "one") is strongest; odd positions (offbeat 16ths) are 0.
METRIC_WEIGHTS = np.array([4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0], dtype=float)


@dataclass
class BasicMetrics:
    """Container for basic evaluation metrics."""

    modal_consistency: float  # 0-1, kept for compatibility but always 0
    best_scale: str  # kept for compatibility
    best_root: int  # kept for compatibility
    activity: float  # 0-1, based on note density
    note_density: float  # notes per beat
    note_count: int  # raw number of notes
    diversity: float  # 0-1, diagnostic only (not in composite)
    pitch_entropy: float  # 0-1, diagnostic only (subsumed by tonal_gravity)
    repetition_score: float  # 0-1, penalty for repeated n-grams (1 = no repetition)
    tonal_gravity: float  # 0-1, joint transition distribution match (captures both idiom + variety)
    metric_gravity: float  # 0-1, stable tones on strong beats
    composite_score: (
        float  # activity * tonal_gravity * repetition_score * metric_gravity
    )

    def __str__(self) -> str:
        return (
            f"BasicMetrics(\n"
            f"  composite: {self.composite_score:.3f} (act * grav * rep * met)\n"
            f"  activity: {self.activity:.3f} ({self.note_count} notes, {self.note_density:.2f}/beat)\n"
            f"  tonal_gravity: {self.tonal_gravity:.3f}\n"
            f"  metric_gravity: {self.metric_gravity:.3f}\n"
            f"  repetition: {self.repetition_score:.3f}\n"
            f"  diversity: {self.diversity:.3f} (entropy={self.pitch_entropy:.3f}) [diagnostic]\n"
            f")"
        )


class BasicAnalyzer:
    """
    Simple analyzer focused on activity, tonal quality, and non-repetition.

    Composite score is multiplicative:
        activity * tonal_gravity * repetition_score * metric_gravity.
    Tonal gravity uses KL divergence against a target joint transition distribution,
    unifying pitch variety and melodic idiom into a single metric. Metric gravity
    rewards stable tones (root/fifth) on metrically strong beats.
    """

    def __init__(
        self,
        target_notes: int = 128,  # Target note count (should match sim_steps from evolve.py)
        scale: Optional[str] = None,  # Scale name for tonal gravity (e.g. "in-sen")
    ):
        """
        Args:
            target_notes: Target note count for activity=1.0 (typically = sim_steps)
            scale: Scale name for transition-based tonal gravity. If None or unknown,
                   tonal gravity defaults to 1.0 (no effect on composite).
        """
        self.target_notes = target_notes
        self.scale = scale
        self._stable_tones = STABLE_TONES.get(scale) if scale else None

        target = TRANSITION_TARGETS.get(scale) if scale else None
        if target is not None:
            self._scale_degrees = target[0]
            self._target_matrix = target[1]
            n = self._target_matrix.shape[0]
            self._n_transition_types = n * (n - 1)
            # Joint transition distribution: each row contributes 1/n of all transitions
            flat = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        flat.append(self._target_matrix[i, j] / n)
            self._target_flat = np.array(flat)
        else:
            self._scale_degrees = None
            self._target_matrix = None
            self._target_flat = None
            self._n_transition_types = 0

    def load_midi(self, midi_path: str) -> Tuple[List[dict], int, int]:
        """
        Load MIDI file and extract note events.

        Returns:
            notes: list of note events {pitch, start_tick, end_tick, channel}
            duration_ticks: total duration in ticks
            ticks_per_beat: MIDI resolution
        """
        mid = mido.MidiFile(midi_path)
        notes = []
        max_tick = 0

        for track_idx, track in enumerate(mid.tracks):
            current_tick = 0
            active_notes = {}  # pitch -> start_tick

            for msg in track:
                current_tick += msg.time

                if msg.type == "note_on" and msg.velocity > 0:
                    key = (msg.note, msg.channel)
                    active_notes[key] = current_tick

                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    key = (msg.note, msg.channel)
                    if key in active_notes:
                        notes.append(
                            {
                                "pitch": msg.note,
                                "start_tick": active_notes[key],
                                "end_tick": current_tick,
                                "channel": msg.channel,
                                "track": track_idx,
                            }
                        )
                        del active_notes[key]

            max_tick = max(max_tick, current_tick)

        return (
            sorted(notes, key=lambda x: x["start_tick"]),
            max_tick,
            mid.ticks_per_beat,
        )

    def compute_modal_consistency(
        self, pitch_classes: List[int]
    ) -> Tuple[float, str, int]:
        """
        Find best-fitting scale, normalized so random chromatic ≈ 0.

        For each of 12 roots × 11 scales, count what fraction of played notes
        fit the scale. Normalize against random baseline (7/12 for 7-note scales).

        Returns: (normalized_score, scale_name, root)
        """
        if not pitch_classes:
            return 0.0, "none", 0  # No notes = no modal consistency

        best_fit = 0.0
        best_scale = "major"
        best_root = 0

        for root in range(12):
            for scale_name, scale_pcs in SCALES.items():
                # Transpose scale to this root
                transposed = {(p + root) % 12 for p in scale_pcs}
                # Count how many of the played pitch classes are in scale
                in_scale = sum(1 for pc in pitch_classes if pc in transposed)
                fit = in_scale / len(pitch_classes)

                if fit > best_fit:
                    best_fit = fit
                    best_scale = scale_name
                    best_root = root

        # Normalize: random chromatic ≈ 7/12 (0.583) for 7-note scales
        # score = (fit - baseline) / (1 - baseline), so random → 0, perfect → 1
        baseline = 7 / 12
        normalized = max(0, (best_fit - baseline) / (1 - baseline))

        return normalized, best_scale, best_root

    def compute_activity(
        self, notes: List[dict], duration_ticks: int, ticks_per_beat: int
    ) -> Tuple[float, float]:
        """
        Compute activity score based on note count vs target.

        Peaked function: score=1.0 at target, decreases for both over and under.
        Uses ratio-based scoring that's symmetric in log space.

        Returns: (activity_score, note_density)
            activity_score: 0-1 (1 = exactly target notes)
            note_density: notes per beat (for display)
        """
        note_count = len(notes)

        if duration_ticks == 0:
            return 0.0, 0.0

        # Compute note density for display
        duration_beats = duration_ticks / ticks_per_beat
        note_density = note_count / duration_beats if duration_beats > 0 else 0.0

        if note_count == 0:
            return 0.0, note_density

        # Peaked activity: penalize both over and under target
        # Use ratio to be symmetric: 64 notes and 256 notes both 2x off from 128
        # score = 1 - |log2(count/target)| / max_log2_deviation
        # At target: score = 1.0
        # At half or double: score = 0.5
        # At quarter or quadruple: score = 0.0
        ratio = note_count / self.target_notes
        if ratio <= 0:
            return 0.0, note_density

        log_deviation = abs(np.log2(ratio))
        # 2 octaves (4x) of deviation = score 0
        max_deviation = 2.0
        score = max(0.0, 1.0 - log_deviation / max_deviation)

        return float(score), note_density

    def _compute_voice_diversity(
        self, pitch_classes: List[int], entropy_weight: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Compute diversity for a single voice.

        Returns: (diversity_score, pitch_entropy, repetition_score)
        """
        from collections import Counter

        if len(pitch_classes) < 2:
            return 0.0, 0.0, 0.0

        # === Pitch Class Entropy ===
        # Entropy of pitch class distribution, normalized by log2(scale_size)
        # A 7-note diatonic scale has max entropy of log2(7) ≈ 2.81
        counts = np.bincount(pitch_classes, minlength=12)
        probs = counts / len(pitch_classes)
        probs = probs[probs > 0]  # Remove zeros for log
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize by ideal scale entropy (7-note scale)
        max_entropy = np.log2(7)  # ≈ 2.81 bits
        pitch_entropy = min(1.0, entropy / max_entropy)

        # === N-gram Repetition Penalty ===
        # Check 2-grams and 3-grams, penalize heavy repetition
        # Score of 1.0 = all unique patterns, 0.0 = all same pattern
        ngram_scores = []

        for n in [2, 3]:
            if len(pitch_classes) < n + 1:
                continue
            ngrams = [
                tuple(pitch_classes[i : i + n])
                for i in range(len(pitch_classes) - n + 1)
            ]
            counts = Counter(ngrams)

            # Compute repetition penalty: (max_count - 1) / (num_ngrams - 1)
            # 0 = all unique, 1 = all same pattern
            max_count = max(counts.values())
            num_ngrams = len(ngrams)

            if num_ngrams > 1:
                # Allow some repetition before penalty kicks in
                allowed_repeats = 2
                excess_repeats = max(0, max_count - allowed_repeats)
                max_possible_excess = num_ngrams - allowed_repeats

                if max_possible_excess > 0:
                    # Quadratic penalty for steep punishment
                    penalty = (excess_repeats / max_possible_excess) ** 2
                    ngram_scores.append(1.0 - penalty)
                else:
                    ngram_scores.append(1.0)

        repetition_score = np.mean(ngram_scores) if ngram_scores else 1.0

        # === Combined Diversity Score ===
        diversity = (
            entropy_weight * pitch_entropy + (1 - entropy_weight) * repetition_score
        )

        return float(diversity), float(pitch_entropy), float(repetition_score)

    def compute_diversity(
        self, notes: List[dict], entropy_weight: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Compute diversity score PER VOICE, then aggregate.

        Each voice must individually be diverse - a voice repeating one note
        will tank the score even if other voices play different notes.

        Args:
            notes: List of note dicts with 'pitch' and 'track' fields
            entropy_weight: Weight for entropy vs repetition (0.5 = equal)

        Returns: (diversity_score, pitch_entropy, repetition_score)
            All values 0-1, where 1 = maximally diverse
        """
        if len(notes) < 2:
            return 0.0, 0.0, 0.0

        # Group notes by track (voice)
        from collections import defaultdict

        notes_by_track = defaultdict(list)
        for note in notes:
            notes_by_track[note["track"]].append(note)

        # Compute diversity per voice
        voice_diversities = []
        voice_entropies = []
        voice_repetitions = []

        for track_notes in notes_by_track.values():
            if len(track_notes) < 2:
                # Voice with 0-1 notes gets zero diversity
                voice_diversities.append(0.0)
                voice_entropies.append(0.0)
                voice_repetitions.append(0.0)
                continue

            # Sort by start time within voice
            track_notes = sorted(track_notes, key=lambda x: x["start_tick"])
            pitch_classes = [n["pitch"] % 12 for n in track_notes]

            div, ent, rep = self._compute_voice_diversity(pitch_classes, entropy_weight)
            voice_diversities.append(div)
            voice_entropies.append(ent)
            voice_repetitions.append(rep)

        if not voice_diversities:
            return 0.0, 0.0, 0.0

        # Aggregate: use MINIMUM to force ALL voices to be diverse
        # (mean would allow one repetitive voice if others compensate)
        diversity = float(np.min(voice_diversities))
        pitch_entropy = float(np.min(voice_entropies))
        repetition_score = float(np.min(voice_repetitions))

        return diversity, pitch_entropy, repetition_score

    def _compute_voice_tonal_quality(self, pitch_classes: List[int]) -> float:
        """
        Score a single voice against the joint transition distribution.

        Counts all non-self transitions, builds an observed distribution over
        the 20 possible transition types (5 notes x 4 destinations), smooths it,
        and compares to the target via KL(target || observed).

        This single metric replaces both pitch entropy and the old per-row gravity:
        - An oscillator (e.g. C<->Db) concentrates on 2 of 20 transitions,
          massively diverging from the target -> near-zero score.
        - A diverse network matching the target idiom -> high score.

        Returns: 0-1 score via exp(-KL). 1.0 = perfect match, decays toward 0.
        """
        n = len(self._scale_degrees)
        pc_to_idx = {pc: i for i, pc in enumerate(self._scale_degrees)}

        observed_counts = np.zeros(self._n_transition_types)
        for k in range(1, len(pitch_classes)):
            pc_from = pitch_classes[k - 1]
            pc_to = pitch_classes[k]
            if pc_from == pc_to:
                continue
            i = pc_to_idx.get(pc_from)
            j = pc_to_idx.get(pc_to)
            if i is not None and j is not None:
                flat_idx = i * (n - 1) + (j if j < i else j - 1)
                observed_counts[flat_idx] += 1

        total = observed_counts.sum()
        if total < 8:
            return 0.0

        # Laplace smoothing so no transition type has zero probability
        alpha = 1.0
        observed_flat = (observed_counts + alpha) / (
            total + self._n_transition_types * alpha
        )

        # KL(target || observed): heavily penalizes transitions the target
        # expects but the network never makes (the oscillator failure mode)
        kl = float(
            np.sum(self._target_flat * np.log(self._target_flat / observed_flat))
        )

        return float(np.exp(-kl))

    def compute_tonal_gravity(self, notes: List[dict]) -> float:
        """
        Compute tonal quality score per voice, then aggregate.

        Compares each voice's transition distribution against the target joint
        distribution for the configured scale. This single metric captures both
        tonal gravity (prefer characteristic transitions) and pitch diversity
        (use all transition types, not just one pair).

        Returns: 0-1 score (1 = transitions match target idiom perfectly).
                 Returns 1.0 if no transition target is configured (no penalty).
        """
        if self._target_flat is None:
            return 1.0

        if len(notes) < 2:
            return 0.0

        from collections import defaultdict

        notes_by_track = defaultdict(list)
        for note in notes:
            notes_by_track[note["track"]].append(note)

        voice_scores = []
        for track_notes in notes_by_track.values():
            if len(track_notes) < 4:
                voice_scores.append(0.0)
                continue

            track_notes = sorted(track_notes, key=lambda x: x["start_tick"])
            pitch_classes = [n["pitch"] % 12 for n in track_notes]

            score = self._compute_voice_tonal_quality(pitch_classes)
            voice_scores.append(score)

        if not voice_scores:
            return 0.0

        return float(np.min(voice_scores))

    def _compute_voice_metric_gravity(
        self,
        notes: List[dict],
        ticks_per_beat: int,
        target: float = 0.5,
    ) -> float:
        """Score a single voice's tendency to play stable tones on strong beats.

        For each note, looks up the fractal metric weight at its position in the
        16-step cycle.  Computes the weighted fraction of notes on strong beats
        that are stable tones.

        Uses a peaked score: ramps from 0 at ratio=0 to 1.0 at ratio=target,
        then gently decays toward 0.5 at ratio=1.0 (all stable is still OK-ish).
        """
        if not notes or self._stable_tones is None:
            return 1.0

        ticks_per_16th = ticks_per_beat // 4
        weighted_stable = 0.0
        total_weight = 0.0

        for note in notes:
            pos = int(note["start_tick"] / ticks_per_16th) % 16
            w = METRIC_WEIGHTS[pos]
            if w > 0:
                total_weight += w
                if note["pitch"] % 12 in self._stable_tones:
                    weighted_stable += w

        if total_weight == 0:
            return 0.0

        ratio = weighted_stable / total_weight

        if ratio <= target:
            return ratio / target
        else:
            return 1.0 - 0.5 * (ratio - target) / (1.0 - target)

    def compute_metric_gravity(self, notes: List[dict], ticks_per_beat: int) -> float:
        """Compute metric gravity per voice, then aggregate (min).

        Returns 0-1 (1 = voices tend to play root/fifth on strong beats).
        Returns 1.0 if no stable tones are configured for this scale.
        """
        if self._stable_tones is None:
            return 1.0
        if len(notes) < 2:
            return 0.0

        from collections import defaultdict

        notes_by_track = defaultdict(list)
        for note in notes:
            notes_by_track[note["track"]].append(note)

        voice_scores = []
        for track_notes in notes_by_track.values():
            if len(track_notes) < 4:
                voice_scores.append(0.0)
                continue
            score = self._compute_voice_metric_gravity(track_notes, ticks_per_beat)
            voice_scores.append(score)

        if not voice_scores:
            return 0.0

        return float(np.min(voice_scores))

    def analyze(self, midi_path: str) -> BasicMetrics:
        """
        Analyze a MIDI file and return basic metrics.

        Args:
            midi_path: Path to MIDI file

        Returns:
            BasicMetrics dataclass with scores
        """
        notes, duration_ticks, ticks_per_beat = self.load_midi(midi_path)

        if not notes:
            return BasicMetrics(
                modal_consistency=0.0,
                best_scale="none",
                best_root=0,
                activity=0.0,
                note_density=0.0,
                note_count=0,
                diversity=0.0,
                pitch_entropy=0.0,
                repetition_score=0.0,
                tonal_gravity=0.0,
                metric_gravity=0.0,
                composite_score=0.0,
            )

        # Compute metrics
        activity, note_density = self.compute_activity(
            notes, duration_ticks, ticks_per_beat
        )
        # Per-voice diversity (diagnostic only — entropy is subsumed by tonal gravity)
        diversity, pitch_entropy, repetition_score = self.compute_diversity(notes)
        # Per-voice tonal quality via joint transition distribution
        tonal_gravity = self.compute_tonal_gravity(notes)
        # Per-voice stable-tone tendency on metrically strong beats
        metric_gravity = self.compute_metric_gravity(notes, ticks_per_beat)

        composite = activity * tonal_gravity * repetition_score * metric_gravity

        return BasicMetrics(
            modal_consistency=0.0,
            best_scale=self.scale or "pentatonic",
            best_root=0,
            activity=activity,
            note_density=note_density,
            note_count=len(notes),
            diversity=diversity,
            pitch_entropy=pitch_entropy,
            repetition_score=repetition_score,
            tonal_gravity=tonal_gravity,
            metric_gravity=metric_gravity,
            composite_score=composite,
        )


# Convenience function
def evaluate_basic(midi_path: str, **kwargs) -> BasicMetrics:
    """Quick evaluation of a single MIDI file."""
    analyzer = BasicAnalyzer(**kwargs)
    return analyzer.analyze(midi_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eval_basic.py <midi_file>")
        sys.exit(1)

    metrics = evaluate_basic(sys.argv[1])
    print(metrics)
