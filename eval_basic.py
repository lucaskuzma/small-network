"""
Basic music evaluation - just modal consistency and activity.
Walk before run: can we even get a network to play a scale?
"""

import numpy as np
from typing import List, Tuple
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


@dataclass
class BasicMetrics:
    """Container for basic evaluation metrics."""

    modal_consistency: float  # 0-1, how well notes fit a scale
    best_scale: str  # which scale fits best
    best_root: int  # root of best-fitting scale (0=C)
    activity: float  # 0-1, based on note density
    note_density: float  # notes per beat
    note_count: int  # raw number of notes
    diversity: float  # 0-1, pitch variety + anti-repetition
    pitch_entropy: float  # 0-1, normalized entropy of pitch classes used
    repetition_score: float  # 0-1, penalty for repeated n-grams (1 = no repetition)
    composite_score: float  # weighted combination

    def __str__(self) -> str:
        root_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return (
            f"BasicMetrics(\n"
            f"  composite: {self.composite_score:.3f}\n"
            f"  modal: {self.modal_consistency:.3f} ({root_names[self.best_root]} {self.best_scale})\n"
            f"  activity: {self.activity:.3f} ({self.note_count} notes, {self.note_density:.2f}/beat)\n"
            f"  diversity: {self.diversity:.3f} (entropy={self.pitch_entropy:.3f}, rep={self.repetition_score:.3f})\n"
            f")"
        )


class BasicAnalyzer:
    """
    Simple analyzer focused on modal consistency, activity, and diversity.

    Composite score is multiplicative: modal * activity * diversity
    This ensures all three must be satisfied - modality/diversity are
    meaningless without sufficient activity.
    """

    def __init__(
        self,
        target_notes: int = 128,  # Target note count (should match sim_steps from evolve.py)
    ):
        """
        Args:
            target_notes: Target note count for activity=1.0 (typically = sim_steps)
        """
        self.target_notes = target_notes

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
                composite_score=0.0,
            )

        pitch_classes = [n["pitch"] % 12 for n in notes]

        # Compute metrics
        modal, best_scale, best_root = self.compute_modal_consistency(pitch_classes)
        activity, note_density = self.compute_activity(
            notes, duration_ticks, ticks_per_beat
        )
        # Per-voice diversity - each voice must individually be diverse
        diversity, pitch_entropy, repetition_score = self.compute_diversity(notes)

        # Composite: multiplicative - all three must be satisfied
        # Any zero kills the score; modality/diversity meaningless without activity
        composite = modal * activity * diversity

        return BasicMetrics(
            modal_consistency=modal,
            best_scale=best_scale,
            best_root=best_root,
            activity=activity,
            note_density=note_density,
            note_count=len(notes),
            diversity=diversity,
            pitch_entropy=pitch_entropy,
            repetition_score=repetition_score,
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
