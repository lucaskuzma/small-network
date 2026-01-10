"""
Ambient music evaluation heuristics.
Works on MIDI files - tempo invariant, symbolic domain only.
"""

import numpy as np
from collections import Counter
from typing import List, Set, Tuple, Optional
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

# Consonant intervals (in semitones, mod 12)
CONSONANT_INTERVALS = {0, 3, 4, 5, 7, 8, 9}  # unison, m3, M3, P4, P5, m6, M6


@dataclass
class AmbientMetrics:
    """Container for all ambient evaluation metrics."""

    modal_consistency: float  # 0-1, how well notes fit a scale
    best_scale: str  # which scale fits best
    best_root: int  # root of best-fitting scale (0=C)
    pitch_vocabulary: int  # number of pitch classes used
    pitch_vocabulary_score: float  # 0-1, penalize too few or too many
    voice_stasis: float  # 0-1, how static the most static voice is
    melodic_smoothness: float  # 0-1, stepwise motion vs leaps
    harmonic_rhythm: float  # 0-1, stability of vertical harmony
    note_duration: float  # 0-1, relative preference for longer notes
    consonance: float  # 0-1, fraction of consonant intervals
    sparsity: float  # 0-1, average fraction of voices silent
    voice_independence: float  # 0-1, avoid parallel motion
    cadence_penalty: float  # 0-1, penalize V-I motion (1 = no cadences)
    composite_score: float  # weighted combination

    def __str__(self) -> str:
        root_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return (
            f"AmbientMetrics(\n"
            f"  composite_score: {self.composite_score:.3f}\n"
            f"  modal_consistency: {self.modal_consistency:.3f} ({root_names[self.best_root]} {self.best_scale})\n"
            f"  pitch_vocabulary: {self.pitch_vocabulary} classes (score: {self.pitch_vocabulary_score:.3f})\n"
            f"  voice_stasis: {self.voice_stasis:.3f}\n"
            f"  melodic_smoothness: {self.melodic_smoothness:.3f}\n"
            f"  harmonic_rhythm: {self.harmonic_rhythm:.3f}\n"
            f"  note_duration: {self.note_duration:.3f}\n"
            f"  consonance: {self.consonance:.3f}\n"
            f"  voice_independence: {self.voice_independence:.3f}\n"
            f"  sparsity: {self.sparsity:.3f}\n"
            f"  cadence_penalty: {self.cadence_penalty:.3f}\n"
            f")"
        )


class AmbientAnalyzer:
    """
    Analyzes MIDI files for ambient music characteristics.

    All metrics are tempo-invariant (work on symbolic note events).
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        ideal_pitch_classes: Tuple[int, int] = (4, 7),
        target_sparsity: float = 0.5,
    ):
        """
        Args:
            weights: Dict of metric weights for composite score.
                     Keys: modal, vocabulary, stasis, melodic_smooth, harmonic_rhythm,
                           note_duration, consonance, independence, sparsity, cadence
            ideal_pitch_classes: (min, max) ideal number of pitch classes
            target_sparsity: Target fraction of voices silent (0-1)
        """
        self.weights = weights or {
            "modal": 0.15,
            "vocabulary": 0.10,
            "stasis": 0.15,
            "melodic_smooth": 0.15,
            "harmonic_rhythm": 0.15,
            "note_duration": 0.10,
            "consonance": 0.10,
            "independence": 0.05,
            "sparsity": 0.00,  # Disabled for now - seems buggy
            "cadence": 0.05,
        }
        self.ideal_pitch_classes = ideal_pitch_classes
        self.target_sparsity = target_sparsity

    def load_midi(self, midi_path: str) -> List[dict]:
        """
        Load MIDI file and extract note events.

        Returns list of note events: {pitch, start_tick, end_tick, channel}
        """
        mid = mido.MidiFile(midi_path)
        notes = []

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

        return sorted(notes, key=lambda x: x["start_tick"])

    def get_pitch_classes(self, notes: List[dict]) -> List[int]:
        """Extract pitch classes (0-11) from note list."""
        return [n["pitch"] % 12 for n in notes]

    def get_pitches_by_voice(self, notes: List[dict]) -> dict:
        """Group pitches by channel/track (as proxy for voice)."""
        voices = {}
        for n in notes:
            voice_id = (n["track"], n["channel"])
            if voice_id not in voices:
                voices[voice_id] = []
            voices[voice_id].append(n)
        return voices

    def compute_modal_consistency(
        self, pitch_classes: List[int]
    ) -> Tuple[float, str, int]:
        """
        Find best-fitting scale and return consistency score.

        Returns: (score, scale_name, root)
        """
        if not pitch_classes:
            return 1.0, "none", 0

        best_fit = 0.0
        best_scale = "major"
        best_root = 0

        pc_set = set(pitch_classes)

        for root in range(12):
            for scale_name, scale_pcs in SCALES.items():
                # Transpose scale to this root
                transposed = {(p + root) % 12 for p in scale_pcs}
                # Count how many of the used pitch classes are in scale
                in_scale = sum(1 for pc in pitch_classes if pc in transposed)
                fit = in_scale / len(pitch_classes)

                if fit > best_fit:
                    best_fit = fit
                    best_scale = scale_name
                    best_root = root

        return best_fit, best_scale, best_root

    def compute_pitch_vocabulary(self, pitch_classes: List[int]) -> Tuple[int, float]:
        """
        Count unique pitch classes and score.

        Ambient typically uses 4-7 pitch classes (pentatonic to modal).
        """
        unique_pcs = len(set(pitch_classes))

        min_ideal, max_ideal = self.ideal_pitch_classes
        if min_ideal <= unique_pcs <= max_ideal:
            score = 1.0
        elif unique_pcs < min_ideal:
            score = unique_pcs / min_ideal  # Too few
        else:
            # Too many - penalize more heavily
            score = max(0, 1 - (unique_pcs - max_ideal) / 5)

        return unique_pcs, score

    def compute_voice_stasis(self, notes: List[dict]) -> float:
        """
        Measure how static the most static voice is.

        Returns: 0-1 (1 = completely static drone)
        """
        voices = self.get_pitches_by_voice(notes)

        if not voices:
            return 1.0

        stasis_scores = []
        for voice_id, voice_notes in voices.items():
            if len(voice_notes) < 2:
                stasis_scores.append(1.0)
                continue

            # Sort by start time
            sorted_notes = sorted(voice_notes, key=lambda x: x["start_tick"])
            pitches = [n["pitch"] for n in sorted_notes]

            # Count pitch changes
            changes = sum(
                1 for i in range(1, len(pitches)) if pitches[i] != pitches[i - 1]
            )
            change_rate = changes / (len(pitches) - 1)
            stasis_scores.append(1 - change_rate)

        # Return best (most static) voice
        return max(stasis_scores) if stasis_scores else 1.0

    def compute_consonance(self, notes: List[dict]) -> float:
        """
        Measure consonance of simultaneous notes.

        Analyzes vertical intervals at each note onset.
        """
        if len(notes) < 2:
            return 1.0

        # Build timeline of active notes at each onset
        onsets = sorted(set(n["start_tick"] for n in notes))
        consonance_scores = []

        for onset in onsets:
            # Find notes active at this onset
            active_pitches = []
            for n in notes:
                if n["start_tick"] <= onset < n["end_tick"]:
                    active_pitches.append(n["pitch"])
                # Also include notes starting at this onset
                elif n["start_tick"] == onset:
                    active_pitches.append(n["pitch"])

            active_pitches = list(set(active_pitches))  # Dedupe

            if len(active_pitches) < 2:
                continue

            # Compute all pairwise intervals
            intervals = []
            for i in range(len(active_pitches)):
                for j in range(i + 1, len(active_pitches)):
                    interval = abs(active_pitches[i] - active_pitches[j]) % 12
                    intervals.append(interval)

            if intervals:
                consonant = sum(1 for iv in intervals if iv in CONSONANT_INTERVALS)
                consonance_scores.append(consonant / len(intervals))

        return np.mean(consonance_scores) if consonance_scores else 1.0

    def compute_sparsity(self, notes: List[dict]) -> float:
        """
        Measure average voice sparsity.

        Returns: 0-1 (1 = very sparse, few voices active at once)
        """
        voices = self.get_pitches_by_voice(notes)
        n_voices = len(voices)

        if n_voices <= 1:
            return 1.0

        # Sample activity at each onset
        onsets = sorted(set(n["start_tick"] for n in notes))
        sparsity_scores = []

        for onset in onsets:
            active_voices = 0
            for voice_id, voice_notes in voices.items():
                for n in voice_notes:
                    if n["start_tick"] <= onset < n["end_tick"]:
                        active_voices += 1
                        break
                    elif n["start_tick"] == onset:
                        active_voices += 1
                        break

            # Sparsity = fraction of voices NOT active
            sparsity_scores.append(1 - active_voices / n_voices)

        avg_sparsity = np.mean(sparsity_scores) if sparsity_scores else 0.5

        # Score based on distance from target
        return 1 - abs(avg_sparsity - self.target_sparsity)

    def compute_cadence_penalty(self, notes: List[dict]) -> float:
        """
        Penalize V-I (dominant-tonic) bass motion.

        Returns: 0-1 (1 = no cadences, 0 = many cadences)
        """
        voices = self.get_pitches_by_voice(notes)

        if not voices:
            return 1.0

        # Find lowest voice (by average pitch)
        lowest_voice = None
        lowest_avg = float("inf")
        for voice_id, voice_notes in voices.items():
            avg_pitch = np.mean([n["pitch"] for n in voice_notes])
            if avg_pitch < lowest_avg:
                lowest_avg = avg_pitch
                lowest_voice = voice_notes

        if not lowest_voice or len(lowest_voice) < 2:
            return 1.0

        # Sort by time and get pitch sequence
        sorted_notes = sorted(lowest_voice, key=lambda x: x["start_tick"])
        pitches = [n["pitch"] % 12 for n in sorted_notes]

        # Count cadential intervals (P5 down = 7 semitones, P4 up = 5)
        cadences = 0
        for i in range(1, len(pitches)):
            interval = (pitches[i] - pitches[i - 1]) % 12
            if interval == 5 or interval == 7:  # P4 up or P5 down
                cadences += 1

        cadence_rate = cadences / (len(pitches) - 1)
        return 1 - cadence_rate

    def compute_melodic_smoothness(self, notes: List[dict]) -> float:
        """
        Measure smoothness of melodic lines per voice.
        Ambient favors stepwise motion over large leaps.

        Returns: 0-1 (1 = mostly stepwise, 0 = lots of leaps)
        """
        voices = self.get_pitches_by_voice(notes)

        if not voices:
            return 0.5

        smoothness_scores = []

        for voice_notes in voices.values():
            if len(voice_notes) < 2:
                continue

            sorted_notes = sorted(voice_notes, key=lambda x: x["start_tick"])
            intervals = []

            for i in range(1, len(sorted_notes)):
                interval = abs(sorted_notes[i]["pitch"] - sorted_notes[i - 1]["pitch"])
                intervals.append(interval)

            if intervals:
                # Stepwise = 1-2 semitones
                stepwise = sum(1 for i in intervals if i <= 2)
                # Large leaps = > 5 semitones
                large_leaps = sum(1 for i in intervals if i > 5)

                # Score: reward stepwise, penalize leaps
                score = (stepwise - large_leaps) / len(intervals)
                smoothness_scores.append(
                    max(0, min(1, (score + 1) / 2))
                )  # Scale to 0-1

        return np.mean(smoothness_scores) if smoothness_scores else 0.5

    def compute_harmonic_rhythm(self, notes: List[dict]) -> float:
        """
        Measure stability of vertical harmony over time.
        Ambient has slow harmonic rhythm (chords change infrequently).

        Returns: 0-1 (1 = slow/stable, 0 = rapid changes)
        """
        if len(notes) < 2:
            return 1.0

        onsets = sorted(set(n["start_tick"] for n in notes))

        if len(onsets) < 2:
            return 1.0

        # Get pitch class set at each onset
        pc_sets = []
        for onset in onsets:
            pcs = set()
            for n in notes:
                if n["start_tick"] <= onset < n["end_tick"]:
                    pcs.add(n["pitch"] % 12)
                elif n["start_tick"] == onset:
                    pcs.add(n["pitch"] % 12)
            if pcs:  # Only add non-empty sets
                pc_sets.append(frozenset(pcs))

        if len(pc_sets) < 2:
            return 1.0

        # Count how often the pitch class set changes
        changes = sum(1 for i in range(1, len(pc_sets)) if pc_sets[i] != pc_sets[i - 1])
        stability = 1 - (changes / (len(pc_sets) - 1))

        return stability

    def compute_note_duration(self, notes: List[dict]) -> float:
        """
        Measure relative note duration (prefer longer notes).
        Tempo-invariant - just compares distribution.

        Returns: 0-1 (higher = longer average durations)
        """
        if not notes:
            return 0.0

        durations = [n["end_tick"] - n["start_tick"] for n in notes]

        if not durations:
            return 0.0

        # Use percentiles to avoid tempo dependence
        median_duration = np.median(durations)
        max_duration = np.max(durations)

        # Score based on median relative to max
        # If median is close to max, notes tend to be long
        if max_duration == 0:
            return 0.0

        score = median_duration / max_duration

        return score

    def _get_simultaneous_notes(
        self, voice1_notes: List[dict], voice2_notes: List[dict]
    ) -> List[Tuple[int, int]]:
        """
        Helper: find pairs of pitches that sound simultaneously in two voices.
        Returns list of (pitch1, pitch2) tuples.
        """
        simultaneous = []

        for n1 in voice1_notes:
            for n2 in voice2_notes:
                # Check if notes overlap in time
                overlap = not (
                    n1["end_tick"] <= n2["start_tick"]
                    or n2["end_tick"] <= n1["start_tick"]
                )
                if overlap:
                    simultaneous.append((n1["pitch"], n2["pitch"]))
                    break  # Only count each n1 once

        return simultaneous

    def compute_voice_independence(self, notes: List[dict]) -> float:
        """
        Penalize parallel motion between voices (especially fifths/octaves).

        Returns: 0-1 (1 = independent, 0 = lots of parallels)
        """
        voices = self.get_pitches_by_voice(notes)
        voice_list = list(voices.values())

        if len(voice_list) < 2:
            return 1.0

        total_penalties = 0
        total_comparisons = 0

        # Check all pairs of voices
        for i in range(len(voice_list)):
            for j in range(i + 1, len(voice_list)):
                voice1 = sorted(voice_list[i], key=lambda x: x["start_tick"])
                voice2 = sorted(voice_list[j], key=lambda x: x["start_tick"])

                # Build list of intervals at each common timestamp
                intervals = []
                for n1 in voice1:
                    # Find closest n2 that overlaps
                    for n2 in voice2:
                        if not (
                            n1["end_tick"] <= n2["start_tick"]
                            or n2["end_tick"] <= n1["start_tick"]
                        ):
                            interval = (n1["pitch"] - n2["pitch"]) % 12
                            intervals.append(interval)
                            break

                # Check for parallel perfect intervals
                for k in range(1, len(intervals)):
                    prev_interval = intervals[k - 1]
                    curr_interval = intervals[k]

                    # Parallel perfect intervals (0=unison, 7=fifth)
                    if prev_interval == curr_interval and prev_interval in {0, 7}:
                        total_penalties += 1
                    total_comparisons += 1

        if total_comparisons == 0:
            return 1.0

        return 1 - (total_penalties / total_comparisons)

    def analyze(self, midi_path: str) -> AmbientMetrics:
        """
        Analyze a MIDI file and return ambient metrics.

        Args:
            midi_path: Path to MIDI file

        Returns:
            AmbientMetrics dataclass with all scores
        """
        notes = self.load_midi(midi_path)

        if not notes:
            return AmbientMetrics(
                modal_consistency=0.0,
                best_scale="none",
                best_root=0,
                pitch_vocabulary=0,
                pitch_vocabulary_score=0.0,
                voice_stasis=0.0,
                melodic_smoothness=0.0,
                harmonic_rhythm=0.0,
                note_duration=0.0,
                consonance=0.0,
                sparsity=0.0,
                voice_independence=0.0,
                cadence_penalty=0.0,
                composite_score=0.0,
            )

        pitch_classes = self.get_pitch_classes(notes)

        # Compute all metrics
        modal, best_scale, best_root = self.compute_modal_consistency(pitch_classes)
        vocab, vocab_score = self.compute_pitch_vocabulary(pitch_classes)
        stasis = self.compute_voice_stasis(notes)
        melodic_smooth = self.compute_melodic_smoothness(notes)
        harmonic_rhythm = self.compute_harmonic_rhythm(notes)
        note_duration = self.compute_note_duration(notes)
        consonance = self.compute_consonance(notes)
        sparsity = self.compute_sparsity(notes)
        independence = self.compute_voice_independence(notes)
        cadence = self.compute_cadence_penalty(notes)

        # Weighted composite
        composite = (
            self.weights["modal"] * modal
            + self.weights["vocabulary"] * vocab_score
            + self.weights["stasis"] * stasis
            + self.weights["melodic_smooth"] * melodic_smooth
            + self.weights["harmonic_rhythm"] * harmonic_rhythm
            + self.weights["note_duration"] * note_duration
            + self.weights["consonance"] * consonance
            + self.weights["sparsity"] * sparsity
            + self.weights["independence"] * independence
            + self.weights["cadence"] * cadence
        )

        return AmbientMetrics(
            modal_consistency=modal,
            best_scale=best_scale,
            best_root=best_root,
            pitch_vocabulary=vocab,
            pitch_vocabulary_score=vocab_score,
            voice_stasis=stasis,
            melodic_smoothness=melodic_smooth,
            harmonic_rhythm=harmonic_rhythm,
            note_duration=note_duration,
            consonance=consonance,
            sparsity=sparsity,
            voice_independence=independence,
            cadence_penalty=cadence,
            composite_score=composite,
        )

    def analyze_batch(self, midi_paths: List[str]) -> List[AmbientMetrics]:
        """Analyze multiple MIDI files."""
        return [self.analyze(path) for path in midi_paths]


# Convenience function
def evaluate_ambient(midi_path: str, **kwargs) -> AmbientMetrics:
    """Quick evaluation of a single MIDI file."""
    analyzer = AmbientAnalyzer(**kwargs)
    return analyzer.analyze(midi_path)


def print_summary_table(results: List[Tuple[str, AmbientMetrics]]):
    """Print a summary table of all analyzed files."""
    import os

    root_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    print("\n" + "=" * 160)
    print("SUMMARY TABLE")
    print("=" * 160)

    # Header
    print(
        f"{'File':<35} {'Composite':>9} {'Modal':>7} {'MeloSm':>7} {'HarmRh':>7} "
        f"{'NoteDur':>7} {'Stasis':>7} {'Conson':>7} {'Indep':>7} {'Cadence':>7} {'Scale':<15}"
    )
    print("-" * 160)

    # Rows
    for filepath, metrics in results:
        filename = os.path.basename(filepath)
        if len(filename) > 32:
            filename = filename[:29] + "..."

        scale_str = f"{root_names[metrics.best_root]} {metrics.best_scale}"
        if len(scale_str) > 15:
            scale_str = scale_str[:12] + "..."

        print(
            f"{filename:<35} {metrics.composite_score:>9.3f} {metrics.modal_consistency:>7.3f} "
            f"{metrics.melodic_smoothness:>7.3f} {metrics.harmonic_rhythm:>7.3f} "
            f"{metrics.note_duration:>7.3f} {metrics.voice_stasis:>7.3f} "
            f"{metrics.consonance:>7.3f} {metrics.voice_independence:>7.3f} "
            f"{metrics.cadence_penalty:>7.3f} {scale_str:<15}"
        )

    print("=" * 160)

    # Statistics
    if len(results) > 1:
        composites = [m.composite_score for _, m in results]
        print(f"\nStatistics:")
        print(f"  Mean composite score: {np.mean(composites):.3f}")
        print(f"  Std dev:              {np.std(composites):.3f}")
        print(f"  Min:                  {np.min(composites):.3f}")
        print(f"  Max:                  {np.max(composites):.3f}")
        print()


if __name__ == "__main__":
    import sys
    import glob
    import os

    if len(sys.argv) < 2:
        # Default: analyze any MIDI files in current directory
        midi_files = glob.glob("*.mid") + glob.glob("*.midi")
    else:
        # Check if argument is a directory
        if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
            dir_path = sys.argv[1]
            midi_files = glob.glob(os.path.join(dir_path, "*.mid")) + glob.glob(
                os.path.join(dir_path, "*.midi")
            )
        else:
            midi_files = sys.argv[1:]

    if not midi_files:
        print("Usage: python eval_ambient.py <midi_file> [midi_file2 ...]")
        print("       python eval_ambient.py <directory>")
        print("       or place .mid files in current directory")
        sys.exit(1)

    analyzer = AmbientAnalyzer()
    results = []

    for midi_path in midi_files:
        print(f"\n{'='*60}")
        print(f"File: {midi_path}")
        print("=" * 60)
        try:
            metrics = analyzer.analyze(midi_path)
            print(metrics)
            results.append((midi_path, metrics))
        except Exception as e:
            print(f"Error: {e}")

    # Print summary table if multiple files
    if len(results) > 1:
        print_summary_table(results)
