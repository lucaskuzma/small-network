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
    composite_score: float  # weighted combination

    def __str__(self) -> str:
        root_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return (
            f"BasicMetrics(\n"
            f"  composite: {self.composite_score:.3f}\n"
            f"  modal: {self.modal_consistency:.3f} ({root_names[self.best_root]} {self.best_scale})\n"
            f"  activity: {self.activity:.3f} ({self.note_count} notes, {self.note_density:.2f}/beat)\n"
            f")"
        )


class BasicAnalyzer:
    """
    Simple analyzer focused on modal consistency and activity.
    """

    def __init__(
        self,
        modal_weight: float = 0.5,
        activity_weight: float = 0.5,
        min_note_density: float = 0.5,
    ):
        """
        Args:
            modal_weight: Weight for modal consistency (0-1)
            activity_weight: Weight for activity (0-1)
            min_note_density: Notes per beat for full activity score
        """
        self.modal_weight = modal_weight
        self.activity_weight = activity_weight
        self.min_note_density = min_note_density

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
        Find best-fitting scale and return consistency score.

        For each of 12 roots × 11 scales, count what fraction of played notes
        fit the scale. Return the best fit.

        Returns: (score, scale_name, root)
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

        return best_fit, best_scale, best_root

    def compute_activity(
        self, notes: List[dict], duration_ticks: int, ticks_per_beat: int
    ) -> Tuple[float, float]:
        """
        Compute activity score based on note density (notes per beat).

        Returns: (activity_score, note_density)
            activity_score: 0-1 (0 = silent, 1 = enough activity)
            note_density: notes per beat
        """
        note_count = len(notes)

        if note_count == 0 or duration_ticks == 0:
            return 0.0, 0.0

        # Compute duration in beats
        duration_beats = duration_ticks / ticks_per_beat

        # Note density = notes per beat
        note_density = note_count / duration_beats

        if note_density >= self.min_note_density:
            return 1.0, note_density

        # Ramp from 0 to 1 as density approaches min_note_density
        # Use sqrt for gentler ramp
        score = np.sqrt(note_density / self.min_note_density)
        return score, note_density

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
                composite_score=0.0,
            )

        pitch_classes = [n["pitch"] % 12 for n in notes]

        # Compute metrics
        modal, best_scale, best_root = self.compute_modal_consistency(pitch_classes)
        activity, note_density = self.compute_activity(
            notes, duration_ticks, ticks_per_beat
        )

        # Weighted composite
        composite = self.modal_weight * modal + self.activity_weight * activity

        return BasicMetrics(
            modal_consistency=modal,
            best_scale=best_scale,
            best_root=best_root,
            activity=activity,
            note_density=note_density,
            note_count=len(notes),
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

