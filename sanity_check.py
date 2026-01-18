"""
Sanity check: Can random networks score well at all?

If 1000 random networks all score 0.3-0.5, the problem is the architecture,
not evolution. If some score 0.7+, evolution should be able to find them.
"""

import numpy as np
from tqdm import tqdm
from network import NetworkGenotype
from evolve import evaluate_genotype, EvolutionConfig


def run_sanity_check(n_samples: int = 500, encoding: str = "motion", seed: int = 45):
    """Generate random networks and see what scores are achievable."""

    np.random.seed(seed)

    n_outputs = 8 if encoding == "motion" else 12
    config = EvolutionConfig(encoding=encoding, evaluator="basic")

    print(f"Generating {n_samples} random networks...")
    print(f"Encoding: {encoding} ({n_outputs} outputs per voice)")
    print("-" * 60)

    scores = []
    best_score = 0
    best_modal = 0
    best_activity = 0
    best_genotype = None
    worst_score = 1.0
    worst_genotype = None

    for i in tqdm(range(n_samples), desc="Random search"):
        g = NetworkGenotype.random(n_outputs_per_readout=n_outputs)
        r = evaluate_genotype(g, config)

        scores.append(
            {
                "fitness": r.fitness,
                "modal": r.modal_consistency,
                "activity": r.activity,
                "notes": r.note_count,
            }
        )

        # Report new bests
        if r.fitness > best_score:
            best_score = r.fitness
            best_modal = r.modal_consistency
            best_activity = r.activity
            best_genotype = g
            tqdm.write(
                f"  [{i+1:5d}] NEW BEST: {r.fitness:.4f} | "
                f"modal:{r.modal_consistency:.2f} act:{r.activity:.2f} | "
                f"notes:{r.note_count}"
            )

        # Report new worsts
        if r.fitness < worst_score:
            worst_score = r.fitness
            worst_genotype = g
            tqdm.write(
                f"  [{i+1:5d}] NEW WORST: {r.fitness:.4f} | "
                f"modal:{r.modal_consistency:.2f} act:{r.activity:.2f} | "
                f"notes:{r.note_count}"
            )

    # Summary statistics
    print("-" * 60)
    print("RESULTS:")
    print(f"  Samples: {n_samples}")

    fitnesses = [s["fitness"] for s in scores]
    modals = [s["modal"] for s in scores]
    activities = [s["activity"] for s in scores]

    print(f"\n  FITNESS:")
    print(f"    Best:   {max(fitnesses):.4f}")
    print(f"    Mean:   {np.mean(fitnesses):.4f}")
    print(f"    Std:    {np.std(fitnesses):.4f}")
    print(f"    Median: {np.median(fitnesses):.4f}")

    print(f"\n  MODAL CONSISTENCY:")
    print(f"    Best:   {max(modals):.4f}")
    print(f"    Mean:   {np.mean(modals):.4f}")
    print(
        f"    >0.5:   {sum(1 for m in modals if m > 0.5)} ({100*sum(1 for m in modals if m > 0.5)/len(modals):.1f}%)"
    )
    print(
        f"    >0.7:   {sum(1 for m in modals if m > 0.7)} ({100*sum(1 for m in modals if m > 0.7)/len(modals):.1f}%)"
    )

    print(f"\n  ACTIVITY:")
    print(f"    Best:   {max(activities):.4f}")
    print(f"    Mean:   {np.mean(activities):.4f}")

    note_counts = [s["notes"] for s in scores]
    print(f"\n  NOTE COUNTS:")
    print(f"    Min:    {min(note_counts)}")
    print(f"    Max:    {max(note_counts)}")
    print(f"    Mean:   {np.mean(note_counts):.1f}")
    print(
        f"    Zero:   {sum(1 for n in note_counts if n == 0)} ({100*sum(1 for n in note_counts if n == 0)/len(note_counts):.1f}%)"
    )

    # Distribution buckets
    print(f"\n  FITNESS DISTRIBUTION:")
    buckets = [
        (0, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 1.0),
    ]
    for lo, hi in buckets:
        count = sum(1 for f in fitnesses if lo <= f < hi)
        pct = 100 * count / len(fitnesses)
        bar = "█" * int(pct / 2)
        print(f"    {lo:.1f}-{hi:.1f}: {count:4d} ({pct:5.1f}%) {bar}")

    # Save best MIDI
    if best_genotype is not None:
        import os

        os.makedirs("sanity_check_output", exist_ok=True)
        midi_path = f"sanity_check_output/best_{best_score:.4f}.mid"
        evaluate_genotype(
            best_genotype, config, save_midi=True, midi_filename=midi_path
        )
        print(f"\n  BEST MIDI SAVED: {midi_path}")

    return scores


if __name__ == "__main__":
    import sys

    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    encoding = sys.argv[2] if len(sys.argv) > 2 else "motion"

    run_sanity_check(n_samples=n_samples, encoding=encoding)
