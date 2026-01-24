"""
Audio synthesis experiments with spiking neural networks.

Quick iteration script for testing network -> audio pipeline
before integrating with full evolution.
"""

import os
import numpy as np
from datetime import datetime

from network import NeuralNetwork, NetworkGenotype
from utils_audio import (
    synthesize_oscillators_vectorized,
    save_wav,
    save_waveform_plot,
    save_frequency_plot,
    save_spectrogram,
    DEFAULT_SAMPLE_RATE,
)
from eval_audio import evaluate_audio, AudioMetrics


# =============================================================================
# Configuration
# =============================================================================

# Audio parameters
SAMPLE_RATE = 11025  # Quick iteration (44100 for final quality)
DURATION_SECONDS = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)

# Network parameters for audio mode
NUM_NEURONS = 16  # Small for fast iteration
NUM_VOICES = 3  # 3 oscillators
OUTPUTS_PER_VOICE = 3  # [freq, phase_mod, amplitude]

# Output directory
OUTPUT_DIR = "exp_audio_output"


# =============================================================================
# Network setup
# =============================================================================


def create_audio_network(
    num_neurons: int = NUM_NEURONS,
    num_voices: int = NUM_VOICES,
) -> NeuralNetwork:
    """Create a network configured for audio output."""
    net = NeuralNetwork(
        num_neurons=num_neurons,
        num_readouts=num_voices,
        n_outputs_per_readout=OUTPUTS_PER_VOICE,
    )

    # Randomize with settings tuned for audio
    # Higher sparsity than MIDI experiments - we need more activity
    net.randomize_weights(sparsity=0.3, scale=0.5)
    net.randomize_output_weights(sparsity=0.3, scale=0.5)
    net.randomize_thresholds()
    net.set_diagonal_weights(0)

    # Enable dynamics
    net.enable_activation_leak(0.98)
    net.enable_refraction_decay(
        refraction_period=4,
        refraction_leak=0.75,
        refraction_variation=8,
    )
    net.set_weight_threshold(0.05)

    return net


def create_audio_genotype(
    num_neurons: int = NUM_NEURONS,
    num_voices: int = NUM_VOICES,
) -> NetworkGenotype:
    """Create a random genotype for audio synthesis."""
    net = create_audio_network(num_neurons, num_voices)
    return NetworkGenotype.from_network(net)


# =============================================================================
# Simulation
# =============================================================================


def run_simulation(
    genotype: NetworkGenotype,
    num_samples: int = NUM_SAMPLES,
) -> np.ndarray:
    """
    Run network simulation and collect outputs.

    Args:
        genotype: Network genotype
        num_samples: Number of timesteps (= audio samples)

    Returns:
        output_history: (T, num_voices, 3) array
    """
    net = genotype.to_network()
    num_voices = genotype.num_readouts
    outputs_per_voice = genotype.n_outputs_per_readout

    output_history = np.zeros((num_samples, num_voices, outputs_per_voice))

    # Initial activation - kick the most connected neuron
    total_weights = np.sum(np.abs(net.state.network_weights), axis=1)
    most_connected = np.argmax(total_weights)
    net.manual_activate(most_connected, 1.0)

    # Run simulation
    for step in range(num_samples):
        net.tick(step)
        output_history[step] = net.get_readout_outputs()

    return output_history


def plot_output_activations(
    output_history: np.ndarray,
    filename: str,
    sample_rate: int = SAMPLE_RATE,
):
    """Plot raw output values over time for each voice and channel."""
    import matplotlib.pyplot as plt

    T, num_voices, num_outputs = output_history.shape
    time_axis = np.arange(T) / sample_rate
    labels = ["Freq", "Phase", "Amp"]

    fig, axes = plt.subplots(
        num_voices, num_outputs, figsize=(14, 3 * num_voices), sharex=True
    )

    for v in range(num_voices):
        for o in range(num_outputs):
            ax = axes[v, o] if num_voices > 1 else axes[o]
            ax.plot(time_axis, output_history[:, v, o], linewidth=0.5, color="C0")
            ax.set_ylabel(f"V{v} {labels[o]}")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            vals = output_history[:, v, o]
            ax.set_title(
                f"min={vals.min():.3f} max={vals.max():.3f} mean={vals.mean():.3f}",
                fontsize=8,
            )

    if num_voices > 1:
        for o in range(num_outputs):
            axes[-1, o].set_xlabel("Time (s)")
    else:
        for o in range(num_outputs):
            axes[o].set_xlabel("Time (s)")

    plt.suptitle("Raw Network Output Activations")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# =============================================================================
# Full pipeline
# =============================================================================


def synthesize_and_evaluate(
    genotype: NetworkGenotype,
    output_dir: str,
    name: str = "test",
    sample_rate: int = SAMPLE_RATE,
) -> tuple[AudioMetrics, str]:
    """
    Full pipeline: simulate -> synthesize -> evaluate -> save.

    Args:
        genotype: Network genotype
        output_dir: Directory to save outputs
        name: Base name for output files
        sample_rate: Audio sample rate

    Returns:
        metrics: AudioMetrics from evaluation
        wav_path: Path to saved WAV file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Run simulation
    num_samples = int(sample_rate * DURATION_SECONDS)
    print(f"Running simulation ({num_samples} samples)...")
    output_history = run_simulation(genotype, num_samples)

    # Plot raw outputs
    outputs_path = os.path.join(output_dir, f"{name}_outputs.png")
    plot_output_activations(output_history, outputs_path, sample_rate)
    print(f"  Outputs: {outputs_path}")

    # Evaluate fitness
    print("Evaluating...")
    metrics = evaluate_audio(output_history)

    # Synthesize audio
    print("Synthesizing audio...")
    audio, freqs = synthesize_oscillators_vectorized(output_history, sample_rate)

    # Save outputs
    wav_path = os.path.join(output_dir, f"{name}.wav")
    save_wav(audio, wav_path, sample_rate)
    print(f"  WAV: {wav_path}")

    waveform_path = os.path.join(output_dir, f"{name}_waveform.png")
    save_waveform_plot(
        audio,
        waveform_path,
        sample_rate,
        title=f"{name} (fitness={metrics.composite_score:.3f})",
    )
    print(f"  Waveform: {waveform_path}")

    freq_path = os.path.join(output_dir, f"{name}_frequencies.png")
    save_frequency_plot(
        freqs, freq_path, sample_rate, title=f"{name} Voice Frequencies"
    )
    print(f"  Frequencies: {freq_path}")

    spec_path = os.path.join(output_dir, f"{name}_spectrogram.png")
    save_spectrogram(audio, spec_path, sample_rate, title=f"{name} Spectrogram")
    print(f"  Spectrogram: {spec_path}")

    return metrics, wav_path


# =============================================================================
# Experiments
# =============================================================================


def experiment_random_networks(
    n_networks: int = 10,
    output_dir: str = OUTPUT_DIR,
):
    """
    Generate and evaluate random networks to see fitness distribution.
    """
    print("=" * 60)
    print(f"Experiment: Random Networks (n={n_networks})")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"random_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    results = []

    for i in range(n_networks):
        print(f"\n--- Network {i+1}/{n_networks} ---")
        genotype = create_audio_genotype()
        metrics, wav_path = synthesize_and_evaluate(
            genotype, exp_dir, name=f"net_{i:03d}"
        )
        results.append(metrics)
        print(metrics)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    consonances = [m.consonance for m in results]
    activities = [m.activity for m in results]
    composites = [m.composite_score for m in results]

    print(f"Consonance: mean={np.mean(consonances):.3f}, std={np.std(consonances):.3f}")
    print(f"Activity:   mean={np.mean(activities):.3f}, std={np.std(activities):.3f}")
    print(f"Composite:  mean={np.mean(composites):.3f}, std={np.std(composites):.3f}")
    print(f"\nBest composite: {np.max(composites):.3f}")
    print(f"Worst composite: {np.min(composites):.3f}")

    return results


def experiment_single_network(
    seed: int = 42,
    output_dir: str = OUTPUT_DIR,
):
    """
    Generate and evaluate a single network with fixed seed for reproducibility.
    """
    print("=" * 60)
    print(f"Experiment: Single Network (seed={seed})")
    print("=" * 60)

    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"single_{timestamp}")

    genotype = create_audio_genotype()
    metrics, wav_path = synthesize_and_evaluate(genotype, exp_dir, name="single")

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(metrics)
    print(f"\nOutput saved to: {exp_dir}")

    return metrics


def experiment_mutation_effect(
    n_mutations: int = 20,
    output_dir: str = OUTPUT_DIR,
):
    """
    Start with one network and apply mutations to see effect on fitness.
    """
    print("=" * 60)
    print(f"Experiment: Mutation Effects (n={n_mutations})")
    print("=" * 60)

    np.random.seed(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"mutation_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Create parent
    parent = create_audio_genotype()
    parent_metrics, _ = synthesize_and_evaluate(parent, exp_dir, name="parent")
    print(f"\nParent fitness: {parent_metrics.composite_score:.3f}")

    # Generate mutations
    results = [("parent", parent_metrics)]

    for i in range(n_mutations):
        child = parent.mutate(
            weight_mutation_rate=0.1,
            weight_mutation_scale=0.1,
            threshold_mutation_rate=0.1,
            threshold_mutation_scale=0.1,
        )
        child_metrics, _ = synthesize_and_evaluate(
            child, exp_dir, name=f"child_{i:03d}"
        )
        results.append((f"child_{i:03d}", child_metrics))

        diff = child_metrics.composite_score - parent_metrics.composite_score
        indicator = "+" if diff > 0 else ""
        print(
            f"  Child {i}: {child_metrics.composite_score:.3f} ({indicator}{diff:.3f})"
        )

    # Summary
    child_scores = [m.composite_score for name, m in results[1:]]
    improvements = sum(1 for s in child_scores if s > parent_metrics.composite_score)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Parent: {parent_metrics.composite_score:.3f}")
    print(
        f"Children: mean={np.mean(child_scores):.3f}, best={np.max(child_scores):.3f}"
    )
    print(
        f"Improvements: {improvements}/{n_mutations} ({100*improvements/n_mutations:.0f}%)"
    )

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio synthesis experiments")
    parser.add_argument(
        "--exp",
        type=str,
        choices=["single", "random", "mutation"],
        default="single",
        help="Experiment type",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of networks/mutations to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    if args.exp == "single":
        experiment_single_network(seed=args.seed)
    elif args.exp == "random":
        experiment_random_networks(n_networks=args.n)
    elif args.exp == "mutation":
        experiment_mutation_effect(n_mutations=args.n)
