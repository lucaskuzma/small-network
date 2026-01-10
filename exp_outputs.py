# %%
"""
Output readout experiments.
Each readout is a voice with 12 outputs (chromatic scale).
"""

import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork

# %%
# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NUM_NEURONS = 256
NUM_READOUTS = 4
N_OUTPUTS_PER_READOUT = 12
SIM_STEPS = 256

# Create network with multiple readouts
net = NeuralNetwork(
    num_neurons=NUM_NEURONS,
    num_readouts=NUM_READOUTS,
    n_outputs_per_readout=N_OUTPUTS_PER_READOUT,
)

print(
    f"Network: {NUM_NEURONS} neurons, {NUM_READOUTS} readouts × {N_OUTPUTS_PER_READOUT} outputs"
)
print(f"Total outputs: {net.state.num_outputs}")

# %%
# Initialize network weights (same as clustering experiment)
net.randomize_weights(sparsity=0.1, scale=0.4)
net.randomize_output_weights(sparsity=0.1, scale=0.2)
net.randomize_thresholds()
net.set_diagonal_weights(0)
net.enable_activation_leak(0.9)
net.enable_refraction_decay(2, 0.75, 32)

print(f"Spectral radius: {net.get_spectral_radius():.3f}")

# %%
# Run simulation and record outputs
output_history = np.zeros((SIM_STEPS, NUM_READOUTS, N_OUTPUTS_PER_READOUT))
firing_history = np.zeros((SIM_STEPS, NUM_NEURONS), dtype=bool)

# Initial activation - activate the most connected neuron
net.manual_activate_most_weighted(1.0)

# Run simulation
for step in range(SIM_STEPS):
    net.tick(step)

    # Record outputs reshaped by readout
    output_history[step] = net.get_readout_outputs()
    firing_history[step] = net.state.firing

print(f"Simulation complete: {np.sum(firing_history)} total firing events")

# %%
# Plot output activations over time for each readout
fig, axes = plt.subplots(NUM_READOUTS, 1, figsize=(14, 3 * NUM_READOUTS), sharex=True)

note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

for r in range(NUM_READOUTS):
    ax = axes[r] if NUM_READOUTS > 1 else axes

    # Heatmap of outputs over time
    im = ax.imshow(
        output_history[:, r, :].T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=1,
        extent=[0, SIM_STEPS, -0.5, N_OUTPUTS_PER_READOUT - 0.5],
    )

    ax.set_ylabel(f"Voice {r + 1}")
    ax.set_yticks(range(N_OUTPUTS_PER_READOUT))
    ax.set_yticklabels(note_names)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Activation", shrink=0.8)

axes[-1].set_xlabel("Time Step")
fig.suptitle("Output Activations by Readout (Voice)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Plot combined activity summary
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Top left: Total activation per readout over time
ax = axes[0, 0]
for r in range(NUM_READOUTS):
    total_activation = np.sum(output_history[:, r, :], axis=1)
    ax.plot(total_activation, label=f"Voice {r + 1}", alpha=0.8)
ax.set_xlabel("Time Step")
ax.set_ylabel("Total Activation")
ax.set_title("Total Activation per Voice")
ax.legend()
ax.grid(True, alpha=0.3)

# Top right: Which note is most active per readout over time
ax = axes[0, 1]
for r in range(NUM_READOUTS):
    # Get the dominant note at each timestep (argmax)
    dominant_notes = np.argmax(output_history[:, r, :], axis=1)
    # Mask out timesteps with no activation
    max_activation = np.max(output_history[:, r, :], axis=1)
    dominant_notes = np.where(max_activation > 0.1, dominant_notes, np.nan)
    ax.scatter(
        range(SIM_STEPS),
        dominant_notes + r * 0.15,
        s=3,
        label=f"Voice {r + 1}",
        alpha=0.6,
    )
ax.set_xlabel("Time Step")
ax.set_ylabel("Dominant Note")
ax.set_yticks(range(N_OUTPUTS_PER_READOUT))
ax.set_yticklabels(note_names)
ax.set_title("Dominant Note per Voice")
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom left: Firing activity (neurons)
ax = axes[1, 0]
firing_count = np.sum(firing_history, axis=1)
ax.plot(firing_count, color="darkgreen", alpha=0.8)
ax.fill_between(range(SIM_STEPS), firing_count, alpha=0.3, color="green")
ax.set_xlabel("Time Step")
ax.set_ylabel("Firing Neurons")
ax.set_title("Network Firing Activity")
ax.grid(True, alpha=0.3)

# Bottom right: Note distribution per readout
ax = axes[1, 1]
width = 0.2
x = np.arange(N_OUTPUTS_PER_READOUT)
for r in range(NUM_READOUTS):
    note_totals = np.sum(output_history[:, r, :], axis=0)
    ax.bar(x + r * width, note_totals, width, label=f"Voice {r + 1}", alpha=0.8)
ax.set_xlabel("Note")
ax.set_ylabel("Total Activation")
ax.set_title("Note Distribution by Voice")
ax.set_xticks(x + width * (NUM_READOUTS - 1) / 2)
ax.set_xticklabels(note_names)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# %%
# Print statistics
print("\n=== Output Statistics ===")
for r in range(NUM_READOUTS):
    readout_data = output_history[:, r, :]
    total = np.sum(readout_data)
    peak = np.max(readout_data)
    active_steps = np.sum(np.max(readout_data, axis=1) > 0.1)
    most_active_note = note_names[np.argmax(np.sum(readout_data, axis=0))]
    print(
        f"Voice {r + 1}: total={total:.1f}, peak={peak:.2f}, active_steps={active_steps}, favorite_note={most_active_note}"
    )

# %%
# Sonification - live audio playback
import pygame
import time

percentile = 95

pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
pygame.init()

# Compute threshold per voice (top 20% of activations)
voice_thresholds = np.zeros(NUM_READOUTS)
for r in range(NUM_READOUTS):
    voice_data = output_history[:, r, :]  # (T, 12)
    voice_thresholds[r] = np.percentile(voice_data, percentile)
print(f"Voice thresholds {percentile}th percentile: {voice_thresholds}")

# Each voice gets its own octave
# Voice 0 = C3 (MIDI 48), Voice 1 = C4 (MIDI 60), etc.
base_notes = [48, 60, 72, 84]  # C3, C4, C5, C6

tempo = 60
step_duration = 60.0 / tempo / 4  # 16th notes

print(f"Playing {SIM_STEPS} steps at {tempo} BPM...")

for step in range(SIM_STEPS):
    mixed_audio = np.zeros(int(44100 * step_duration))

    for voice in range(NUM_READOUTS):
        for note_idx in range(N_OUTPUTS_PER_READOUT):
            activation = output_history[step, voice, note_idx]

            if activation > voice_thresholds[voice]:
                # MIDI note = base octave + semitone offset
                midi_note = base_notes[voice] + note_idx
                frequency = 440 * (2 ** ((midi_note - 69) / 12))

                # Generate sine wave
                t = np.linspace(0, step_duration, len(mixed_audio))
                wave = np.sin(2 * np.pi * frequency * t) * activation

                # Fade out
                wave *= np.linspace(1, 0, len(t))

                mixed_audio += wave

    # Normalize and play
    if np.max(np.abs(mixed_audio)) > 0:
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.8
    audio = (mixed_audio * 32767).astype(np.int16)

    sound = pygame.sndarray.make_sound(audio)
    sound.play()

    time.sleep(step_duration)

print("Playback complete.")

# %%
# Export to MIDI with chromatic outputs
from datetime import datetime
from utils_sonic import save_readout_outputs_as_midi

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
midi_filename = f"midi/neural_output_{timestamp}.mid"

# Use the same thresholds and settings as playback
save_readout_outputs_as_midi(
    output_history,
    filename=midi_filename,
    tempo=60,
    threshold=voice_thresholds,
    base_notes=[48, 60, 72, 84],  # C3, C4, C5, C6
)

# %%
# Evaluate MIDI file and plot metrics
from eval_ambient import evaluate_ambient
from dataclasses import asdict

metrics_obj = evaluate_ambient(midi_filename)
print(metrics_obj)

# Convert to dict for plotting (exclude non-metric fields)
metrics_dict = asdict(metrics_obj)

# Remove non-numeric/non-plottable fields
exclude_fields = ["best_scale", "best_root", "pitch_vocabulary", "composite_score"]
plot_metrics = {k: v for k, v in metrics_dict.items() if k not in exclude_fields}

# Plot individual metrics and composite score
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Individual metrics
axes[0].bar(range(len(plot_metrics)), list(plot_metrics.values()))
axes[0].set_xticks(range(len(plot_metrics)))
axes[0].set_xticklabels(plot_metrics.keys(), rotation=45, ha="right")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Score")
axes[0].set_title("Individual Ambient Metrics")
axes[0].grid(True, alpha=0.3, axis="y")

# Composite score
axes[1].bar(["Composite"], [metrics_obj.composite_score], color="green", width=0.5)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Score")
axes[1].set_title(f"Overall Score: {metrics_obj.composite_score:.3f}")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# %%
# Compare metrics across all MIDI files in folder
import glob
import os

midi_files = sorted(glob.glob("midi/*.mid"))
all_metrics = []
filenames = []

for mf in midi_files:
    try:
        m = evaluate_ambient(mf)
        all_metrics.append(m)
        filenames.append(os.path.basename(mf))
    except Exception as e:
        print(f"Error analyzing {mf}: {e}")

if all_metrics:
    # Extract metric names (excluding non-plottable fields)
    exclude_fields = ["best_scale", "best_root", "pitch_vocabulary"]
    metric_names = [k for k in asdict(all_metrics[0]).keys() if k not in exclude_fields]

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(filenames))
    width = 0.8 / len(metric_names)

    # Color neural outputs differently
    colors = plt.cm.tab20(np.linspace(0, 1, len(metric_names)))

    for i, metric_name in enumerate(metric_names):
        values = [getattr(m, metric_name) for m in all_metrics]
        offset = (i - len(metric_names) / 2) * width
        ax.bar(x + offset, values, width, label=metric_name, alpha=0.8, color=colors[i])

    ax.set_xlabel("MIDI File")
    ax.set_ylabel("Score")
    ax.set_title("Ambient Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(filenames, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # Highlight neural outputs with background color
    for i, fn in enumerate(filenames):
        if "neural_output" in fn:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="red")

    plt.tight_layout()
    plt.show()

    # Plot composite scores comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    composite_scores = [m.composite_score for m in all_metrics]
    colors_composite = [
        "red" if "neural_output" in fn else "steelblue" for fn in filenames
    ]

    bars = ax.bar(
        range(len(filenames)), composite_scores, color=colors_composite, alpha=0.7
    )
    ax.set_xlabel("MIDI File")
    ax.set_ylabel("Composite Score")
    ax.set_title("Composite Ambient Score Comparison (Red = Neural Network)")
    ax.set_xticks(range(len(filenames)))
    ax.set_xticklabels(filenames, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add mean line for benchmark (non-neural) files
    benchmark_scores = [
        s for i, s in enumerate(composite_scores) if "neural_output" not in filenames[i]
    ]
    if benchmark_scores:
        mean_benchmark = np.mean(benchmark_scores)
        ax.axhline(
            mean_benchmark,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Benchmark Mean: {mean_benchmark:.3f}",
        )
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Benchmark files (Dark Ambient Piano):")
    print(f"  Count: {len(benchmark_scores)}")
    print(f"  Mean: {np.mean(benchmark_scores):.3f}")
    print(f"  Std: {np.std(benchmark_scores):.3f}")
    print(f"  Range: {np.min(benchmark_scores):.3f} - {np.max(benchmark_scores):.3f}")

    neural_scores = [
        s for i, s in enumerate(composite_scores) if "neural_output" in filenames[i]
    ]
    if neural_scores:
        print(f"\nNeural network outputs:")
        print(f"  Count: {len(neural_scores)}")
        print(f"  Mean: {np.mean(neural_scores):.3f}")
        print(f"  Std: {np.std(neural_scores):.3f}")
        print(f"  Range: {np.min(neural_scores):.3f} - {np.max(neural_scores):.3f}")
        print(
            f"  Gap from benchmark: {np.mean(benchmark_scores) - np.mean(neural_scores):.3f}"
        )
