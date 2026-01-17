# %%
"""
Motion encoding experiments.
Each voice has 8 outputs: [u1, u4, u7, d1, d3, d8, v1, v2]
Motion = (u1 + u4×4 + u7×7) - (d1 + d3×3 + d8×8)
Velocity = v1 × v2 (soft AND)
"""

import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork

# %%
# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NUM_NEURONS = 256
NUM_VOICES = 4
N_OUTPUTS_PER_VOICE = 8  # Motion encoding: [u1, u4, u7, d1, d3, d8, v1, v2]
SIM_STEPS = 256

# Output labels
OUTPUT_NAMES = ["u1", "u4", "u7", "d1", "d3", "d8", "v1", "v2"]

# Create network with motion encoding outputs
net = NeuralNetwork(
    num_neurons=NUM_NEURONS,
    num_readouts=NUM_VOICES,
    n_outputs_per_readout=N_OUTPUTS_PER_VOICE,
)

print(f"Network: {NUM_NEURONS} neurons, {NUM_VOICES} voices × {N_OUTPUTS_PER_VOICE} outputs")
print(f"Total outputs: {net.state.num_outputs}")

# %%
# Initialize network weights
net.randomize_weights(sparsity=0.1, scale=0.4)
net.randomize_output_weights(sparsity=0.1, scale=0.2)
net.randomize_thresholds()
net.set_diagonal_weights(0)
net.enable_activation_leak(0.9)
net.enable_refraction_decay(2, 0.75, 32)

print(f"Spectral radius: {net.get_spectral_radius():.3f}")

# %%
# Run simulation and record outputs
output_history = np.zeros((SIM_STEPS, NUM_VOICES, N_OUTPUTS_PER_VOICE))
firing_history = np.zeros((SIM_STEPS, NUM_NEURONS), dtype=bool)

# Initial activation - activate the most connected neuron
net.manual_activate_most_weighted(1.0)

# Run simulation
for step in range(SIM_STEPS):
    net.tick(step)
    output_history[step] = net.get_readout_outputs()
    firing_history[step] = net.state.firing

print(f"Simulation complete: {np.sum(firing_history)} total firing events")

# %%
# Compute motion and velocity from outputs
up_weights = np.array([1, 4, 7])
down_weights = np.array([1, 3, 8])

# Compute motion and velocity for each voice over time
motion_history = np.zeros((SIM_STEPS, NUM_VOICES))
velocity_history = np.zeros((SIM_STEPS, NUM_VOICES))

for step in range(SIM_STEPS):
    for voice in range(NUM_VOICES):
        outputs = output_history[step, voice, :]
        
        # Motion from bits 0-5
        up_bits = (outputs[0:3] > 0.5).astype(int)
        down_bits = (outputs[3:6] > 0.5).astype(int)
        up_sum = np.dot(up_bits, up_weights)
        down_sum = np.dot(down_bits, down_weights)
        motion_history[step, voice] = up_sum - down_sum
        
        # Velocity from bits 6-7
        velocity_history[step, voice] = outputs[6] * outputs[7]

# %%
# Plot output activations for each voice
fig, axes = plt.subplots(NUM_VOICES, 1, figsize=(14, 3 * NUM_VOICES), sharex=True)

for v in range(NUM_VOICES):
    ax = axes[v] if NUM_VOICES > 1 else axes
    
    im = ax.imshow(
        output_history[:, v, :].T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=1,
        extent=[0, SIM_STEPS, -0.5, N_OUTPUTS_PER_VOICE - 0.5],
    )
    
    ax.set_ylabel(f"Voice {v + 1}")
    ax.set_yticks(range(N_OUTPUTS_PER_VOICE))
    ax.set_yticklabels(OUTPUT_NAMES)
    plt.colorbar(im, ax=ax, label="Activation", shrink=0.8)

axes[-1].set_xlabel("Time Step")
fig.suptitle("Motion Encoding Outputs by Voice", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Plot motion and velocity over time
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Motion over time
ax = axes[0]
for v in range(NUM_VOICES):
    ax.plot(motion_history[:, v], label=f"Voice {v + 1}", alpha=0.8)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Motion (semitones)")
ax.set_title("Voice Motion over Time")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-15, 15)

# Cumulative pitch (starting from 0)
ax = axes[1]
cumulative_pitch = np.cumsum(motion_history, axis=0)
for v in range(NUM_VOICES):
    ax.plot(cumulative_pitch[:, v], label=f"Voice {v + 1}", alpha=0.8)
ax.set_ylabel("Cumulative Pitch (semitones from start)")
ax.set_title("Voice Pitch Trajectory")
ax.legend()
ax.grid(True, alpha=0.3)

# Velocity over time
ax = axes[2]
for v in range(NUM_VOICES):
    ax.plot(velocity_history[:, v], label=f"Voice {v + 1}", alpha=0.8)
ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Threshold (0.3)")
ax.set_ylabel("Velocity (v1 × v2)")
ax.set_xlabel("Time Step")
ax.set_title("Voice Velocity over Time")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %%
# Motion distribution analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Motion distribution histogram
ax = axes[0]
all_motions = motion_history.flatten()
motion_values, motion_counts = np.unique(all_motions, return_counts=True)
ax.bar(motion_values, motion_counts, alpha=0.8)
ax.set_xlabel("Motion (semitones)")
ax.set_ylabel("Count")
ax.set_title("Motion Distribution (all voices)")
ax.grid(True, alpha=0.3, axis="y")

# Velocity distribution
ax = axes[1]
all_velocities = velocity_history.flatten()
ax.hist(all_velocities, bins=50, alpha=0.8, edgecolor="black")
ax.axvline(0.3, color="red", linestyle="--", linewidth=2, label="Threshold (0.3)")
ax.set_xlabel("Velocity (v1 × v2)")
ax.set_ylabel("Count")
ax.set_title("Velocity Distribution")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# %%
# Print statistics
print("\n=== Motion Encoding Statistics ===")
for v in range(NUM_VOICES):
    voice_motion = motion_history[:, v]
    voice_velocity = velocity_history[:, v]
    active_steps = np.sum(voice_velocity > 0.3)
    motion_events = np.sum(voice_motion != 0)
    mean_motion = np.mean(np.abs(voice_motion[voice_motion != 0])) if motion_events > 0 else 0
    
    print(f"Voice {v + 1}:")
    print(f"  Active steps (vel > 0.3): {active_steps} ({active_steps/SIM_STEPS*100:.1f}%)")
    print(f"  Motion events (≠0): {motion_events} ({motion_events/SIM_STEPS*100:.1f}%)")
    print(f"  Mean |motion| when moving: {mean_motion:.2f} semitones")
    print(f"  Pitch range: {cumulative_pitch[:, v].min():.0f} to {cumulative_pitch[:, v].max():.0f}")

# %%
# Export to MIDI using motion encoding
from datetime import datetime
from utils_sonic import save_motion_outputs_as_midi

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
midi_filename = f"midi/motion_output_{timestamp}.mid"

save_motion_outputs_as_midi(
    output_history,
    filename=midi_filename,
    tempo=60,
    velocity_threshold=0.3,
    start_pitches=[48, 60, 72, 84],  # C3, C4, C5, C6 (unison across octaves)
)

# %%
# Evaluate MIDI file
from eval_ambient import evaluate_ambient
from dataclasses import asdict

metrics_obj = evaluate_ambient(midi_filename)
print(f"\n=== Ambient Evaluation ===")
print(metrics_obj)
print(f"\nComposite Score: {metrics_obj.composite_score:.3f}")

# Plot metrics
metrics_dict = asdict(metrics_obj)
exclude_fields = ["best_scale", "best_root", "pitch_vocabulary", "composite_score"]
plot_metrics = {k: v for k, v in metrics_dict.items() if k not in exclude_fields}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(len(plot_metrics)), list(plot_metrics.values()))
axes[0].set_xticks(range(len(plot_metrics)))
axes[0].set_xticklabels(plot_metrics.keys(), rotation=45, ha="right")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Score")
axes[0].set_title("Individual Ambient Metrics")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(["Composite"], [metrics_obj.composite_score], color="green", width=0.5)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Score")
axes[1].set_title(f"Overall Score: {metrics_obj.composite_score:.3f}")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

