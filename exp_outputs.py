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
plt.savefig("output_readouts.png", dpi=150, bbox_inches="tight")
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
plt.savefig("output_summary.png", dpi=150, bbox_inches="tight")
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
