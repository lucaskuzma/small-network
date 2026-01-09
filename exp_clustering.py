# %%
# Clustering experiment scratchpad

import numpy as np
import datetime
import pickle

from network import NeuralNetwork
from utils_sync import (
    plot_weight_heatmap,
    plot_neural_heatmap,
    compute_synchronization_over_time,
    plot_synchronization_matrix,
    plot_synchronization_evolution,
    plot_pairwise_sync_over_time,
    cluster_neurons_by_sync,
    plot_sync_dendrogram,
    plot_clustered_sync_matrix,
    get_cluster_members,
    plot_cluster_activations,
    compute_cluster_means,
    compute_windowed_coherence,
    plot_coherence_comparison,
    plot_cluster_metrics_overlaid,
)
from utils_sonic import (
    find_threshold,
    play_neural_outputs_live,
    save_neural_outputs_as_midi,
    play_midi,
)

# %%
# =======================================================================
# Network setup and simulation
# =======================================================================

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

network = NeuralNetwork(num_neurons=256, num_outputs=8)
steps = 256
warmup = steps // 8  # 32 steps - ignore initial transient for sync calculation
network.clear()
network.set_output_identity()

# Modular weight structure: 4 modules with high intra, low inter connectivity
# n_modules = 4
# network.randomize_modular_weights(
#     n_modules=n_modules,
#     intra_sparsity=0.25,  # 25% connections within modules
#     inter_sparsity=0.05,  # 2% connections between modules
#     scale=0.4,
# )
network.randomize_weights(sparsity=0.1, scale=0.4)  # original random weights
network.randomize_output_weights(sparsity=0.1, scale=0.2)
# network.sinusoidal_weights()
network.randomize_thresholds()
network.set_diagonal_weights(0)  # no self-feedback

# Print spectral radius to see network dynamics regime
print(f"Spectral radius: {network.get_spectral_radius():.3f}")

network.enable_activation_leak(0.9)
network.enable_refraction_decay(2, 0.75, 32)

network.randomize_threshold_variations(range=0.0, period=0)


history = {"activations": [], "firing": [], "outputs": [], "thresholds": [], "step": []}

# run simulation - activate most connected neuron in each module
# network.manual_activate_most_weighted_per_module(1.0)
network.manual_activate_most_weighted(1.0)

for step in range(steps):
    # for i, pattern in enumerate(stimulators):
    #     network.manual_activate(i, pattern[step % len(pattern)] * stimulator_strength)
    network.tick(step)
    history["activations"].append(network.state.activations.copy())
    history["firing"].append(network.state.firing.copy())
    history["outputs"].append(network.state.outputs.copy())
    history["thresholds"].append(network.state.thresholds_current.copy())
    history["step"].append(step)

# %%
# =======================================================================
# Visualization
# =======================================================================

plot_weight_heatmap(network)
plot_neural_heatmap(history, "thresholds", network.state.num_neurons)
plot_neural_heatmap(history, "activations", network.state.num_neurons)
# plot_neural_heatmap(history, "firing", network.state.num_neurons)
plot_neural_heatmap(history, "outputs", network.state.num_outputs)

# %%
# =======================================================================
# Synchronization analysis
# =======================================================================

# Compute synchronization from activation history (with warmup to ignore initial transient)
sync_matrices = compute_synchronization_over_time(history, "activations", warmup=warmup)
print(f"Sync computation using warmup={warmup} steps (ignoring initial transient)")

# Plot final synchronization matrix
plot_synchronization_matrix(
    sync_matrices[-1],
    title=f"Final Synchronization Matrix (t={len(sync_matrices)})",
    num_neurons=network.state.num_neurons,
)

# Plot evolution of synchronization over time
plot_synchronization_evolution(
    sync_matrices, sample_steps=8, num_neurons=network.state.num_neurons
)

# Plot pairwise synchronization for top synchronized neuron pairs
plot_pairwise_sync_over_time(sync_matrices, num_neurons=network.state.num_neurons)

# %%
# =======================================================================
# Clustering
# =======================================================================

# Cluster neurons by synchronization patterns
n_clusters = 8  # Adjust this to change number of clusters
cluster_labels, linkage_matrix = cluster_neurons_by_sync(
    sync_matrices[-1], n_clusters=n_clusters
)

# Show dendrogram
plot_sync_dendrogram(
    linkage_matrix, cluster_labels, num_neurons=network.state.num_neurons
)

# Show reordered sync matrix with cluster boundaries
sorted_indices = plot_clustered_sync_matrix(
    sync_matrices[-1], cluster_labels, num_neurons=network.state.num_neurons
)

# Print cluster membership
clusters = get_cluster_members(cluster_labels)
print("\n=== Neuron Clusters by Synchronization ===")
for cluster_id, members in clusters.items():
    print(f"Cluster {cluster_id}: {len(members)} neurons → {members}")

# Plot activations for each cluster
plot_cluster_activations(history, clusters, data_type="activations")

# %%
# =======================================================================
# Cluster metrics
# =======================================================================

# Compute and save cluster mean activations to history
history["clusters"] = compute_cluster_means(history, clusters, data_type="activations")
cluster_array = np.array(history["clusters"])  # (T, n_clusters)
print("\n=== Cluster Mean Activations Saved to history['clusters'] ===")
print(
    f"Format: list of {len(history['clusters'])} timesteps, each with {cluster_array.shape[1]} cluster means"
)
print(f"Shape when converted to array: {cluster_array.shape} (time, n_clusters)")
print(f"Value range: [{cluster_array.min():.3f}, {cluster_array.max():.3f}]")
print(f"Cluster indices: {sorted(clusters.keys())}")

# Compute within-cluster coherence
window_size = 16  # Adjust this to see different temporal scales
cluster_coherence = compute_windowed_coherence(
    history, clusters, window_size=window_size
)

# Compare all metrics side by side (5 columns)
plot_coherence_comparison(cluster_coherence, history, clusters, window_size=window_size)

# Overlay all metrics on one graph per cluster for easier comparison
plot_cluster_metrics_overlaid(
    cluster_coherence, history, clusters, window_size=window_size
)

# %%
# =======================================================================
# Sonification - live audio
# =======================================================================

key = "clusters"
thresholds = find_threshold(history, percentile=0.80, key=key, per_channel=True)
print(f"Thresholds (per cluster): {thresholds}")
play_neural_outputs_live(history, tempo=60, threshold=thresholds, key=key)

# %%
# =======================================================================
# Save weights to pickle
# =======================================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# save all weights and initial parameters to pickle
params = {
    "network": network,
    "network_weights": network.state.network_weights,
    "output_weights": network.state.output_weights,
    "thresholds": network.state.thresholds,
    "thresholds_current": network.state.thresholds_current,
    "threshold_variation_ranges": network.state.threshold_variation_ranges,
    "threshold_variation_periods": network.state.threshold_variation_periods,
    "history": history,
}
with open(f"weights_{timestamp}.pkl", "wb") as f:
    pickle.dump(params, f)

# load history from pickle
# with open("history.pkl", "rb") as f:
#     history = pickle.load(f)

# %%
# =======================================================================
# Save to MIDI
# =======================================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"neural_output_{timestamp}.mid"
save_neural_outputs_as_midi(
    history, filename=filename, tempo=60, threshold=thresholds, key=key
)

# %%
# =======================================================================
# Play MIDI
# =======================================================================

play_midi(filename)
