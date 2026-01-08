# %%

from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class NeuralNetworkState:
    num_neurons: int = 16
    num_outputs: int = 16
    network_weights: np.ndarray = field(init=False)
    thresholds: np.ndarray = field(init=False)
    thresholds_current: np.ndarray = field(init=False)
    threshold_variation_ranges: np.ndarray = field(init=False)
    threshold_variation_periods: np.ndarray = field(init=False)
    output_weights: np.ndarray = field(init=False)
    activations: np.ndarray = field(init=False)
    firing: np.ndarray = field(init=False)
    outputs: np.ndarray = field(init=False)
    refractory_counters: np.ndarray = field(init=False)
    refraction_period: np.ndarray = field(init=False)

    use_activation_leak: bool = False
    activation_leak: float = 0.95
    refraction_leak: float = 0.4
    use_refraction_decay: bool = False
    use_tanh_activation: bool = False

    def __post_init__(self):
        self.network_weights = np.zeros((self.num_neurons, self.num_neurons))
        self.thresholds = np.full((self.num_neurons,), 0.5)
        self.thresholds_current = np.full((self.num_neurons,), 0.5)
        self.threshold_variation_ranges = np.full((self.num_neurons,), 0)
        self.threshold_variation_periods = np.full((self.num_neurons,), 0)
        # Output weights map from num_neurons to num_outputs
        self.output_weights = np.zeros((self.num_neurons, self.num_outputs))
        # Initialize with identity-like mapping for neurons that have corresponding outputs
        for i in range(min(self.num_neurons, self.num_outputs)):
            self.output_weights[i, i] = 1.0
        self.activations = np.zeros(self.num_neurons)
        self.firing = np.zeros(self.num_neurons, dtype=bool)
        self.outputs = np.zeros(self.num_outputs)
        self.refractory_counters = np.zeros(self.num_neurons, dtype=int)


class NeuralNetwork:
    def __init__(
        self,
        num_neurons: int = 64,
        num_outputs: Optional[int] = None,
        initial_state: Optional[NeuralNetworkState] = None,
    ):
        if initial_state is None:
            # Default num_outputs to num_neurons if not specified
            if num_outputs is None:
                num_outputs = num_neurons
            self.state = NeuralNetworkState(
                num_neurons=num_neurons, num_outputs=num_outputs
            )
        else:
            self.state = initial_state

    def tick(self, step: int):
        # Calculate new activations from current firing neurons
        new_activations = self.state.activations.copy()
        new_firing = np.zeros(self.state.num_neurons, dtype=bool)
        new_refractory_counters = self.state.refractory_counters.copy()

        for i in range(self.state.num_neurons):
            # sum up incoming activation from all firing neurons

            if not self.state.use_refraction_decay or new_refractory_counters[i] == 0:
                incoming_activation = 0
                for j in range(self.state.num_neurons):
                    if self.state.firing[j]:
                        # add weight from firing neuron j to current neuron i
                        incoming_activation += self.state.network_weights[j, i]

                # normalize incoming activation
                # incoming_activation /= self.state.num_neurons

                # clip or saturate activation
                if self.state.use_tanh_activation:
                    new_activations[i] = (
                        np.tanh(new_activations[i] + incoming_activation) + 1
                    ) / 2
                else:
                    new_activations[i] = np.clip(
                        new_activations[i] + incoming_activation, 0, 1
                    )

                # check if neuron should fire
                if new_activations[i] >= self.state.thresholds_current[i]:

                    # fire!
                    new_firing[i] = True

                    if self.state.use_refraction_decay:
                        new_refractory_counters[i] = self.state.refraction_period[i]
                    else:
                        new_activations[i] = 0

            # if self.state.use_refraction_decay:
            #     if new_refractory_counters[i] > 0:
            #         new_activations[i] *= self.state.refraction_leak

            if self.state.use_refraction_decay:
                if self.state.refractory_counters[i] > 0:  # Check OLD counter, not new!
                    new_activations[i] *= self.state.refraction_leak

            if self.state.threshold_variation_periods[i] > 0:
                self.state.thresholds_current[i] = (
                    np.sin(step * 2 * np.pi / self.state.threshold_variation_periods[i])
                    * self.state.threshold_variation_ranges[i]
                ) + self.state.thresholds[i]
                self.state.thresholds_current[i] = np.clip(
                    self.state.thresholds_current[i], 0, 1
                )

        # Calculate outputs based on firing neurons
        new_outputs = self.state.outputs.copy()
        new_outputs *= self.state.refraction_leak
        for i in range(self.state.num_neurons):
            if new_firing[i]:
                for j in range(self.state.num_outputs):
                    new_outputs[j] += self.state.output_weights[i, j]

        # Apply activation leak if enabled
        if self.state.use_activation_leak:
            new_activations *= self.state.activation_leak

        # Decrement refractory counters
        if self.state.use_refraction_decay:
            new_refractory_counters = np.maximum(0, new_refractory_counters - 1)

        # Update state
        self.state.activations = new_activations
        self.state.firing = new_firing
        self.state.outputs = np.clip(new_outputs, 0, 1)
        if self.state.use_refraction_decay:
            self.state.refractory_counters = new_refractory_counters

    def manual_trigger(self, neuron_index: int):
        if not 0 <= neuron_index < self.state.num_neurons:
            raise ValueError(
                f"Neuron index must be between 0 and {self.state.num_neurons - 1}"
            )

        # Set the neuron to fire
        # self.state.firing[neuron_index] = True

        # Set activation to 1
        self.state.activations[neuron_index] = 1

    def manual_activate(self, neuron_index: int, value: float):
        self.state.activations[neuron_index] += value
        self.state.activations[neuron_index] = np.clip(
            self.state.activations[neuron_index], 0, 1
        )

    def manual_activate_most_weighted(self, value: float):
        """Activate the neuron with the highest total network weight (impact on other neurons)."""
        # Calculate total network weight for each neuron (sum of all outgoing connections)
        total_network_weights = np.sum(self.state.network_weights, axis=1)

        # Find neuron with highest total network weight
        most_weighted_neuron = np.argmax(total_network_weights)

        print(
            f"Most weighted neuron: {most_weighted_neuron} (total weight: {total_network_weights[most_weighted_neuron]:.3f})"
        )

        # Activate that neuron
        self.manual_activate(most_weighted_neuron, value)

        return most_weighted_neuron

    def clear_firing(self):
        """Clear all firing states and outputs."""
        self.state.firing = np.zeros(self.state.num_neurons, dtype=bool)
        self.state.outputs = np.zeros(self.state.num_outputs)

    def enable_activation_leak(self, leak_factor: float = 0.95):
        self.state.use_activation_leak = True
        self.state.activation_leak = np.clip(leak_factor, 0, 1)

    def disable_activation_leak(self):
        self.state.use_activation_leak = False

    def enable_refraction_decay(
        self,
        refraction_period: int = 3,
        refraction_leak: float = 0.4,
        refraction_variation: int = 0,
    ):
        self.state.refraction_leak = np.clip(refraction_leak, 0, 1)
        self.state.use_refraction_decay = True

        # for each neuron, add a random variation to the refraction period
        self.state.refraction_period = np.full(
            self.state.num_neurons, refraction_period
        )
        if refraction_variation > 0:
            self.state.refraction_period += np.random.randint(
                0, refraction_variation, self.state.num_neurons
            )

    def disable_refraction_decay(self):
        self.state.use_refraction_decay = False

    def randomize_weights(self, sparsity=0.25, scale=0.4):
        """Randomize weights with gaussian distribution centered at 0, clipped to [-1, 1].

        Args:
            sparsity: Fraction of connections that exist (0-1)
            scale: Standard deviation of the gaussian distribution
        """
        # Gaussian distribution centered at 0, clipped to [-1, 1]
        self.state.network_weights = np.clip(
            np.random.randn(self.state.num_neurons, self.state.num_neurons) * scale,
            -1,
            1,
        )

        # Apply sparsity mask
        mask = (
            np.random.random((self.state.num_neurons, self.state.num_neurons))
            < sparsity
        )
        self.state.network_weights *= mask

    def get_spectral_radius(self) -> float:
        """Calculate the spectral radius (largest absolute eigenvalue) of the weight matrix."""
        eigenvalues = np.linalg.eigvals(self.state.network_weights)
        return np.max(np.abs(eigenvalues))

    def randomize_output_weights(self, sparsity=0.1, scale=0.3):
        self.state.output_weights = (
            np.random.random((self.state.num_neurons, self.state.num_outputs)) * scale
        )
        mask = (
            np.random.random((self.state.num_neurons, self.state.num_outputs))
            < sparsity
        )
        self.state.output_weights *= mask

    def randomize_thresholds(self):
        self.state.thresholds = np.random.random(self.state.num_neurons)
        self.state.thresholds_current = self.state.thresholds.copy()

    def randomize_threshold_variations(self, range=0.1, period=8):
        self.state.threshold_variation_ranges = (
            np.random.random(self.state.num_neurons) * range
        )
        if period > 0:
            self.state.threshold_variation_periods = np.random.randint(
                0, period, self.state.num_neurons
            )
        else:
            self.state.threshold_variation_periods = np.full(self.state.num_neurons, 0)

    def sinusoidal_weights(self):
        for i in range(self.state.num_neurons):
            for j in range(self.state.num_neurons):
                self.state.network_weights[i, j] = (
                    np.sin((i + j) * np.pi / self.state.num_neurons) * 0.5 + 0.5
                )

    def set_diagonal_weights(self, value: float):
        for i in range(self.state.num_neurons):
            self.state.network_weights[i, i] = value

    def clear(self):
        self.state = NeuralNetworkState(
            num_neurons=self.state.num_neurons, num_outputs=self.state.num_outputs
        )

    def set_output_identity(self):
        """Set output weights to identity-like mapping (neurons map 1:1 to outputs where possible)."""
        self.state.output_weights = np.zeros(
            (self.state.num_neurons, self.state.num_outputs)
        )
        for i in range(min(self.state.num_neurons, self.state.num_outputs)):
            self.state.output_weights[i, i] = 1.0

    def get_network_summary(self) -> dict:
        return {
            "activations": self.state.activations.tolist(),
            "firing": self.state.firing.tolist(),
            "outputs": self.state.outputs.tolist(),
        }

    def __str__(self) -> str:
        summary = self.get_network_summary()
        return f"NeuralNetwork(activations={summary['activations']}, firing={summary['firing']}, outputs={summary['outputs']})"

    def __repr__(self) -> str:
        return f"NeuralNetwork(state={self.state})"


# =======================================================================
def plot_weight_heatmap(num_neurons=64):
    data_matrix = np.array(network.state.network_weights)

    # Calculate symmetric range centered at 0
    max_abs = max(abs(data_matrix.min()), abs(data_matrix.max()))

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.heatmap(
        data_matrix.T,
        annot=False,
        cmap="RdBu",  # Diverging colormap: red=negative, blue=positive
        vmin=-max_abs,
        vmax=max_abs,
        center=0,
        ax=ax,
        cbar_kws={"label": "Weight Value"},
        yticklabels=[f"N{i}" for i in range(num_neurons)],
    )

    ax.set_title("Weight Heatmap (Red=Inhibitory, Blue=Excitatory)")

    plt.tight_layout()
    plt.show()


# =======================================================================
def plot_neural_heatmap(history, data_type="activations", num_neurons=64):
    steps_to_show = len(history.get("step", []))
    tick_step = steps_to_show // 16

    data_matrix = np.array(history[data_type][:steps_to_show])

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.heatmap(
        data_matrix.T,
        annot=False,
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": f"{data_type.capitalize()} Value"},
        yticklabels=[f"N{i}" for i in range(num_neurons)],
    )

    tick_positions = np.arange(0, steps_to_show, tick_step)
    tick_labels = history["step"][:steps_to_show:][::tick_step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Neuron")
    ax.set_title(f"{data_type.capitalize()} Over Time")

    plt.tight_layout()
    plt.show()


# =======================================================================

network = NeuralNetwork(num_neurons=256, num_outputs=8)
steps = 256
network.clear()
network.set_output_identity()

network.randomize_weights(sparsity=0.1, scale=0.4)
network.randomize_output_weights(sparsity=0.1, scale=0.2)
# network.sinusoidal_weights()
network.randomize_thresholds()
network.set_diagonal_weights(0)  # no self-feedback

# Print spectral radius to see network dynamics regime
print(f"Spectral radius: {network.get_spectral_radius():.3f}")

network.enable_activation_leak(0.9)
network.enable_refraction_decay(2, 0.75, 32)

network.randomize_threshold_variations(range=0.0, period=0)

# network.state.network_weights[0, 1] = 0.9  # N0 → N1
# network.state.network_weights[1, 2] = 0.9  # N1 → N2
# network.state.network_weights[2, 3] = 0.9  # N2 → N3
# network.state.network_weights[3, 0] = 0.9  # N3 → N0 (feedback loop)


# input patterns
# stimulators = [
#     [1, 0, 0, 0, 0, 0, 0, 0],
# ]
# stimulator_strength = 0.0

history = {"activations": [], "firing": [], "outputs": [], "thresholds": [], "step": []}

# run simulation
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


plot_weight_heatmap(network.state.num_neurons)
plot_neural_heatmap(history, "thresholds", network.state.num_neurons)
plot_neural_heatmap(history, "activations", network.state.num_neurons)
# plot_neural_heatmap(history, "firing", network.state.num_neurons)
plot_neural_heatmap(history, "outputs", network.state.num_outputs)

# =======================================================================
# Calculate synchronization from activation history (CTM-inspired)
# S^t = Z^t · (Z^t)^T where Z^t is the history of activations up to time t


def compute_synchronization_over_time(history, data_type="activations"):
    """
    Compute synchronization matrix at each time step.
    Returns a list of (D×D) synchronization matrices, one per time step.
    """
    activations = np.array(history[data_type])  # (T, D)
    T, D = activations.shape

    sync_matrices = []
    for t in range(1, T + 1):
        Z_t = activations[:t, :].T  # (D, t) - history up to time t
        S_t = Z_t @ Z_t.T  # (D, D) - synchronization matrix
        sync_matrices.append(S_t)

    return sync_matrices


def plot_synchronization_matrix(
    sync_matrix, title="Neural Synchronization", num_neurons=64
):
    """Plot a single synchronization matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize for visualization
    max_val = np.max(np.abs(sync_matrix))
    if max_val > 0:
        normalized = sync_matrix / max_val
    else:
        normalized = sync_matrix

    sns.heatmap(
        normalized,
        annot=False,
        cmap="magma",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Synchronization (normalized)"},
        xticklabels=(
            [f"N{i}" for i in range(num_neurons)] if num_neurons <= 16 else False
        ),
        yticklabels=(
            [f"N{i}" for i in range(num_neurons)] if num_neurons <= 16 else False
        ),
    )

    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_synchronization_evolution(sync_matrices, sample_steps=8, num_neurons=64):
    """Plot synchronization matrices at several time points to show evolution."""
    T = len(sync_matrices)
    step_indices = np.linspace(0, T - 1, sample_steps, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (ax, step) in enumerate(zip(axes, step_indices)):
        S = sync_matrices[step]
        max_val = np.max(np.abs(S))
        normalized = S / max_val if max_val > 0 else S

        im = ax.imshow(normalized, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"t={step + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Synchronization Matrix Evolution Over Time", fontsize=14)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Sync (normalized)")
    # plt.tight_layout()  # not compatible
    plt.show()


def plot_pairwise_sync_over_time(sync_matrices, neuron_pairs=None, num_neurons=64):
    """
    Plot how synchronization between specific neuron pairs evolves over time.
    If no pairs specified, picks the top synchronized pairs from final state.
    """
    T = len(sync_matrices)
    final_sync = sync_matrices[-1]

    if neuron_pairs is None:
        # Find top 5 most synchronized pairs (excluding diagonal)
        sync_copy = final_sync.copy()
        np.fill_diagonal(sync_copy, 0)
        flat_indices = np.argsort(sync_copy.flatten())[::-1][:5]
        neuron_pairs = [np.unravel_index(idx, final_sync.shape) for idx in flat_indices]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, j in neuron_pairs:
        if i >= j:  # Skip duplicates and self-pairs
            continue
        sync_over_time = [sync_matrices[t][i, j] for t in range(T)]
        ax.plot(sync_over_time, label=f"N{i} ↔ N{j}", alpha=0.8)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Synchronization (unnormalized)")
    ax.set_title("Pairwise Synchronization Over Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def cluster_neurons_by_sync(sync_matrix, n_clusters=4, method="ward"):
    """
    Cluster neurons based on their synchronization patterns using hierarchical clustering.

    Args:
        sync_matrix: (D, D) synchronization matrix
        n_clusters: number of clusters to form
        method: linkage method ('ward', 'complete', 'average', 'single')

    Returns:
        cluster_labels: array of cluster assignments for each neuron
        linkage_matrix: hierarchical clustering linkage matrix
    """
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform

    # Convert similarity to distance (higher sync = lower distance)
    # Normalize first
    max_val = np.max(sync_matrix)
    if max_val > 0:
        normalized = sync_matrix / max_val
    else:
        normalized = sync_matrix

    # Distance = 1 - similarity (works because normalized is in [0,1])
    distance_matrix = 1 - normalized
    np.fill_diagonal(distance_matrix, 0)  # Self-distance is 0

    # Make symmetric and convert to condensed form for scipy
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    condensed = squareform(distance_matrix)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed, method=method)
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    return cluster_labels, linkage_matrix


def plot_sync_dendrogram(linkage_matrix, cluster_labels, num_neurons=64):
    """Plot dendrogram showing hierarchical clustering of neurons."""
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(14, 6))

    # Color threshold to match n_clusters
    n_clusters = len(np.unique(cluster_labels))
    color_threshold = linkage_matrix[-(n_clusters - 1), 2] if n_clusters > 1 else 0

    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=[f"N{i}" for i in range(num_neurons)],
        leaf_rotation=90,
        leaf_font_size=8 if num_neurons <= 32 else 6,
        color_threshold=color_threshold,
    )

    ax.set_xlabel("Neuron")
    ax.set_ylabel("Distance (1 - sync)")
    ax.set_title("Hierarchical Clustering of Neurons by Synchronization")

    plt.tight_layout()
    plt.show()


def plot_clustered_sync_matrix(sync_matrix, cluster_labels, num_neurons=64):
    """Plot synchronization matrix reordered by cluster membership."""
    # Sort neurons by cluster
    sorted_indices = np.argsort(cluster_labels)
    reordered = sync_matrix[sorted_indices][:, sorted_indices]
    sorted_labels = cluster_labels[sorted_indices]

    # Normalize for visualization
    max_val = np.max(reordered)
    if max_val > 0:
        reordered = reordered / max_val

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(reordered, cmap="magma", vmin=0, vmax=1, aspect="auto")

    # Draw cluster boundaries
    unique_clusters = np.unique(sorted_labels)
    boundaries = []
    for c in unique_clusters[:-1]:
        boundary = np.where(sorted_labels == c)[0][-1] + 0.5
        boundaries.append(boundary)
        ax.axhline(y=boundary, color="white", linewidth=2, linestyle="--")
        ax.axvline(x=boundary, color="white", linewidth=2, linestyle="--")

    ax.set_xlabel("Neuron (sorted by cluster)")
    ax.set_ylabel("Neuron (sorted by cluster)")
    ax.set_title(
        f"Synchronization Matrix (neurons grouped into {len(unique_clusters)} clusters)"
    )

    plt.colorbar(im, ax=ax, label="Sync (normalized)")
    plt.tight_layout()
    plt.show()

    return sorted_indices


def get_cluster_members(cluster_labels, n_clusters=None):
    """Return a dict mapping cluster ID to list of neuron indices."""
    if n_clusters is None:
        n_clusters = len(np.unique(cluster_labels))

    clusters = {}
    for cluster_id in range(1, n_clusters + 1):
        members = np.where(cluster_labels == cluster_id)[0]
        clusters[cluster_id] = members.tolist()

    return clusters


def plot_cluster_activations(history, clusters, data_type="activations"):
    """
    Plot activations over time for each cluster.
    Shows individual neuron traces (faded) and cluster mean (bold).

    Args:
        history: dict with 'activations' key containing (T, D) data
        clusters: dict mapping cluster_id to list of neuron indices
    """
    activations = np.array(history[data_type])  # (T, D)
    T = activations.shape[0]
    time_steps = np.arange(T)

    n_clusters = len(clusters)

    # Use a nice color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))

    # Calculate grid dimensions
    cols = min(3, n_clusters)
    rows = (n_clusters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, (cluster_id, members) in enumerate(clusters.items()):
        ax = axes[idx]
        color = colors[idx % len(colors)]

        if len(members) == 0:
            ax.set_visible(False)
            continue

        # Get activations for this cluster's neurons
        cluster_activations = activations[:, members]  # (T, n_members)

        # Plot individual neuron traces (faded)
        for i, neuron_idx in enumerate(members):
            ax.plot(
                time_steps,
                cluster_activations[:, i],
                color=color,
                alpha=0.15,
                linewidth=0.8,
            )

        # Plot cluster mean (bold)
        cluster_mean = np.mean(cluster_activations, axis=1)
        ax.plot(
            time_steps,
            cluster_mean,
            color=color,
            linewidth=2.5,
            label=f"Mean (n={len(members)})",
        )

        # Plot ±1 std as shaded region
        cluster_std = np.std(cluster_activations, axis=1)
        ax.fill_between(
            time_steps,
            cluster_mean - cluster_std,
            cluster_mean + cluster_std,
            color=color,
            alpha=0.2,
        )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Activation")
        ax.set_title(f"Cluster {cluster_id} ({len(members)} neurons)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Cluster Activations Over Time", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def compute_cluster_means(history, clusters, data_type="activations"):
    """
    Compute mean activation for each cluster over time.

    Args:
        history: dict with 'activations' key containing (T, D) data
        clusters: dict mapping cluster_id to list of neuron indices
        data_type: which data to use from history (default: "activations")

    Returns:
        list of arrays, where each element is the cluster means at that timestep
        Format matches history["outputs"]: [(n_clusters,), (n_clusters,), ...]
    """
    activations = np.array(history[data_type])  # (T, D)
    T = activations.shape[0]

    # Get cluster IDs in sorted order for consistent indexing
    cluster_ids = sorted(clusters.keys())
    n_clusters = len(cluster_ids)

    # Build list of arrays, one per timestep
    cluster_means_list = []
    for t in range(T):
        means_at_t = np.zeros(n_clusters)
        for idx, cluster_id in enumerate(cluster_ids):
            members = clusters[cluster_id]
            if len(members) > 0:
                means_at_t[idx] = np.mean(activations[t, members])
            else:
                means_at_t[idx] = 0.0
        cluster_means_list.append(means_at_t)

    return cluster_means_list


def compute_windowed_coherence(
    history, clusters, window_size=16, data_type="activations"
):
    """
    Compute within-cluster correlation over time using sliding windows.

    Coherence measures how correlated the neurons within a cluster are at each moment.
    High coherence = neurons firing together. Low coherence = neurons out of sync.
    NOTE: This is NORMALIZED (correlation) - magnitude doesn't matter, only pattern.

    Returns:
        dict of {cluster_id: array of coherence values over time}
    """
    import warnings

    activations = np.array(history[data_type])  # (T, D)
    T = activations.shape[0]

    cluster_coherence = {cid: np.zeros(T) for cid in clusters}

    for t in range(T):
        start = max(0, t - window_size + 1)
        window = activations[start : t + 1, :]

        for cluster_id, members in clusters.items():
            if len(members) < 2:
                cluster_coherence[cluster_id][t] = 1.0
                continue

            cluster_window = window[:, members]

            if cluster_window.shape[0] > 1:
                # Check if there's enough variance
                variances = np.var(cluster_window, axis=0)
                if np.all(variances < 1e-10):
                    coherence = 0.0  # No variance = can't compute correlation
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        corr_matrix = np.corrcoef(cluster_window.T)
                        upper_tri = corr_matrix[np.triu_indices(len(members), k=1)]
                        valid = upper_tri[~np.isnan(upper_tri)]
                        coherence = np.mean(valid) if len(valid) > 0 else 0.0
            else:
                coherence = 1.0

            cluster_coherence[cluster_id][t] = coherence

    return cluster_coherence


def compute_windowed_sync(history, clusters, window_size=16, data_type="activations"):
    """
    Compute within-cluster sync (dot product) over time using sliding windows.

    This matches how clusters were built (cumulative sync = inner product of histories).
    Unlike correlation, this is NOT normalized - magnitude matters.
    High when neurons are active together. Low when neurons are inactive or out of sync.

    Returns:
        dict of {cluster_id: array of sync values over time}
    """
    activations = np.array(history[data_type])  # (T, D)
    T = activations.shape[0]

    cluster_sync = {cid: np.zeros(T) for cid in clusters}

    for t in range(T):
        start = max(0, t - window_size + 1)
        window = activations[start : t + 1, :]

        for cluster_id, members in clusters.items():
            if len(members) < 2:
                cluster_sync[cluster_id][t] = 0.0
                continue

            cluster_window = window[:, members]  # (window, n_members)

            # Compute sync matrix: Z @ Z.T (like the full sync matrix, but windowed)
            Z = cluster_window.T  # (n_members, window)
            S = Z @ Z.T  # (n_members, n_members)

            # Mean of off-diagonal elements (pairwise sync)
            n = len(members)
            off_diag_sum = np.sum(S) - np.trace(S)
            n_pairs = n * (n - 1)
            sync_value = off_diag_sum / n_pairs if n_pairs > 0 else 0.0

            cluster_sync[cluster_id][t] = sync_value

    return cluster_sync


def compute_windowed_inv_mse(
    history, clusters, window_size=16, data_type="activations"
):
    """
    Compute within-cluster inverse MSE over time using sliding windows.

    1 / (1 + MSE) - high when neurons have similar values at all times,
    INCLUDING when they're both at zero (unlike correlation).

    Returns:
        dict of {cluster_id: array of inv_mse values over time}
    """
    activations = np.array(history[data_type])  # (T, D)
    T = activations.shape[0]

    cluster_inv_mse = {cid: np.zeros(T) for cid in clusters}

    for t in range(T):
        start = max(0, t - window_size + 1)
        window = activations[start : t + 1, :]

        for cluster_id, members in clusters.items():
            if len(members) < 2:
                cluster_inv_mse[cluster_id][
                    t
                ] = 1.0  # Single neuron = perfect similarity
                continue

            cluster_window = window[:, members]  # (window, n_members)

            # Compute mean pairwise MSE between neurons
            n = len(members)
            total_mse = 0.0
            n_pairs = 0

            for i in range(n):
                for j in range(i + 1, n):
                    mse = np.mean((cluster_window[:, i] - cluster_window[:, j]) ** 2)
                    total_mse += mse
                    n_pairs += 1

            mean_mse = total_mse / n_pairs if n_pairs > 0 else 0.0
            inv_mse = 1.0 / (1.0 + mean_mse)

            cluster_inv_mse[cluster_id][t] = inv_mse

    return cluster_inv_mse


def compute_instantaneous_sync(history, clusters, data_type="activations"):
    """
    Compute instantaneous sync at each timestep (no window).

    Instantaneous sync = mean pairwise product of activations at each instant.
    High when multiple neurons are active at the same time.
    No window = duration comes from network dynamics, not hyperparameters.

    Returns:
        dict of {cluster_id: array of instant sync values over time}
    """
    activations = np.array(history[data_type])  # (T, D)
    T = activations.shape[0]

    cluster_sync = {cid: np.zeros(T) for cid in clusters}

    for t in range(T):
        for cluster_id, members in clusters.items():
            if len(members) < 2:
                cluster_sync[cluster_id][t] = 0.0
                continue

            acts = activations[t, members]  # (n_members,)

            # Mean of all pairwise products: (a_i * a_j) for i < j
            n = len(members)
            total = 0.0
            n_pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total += acts[i] * acts[j]
                    n_pairs += 1

            cluster_sync[cluster_id][t] = total / n_pairs if n_pairs > 0 else 0.0

    return cluster_sync


def compute_activation_derivative(history, clusters, data_type="activations"):
    """
    Compute derivative (rate of change) of mean activation for each cluster.

    Positive = rising (note attack)
    Negative = falling (note release)
    Near zero = stable

    Returns:
        dict of {cluster_id: array of derivative values over time}
    """
    activations = np.array(history[data_type])  # (T, D)
    T = activations.shape[0]

    cluster_deriv = {cid: np.zeros(T) for cid in clusters}

    for cluster_id, members in clusters.items():
        mean_act = np.mean(activations[:, members], axis=1)
        # Compute derivative (diff with padding to keep same length)
        deriv = np.zeros(T)
        deriv[1:] = np.diff(mean_act)
        cluster_deriv[cluster_id] = deriv

    return cluster_deriv


def plot_cluster_coherence(cluster_coherence, clusters, window_size=16):
    """
    Plot coherence over time for each cluster.
    Shows how synchronized neurons within each cluster are at each moment.
    """
    n_clusters = len(clusters)

    # Use same color palette as cluster activations
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))

    # Calculate grid dimensions
    cols = min(3, n_clusters)
    rows = (n_clusters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    axes = axes.flatten()

    cluster_ids = sorted(cluster_coherence.keys())

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        color = colors[idx % len(colors)]
        coherence = cluster_coherence[cluster_id]
        members = clusters[cluster_id]

        ax.plot(coherence, color=color, linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Coherence")
        ax.set_title(f"Cluster {cluster_id} ({len(members)} neurons)")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

        # Show stats
        mean_coh = np.mean(coherence)
        ax.text(
            0.02,
            0.98,
            f"μ={mean_coh:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
        )

    # Hide unused axes
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Within-Cluster Coherence Over Time (window={window_size})",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def compute_windowed_variance(
    history, clusters, window_size=16, data_type="activations"
):
    """
    Compute mean within-cluster variance over time.
    Low variance = neurons are all doing the same thing (stable but poor for correlation).
    """
    activations = np.array(history[data_type])
    T = activations.shape[0]

    cluster_variance = {cid: np.zeros(T) for cid in clusters}

    for t in range(T):
        start = max(0, t - window_size + 1)
        window = activations[start : t + 1, :]

        for cluster_id, members in clusters.items():
            if len(members) < 2:
                cluster_variance[cluster_id][t] = 0.0
                continue

            cluster_window = window[:, members]
            # Mean variance across neurons in the window
            var_per_neuron = np.var(cluster_window, axis=0)
            cluster_variance[cluster_id][t] = np.mean(var_per_neuron)

    return cluster_variance


def plot_coherence_comparison(
    cluster_coherence, history, clusters, data_type="activations", window_size=16
):
    """
    Compare mean activation, variance, windowed sync, correlation, and inverse MSE for each cluster.
    Shows relationship between different measures of cluster activity.
    """
    activations = np.array(history[data_type])
    n_clusters = len(clusters)

    # Compute additional metrics
    cluster_variance = compute_windowed_variance(
        history, clusters, window_size, data_type
    )
    cluster_sync = compute_windowed_sync(history, clusters, window_size, data_type)
    cluster_inv_mse = compute_windowed_inv_mse(
        history, clusters, window_size, data_type
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
    cluster_ids = sorted(cluster_coherence.keys())

    fig, axes = plt.subplots(
        n_clusters, 5, figsize=(22, 2.5 * n_clusters), squeeze=False
    )

    for idx, cluster_id in enumerate(cluster_ids):
        color = colors[idx % len(colors)]
        members = clusters[cluster_id]
        coherence = cluster_coherence[cluster_id]
        variance = cluster_variance[cluster_id]
        sync = cluster_sync[cluster_id]
        inv_mse = cluster_inv_mse[cluster_id]

        # Compute mean activation
        mean_activation = np.mean(activations[:, members], axis=1)

        # Col 0: Mean activation
        axes[idx, 0].plot(mean_activation, color=color, linewidth=1.5)
        axes[idx, 0].set_ylabel("Mean Act")
        axes[idx, 0].set_title(f"Cluster {cluster_id}: Mean Activation")
        axes[idx, 0].set_ylim(0, 1)
        axes[idx, 0].grid(True, alpha=0.3)

        # Col 1: Variance
        axes[idx, 1].plot(variance, color=color, linewidth=1.5)
        axes[idx, 1].set_ylabel("Variance")
        axes[idx, 1].set_title(f"Cluster {cluster_id}: Variance")
        axes[idx, 1].grid(True, alpha=0.3)

        # Col 2: Windowed Sync (dot product - matches how clusters were built)
        axes[idx, 2].plot(sync, color=color, linewidth=1.5)
        axes[idx, 2].set_ylabel("Sync")
        axes[idx, 2].set_title(f"Cluster {cluster_id}: Sync (dot product)")
        axes[idx, 2].grid(True, alpha=0.3)

        # Col 3: Coherence (correlation - normalized)
        axes[idx, 3].plot(coherence, color=color, linewidth=1.5)
        axes[idx, 3].set_ylabel("Corr")
        axes[idx, 3].set_title(f"Cluster {cluster_id}: Correlation")
        axes[idx, 3].set_ylim(-1.1, 1.1)
        axes[idx, 3].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[idx, 3].grid(True, alpha=0.3)

        # Col 4: Inverse MSE (high when similar, including both at zero)
        axes[idx, 4].plot(inv_mse, color=color, linewidth=1.5)
        axes[idx, 4].set_ylabel("Inv MSE")
        axes[idx, 4].set_title(f"Cluster {cluster_id}: 1/(1+MSE)")
        axes[idx, 4].set_ylim(0, 1.05)
        axes[idx, 4].grid(True, alpha=0.3)
        axes[idx, 4].text(
            0.02,
            0.98,
            f"μ={np.mean(inv_mse):.3f}",
            transform=axes[idx, 4].transAxes,
            fontsize=8,
            verticalalignment="top",
        )

    for i in range(5):
        axes[-1, i].set_xlabel("Time Step")

    fig.suptitle(
        "Mean Act | Variance | Sync (dot) | Correlation | Inv MSE (sameness)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_cluster_metrics_overlaid(
    cluster_coherence, history, clusters, data_type="activations", window_size=16
):
    """
    Plot all metrics overlaid on one graph per cluster for easier comparison.
    Shows: Mean Activation, Windowed Sync, Instant Sync, Correlation, Inv MSE, Derivative
    """
    activations = np.array(history[data_type])
    n_clusters = len(clusters)

    # Compute all metrics
    cluster_variance = compute_windowed_variance(
        history, clusters, window_size, data_type
    )
    cluster_sync_windowed = compute_windowed_sync(
        history, clusters, window_size, data_type
    )
    cluster_sync_instant = compute_instantaneous_sync(history, clusters, data_type)
    cluster_inv_mse = compute_windowed_inv_mse(
        history, clusters, window_size, data_type
    )
    cluster_deriv = compute_activation_derivative(history, clusters, data_type)

    cluster_ids = sorted(cluster_coherence.keys())

    # Calculate grid dimensions
    cols = min(2, n_clusters)
    rows = (n_clusters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        members = clusters[cluster_id]

        coherence = cluster_coherence[cluster_id]
        variance = cluster_variance[cluster_id]
        sync_win = cluster_sync_windowed[cluster_id]
        sync_inst = cluster_sync_instant[cluster_id]
        inv_mse = cluster_inv_mse[cluster_id]
        deriv = cluster_deriv[cluster_id]
        mean_activation = np.mean(activations[:, members], axis=1)

        T = len(mean_activation)
        t = np.arange(T)

        # Normalize windowed sync for comparison
        sync_win_max = np.max(sync_win) if np.max(sync_win) > 0 else 1
        sync_win_norm = sync_win / sync_win_max

        # Normalize instant sync for comparison
        sync_inst_max = np.max(sync_inst) if np.max(sync_inst) > 0 else 1
        sync_inst_norm = sync_inst / sync_inst_max

        # Normalize derivative to 0-1 range (center at 0.5)
        deriv_max = np.max(np.abs(deriv)) if np.max(np.abs(deriv)) > 0 else 1
        deriv_norm = (deriv / deriv_max) * 0.5 + 0.5  # Maps [-1,1] to [0,1]

        # Plot all metrics with distinct solid colors
        ax.plot(
            t, mean_activation, label="Mean Activation", linewidth=2.5, color="#1f77b4"
        )  # blue
        ax.plot(
            t,
            sync_win_norm,
            label=f"Windowed Sync (w={window_size})",
            linewidth=1.5,
            color="#ff7f0e",
        )  # orange
        ax.plot(
            t,
            sync_inst_norm,
            label="Instant Sync (no window)",
            linewidth=1.5,
            color="#2ca02c",
        )  # green
        ax.plot(
            t,
            (coherence + 1) / 2,
            label="Correlation (scaled)",
            linewidth=1.5,
            color="#d62728",
        )  # red
        ax.plot(t, inv_mse, label="Inv MSE", linewidth=1.5, color="#9467bd")  # purple
        ax.plot(
            t,
            deriv_norm,
            label="Derivative (centered)",
            linewidth=1.5,
            color="#8c564b",
            alpha=0.8,
        )  # brown

        # Add zero line for derivative reference
        ax.axhline(y=0.5, color="#8c564b", linestyle=":", alpha=0.3, linewidth=1)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value (normalized to 0-1)")
        ax.set_title(f"Cluster {cluster_id} ({len(members)} neurons) - All Metrics")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Cluster Metrics Overlaid | Instant Sync = no window lag", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.show()


# Compute synchronization from activation history
sync_matrices = compute_synchronization_over_time(history, "activations")

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
# each neuron gets a different frequency

import pygame
import numpy as np
import time


def find_threshold(history, percentile=0.8, key="outputs", per_channel=True):
    """
    Find threshold value(s) that keep the top percentile of the output history.

    Args:
        history: dict containing time series data
        percentile: percentile threshold (0-1)
        key: which data to use from history
        per_channel: if True, compute one threshold per channel; if False, compute global threshold

    Returns:
        array of thresholds (one per channel) if per_channel=True, else single float
    """
    data = np.array(history[key])  # (T, D) shape

    if per_channel:
        # Compute threshold for each channel independently
        thresholds = np.percentile(data, percentile * 100, axis=0)
        return thresholds
    else:
        # Global threshold across all data
        threshold = np.percentile(data, percentile * 100)
    return threshold


def play_neural_outputs_live(history, tempo=120, threshold=0.5, key="outputs"):
    """
    Play neural network data as audio in real-time.

    Args:
        history: dict containing time series data
        tempo: beats per minute
        threshold: single float or array of thresholds (one per channel)
        key: which data to use from history
    """
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    pygame.init()

    # Map each neuron to a different frequency
    base_freq = 220  # A3
    freq_ratio = 2 ** (1 / 12)  # Semitone ratio

    # try with phrygian dominant scale
    phrygian_dominant_scale = [0, 1, 4, 5, 7, 8, 10]
    phrygian_dominant_scale_freqs = [
        base_freq * (freq_ratio**i) for i in phrygian_dominant_scale
    ]

    # or major scale
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    major_scale_freqs = [base_freq * (freq_ratio**i) for i in major_scale]

    # Calculate time per step
    step_duration = 60.0 / tempo / 4  # Assuming 16th notes

    # Convert threshold to array if needed
    threshold_array = np.atleast_1d(threshold)

    for step, outputs in enumerate(history[key]):
        # Mix all active neurons for this time step
        mixed_audio = np.zeros(int(44100 * step_duration))

        for index, output_value in enumerate(outputs):
            # Get threshold for this channel (or use single threshold for all)
            thresh = (
                threshold_array[index]
                if index < len(threshold_array)
                else threshold_array[0]
            )

            if output_value > thresh:  # Threshold for activation
                # Map neuron index to frequency (each neuron gets a different note)
                # frequency = base_freq * (freq_ratio ** ((index * 5) % 36))
                frequency = major_scale_freqs[index % len(major_scale)]

                # Generate sine wave for this neuron
                t = np.linspace(0, step_duration, len(mixed_audio))
                wave = np.sin(2 * np.pi * frequency * t) * output_value

                # fade out over duration
                wave *= np.linspace(1, 0, len(t))

                # Add to mixed audio
                mixed_audio += wave

        # Normalize and convert to 16-bit audio
        if np.max(np.abs(mixed_audio)) > 0:
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.8
        audio = (mixed_audio * 32767).astype(np.int16)

        # Play the mixed audio for this time step
        sound = pygame.sndarray.make_sound(audio)
        sound.play()

        time.sleep(step_duration)


key = "clusters"
thresholds = find_threshold(history, percentile=0.80, key=key, per_channel=True)
print(f"Thresholds (per cluster): {thresholds}")
play_neural_outputs_live(history, tempo=60, threshold=thresholds, key=key)

# %%
# note number is neurons as 16 bit integer

import pygame
import numpy as np
import time

volume = 0.5


def play_neural_outputs_live(history, tempo=120, threshold=0.298):
    """Play neural network outputs as audio in real-time"""
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    pygame.init()

    # Map each neuron to a different frequency
    base_freq = 110  # A2
    freq_ratio = 2 ** (1 / 12)  # Semitone ratio

    # Calculate time per step
    step_duration = 60.0 / tempo / 4  # Assuming 16th notes

    for step, outputs in enumerate(history["outputs"]):
        wave = np.zeros(int(44100 * step_duration))

        # make binary digits from outputs
        binary_digits = list(map(lambda x: 1 if x > threshold else 0, outputs))

        # truncate binary digits to 2^7 bits
        binary_digits = binary_digits[: 2**7]
        if len(binary_digits) < 2**7:
            binary_digits.extend([0] * (2**7 - len(binary_digits)))

        # convert binary digits to integer
        note_number = int("".join(map(str, binary_digits)), 2)

        if note_number > 0:
            # map note number to frequency
            frequency = base_freq * (freq_ratio ** (note_number % 24))

            # generate sine wave
            t = np.linspace(0, step_duration, len(wave))
            wave = np.sin(2 * np.pi * frequency * t)

            # fade out over duration
            wave *= np.linspace(1, 0, len(t))

        audio = (wave * 32767 * volume).astype(np.int16)

        # play the wave for this time step
        sound = pygame.sndarray.make_sound(audio)
        sound.play()

        time.sleep(step_duration)


play_neural_outputs_live(history, tempo=90)

# %%

# import sys
# print(sys.executable)
# !{sys.executable} -m pip install pygame

# !pip install "scipy>=1.7.0"
