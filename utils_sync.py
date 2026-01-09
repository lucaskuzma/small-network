# %%
# Synchronization, clustering, and analysis utilities

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =======================================================================
# Plotting helpers
# =======================================================================


def plot_weight_heatmap(network):
    """Plot the network weight matrix as a heatmap."""
    num_neurons = network.state.num_neurons
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


def plot_neural_heatmap(history, data_type="activations", num_neurons=64):
    """Plot neural activity over time as a heatmap."""
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
# Synchronization computation
# =======================================================================


def compute_synchronization_over_time(history, data_type="activations", warmup=0):
    """
    Compute synchronization matrix at each time step.
    Returns a list of (D×D) synchronization matrices, one per time step.

    warmup: number of initial timesteps to ignore (ESN-style washout)
            Helps avoid initial transient spikes dominating the sync calculation.
    """
    activations = np.array(history[data_type])  # (T, D)
    T, D = activations.shape

    sync_matrices = []
    for t in range(1, T + 1):
        start = min(warmup, t - 1)  # Don't start beyond current time
        Z_t = activations[start:t, :].T  # (D, t-start) - history after warmup
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


# =======================================================================
# Clustering
# =======================================================================


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
    from scipy.cluster.hierarchy import linkage, fcluster
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


# =======================================================================
# Cluster analysis and visualization
# =======================================================================


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


# =======================================================================
# Coherence and sync metrics
# =======================================================================


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

