"""
(μ + λ) Evolution Strategy for neural network ambient music generation.

Evolves network genotypes to maximize ambient music fitness score.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from tqdm import tqdm

from network import NeuralNetwork, NetworkGenotype
from utils_sonic import save_readout_outputs_as_midi
from eval_ambient import evaluate_ambient


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout (for noisy function calls)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""

    # Evolution parameters
    mu: int = 20  # Number of parents
    lambda_: int = 100  # Number of offspring per generation
    generations: int = 50

    # Simulation parameters (from exp_outputs.py)
    sim_steps: int = 256
    tempo: int = 60
    percentile: int = 95  # For voice thresholds
    base_notes: list = field(default_factory=lambda: [48, 60, 72, 84])  # C3, C4, C5, C6

    # Mutation parameters
    weight_mutation_rate: float = 0.1
    weight_mutation_scale: float = 0.1
    threshold_mutation_rate: float = 0.1
    threshold_mutation_scale: float = 0.1
    refraction_mutation_rate: float = 0.05

    # Output
    output_dir: str = "evolve_midi"
    save_every_n_generations: int = 5  # Save best MIDI every N generations
    random_seed: Optional[int] = None


# =============================================================================
# Evaluation
# =============================================================================


@dataclass
class EvalResult:
    """Result from evaluating a single genotype."""

    fitness: float
    spectral_radius: float
    total_firing_events: int
    mean_activation: float
    midi_path: Optional[str] = None


def evaluate_genotype(
    genotype: NetworkGenotype,
    config: EvolutionConfig,
    save_midi: bool = False,
    midi_filename: Optional[str] = None,
) -> EvalResult:
    """
    Evaluate a genotype by running simulation and computing fitness.

    Follows the same process as exp_outputs.py.
    """
    # Create network from genotype
    net = genotype.to_network()

    # Get network properties
    spectral_radius = net.get_spectral_radius()

    # Run simulation
    num_readouts = genotype.num_readouts
    n_outputs_per_readout = genotype.n_outputs_per_readout
    output_history = np.zeros((config.sim_steps, num_readouts, n_outputs_per_readout))
    firing_history = np.zeros((config.sim_steps, genotype.num_neurons), dtype=bool)

    # Initial activation - activate the most connected neuron (suppress print)
    with suppress_stdout():
        net.manual_activate_most_weighted(1.0)

    # Run simulation
    for step in range(config.sim_steps):
        net.tick(step)
        output_history[step] = net.get_readout_outputs()
        firing_history[step] = net.state.firing

    total_firing_events = int(np.sum(firing_history))
    mean_activation = float(np.mean(output_history))

    # Check for degenerate cases (no activity)
    if total_firing_events == 0 or mean_activation < 1e-6:
        return EvalResult(
            fitness=0.0,
            spectral_radius=spectral_radius,
            total_firing_events=total_firing_events,
            mean_activation=mean_activation,
            midi_path=None,
        )

    # Compute voice thresholds (same as exp_outputs.py)
    voice_thresholds = np.zeros(num_readouts)
    for r in range(num_readouts):
        voice_data = output_history[:, r, :]
        voice_thresholds[r] = np.percentile(voice_data, config.percentile)

    # Save MIDI (suppress verbose output)
    if save_midi and midi_filename:
        os.makedirs(os.path.dirname(midi_filename), exist_ok=True)
        with suppress_stdout():
            save_readout_outputs_as_midi(
                output_history,
                filename=midi_filename,
                tempo=config.tempo,
                threshold=voice_thresholds,
                base_notes=config.base_notes,
            )

    # Create temporary MIDI for evaluation (suppress verbose output)
    temp_midi = f"/tmp/evolve_eval_{os.getpid()}_{id(genotype)}.mid"
    with suppress_stdout():
        save_readout_outputs_as_midi(
            output_history,
            filename=temp_midi,
            tempo=config.tempo,
            threshold=voice_thresholds,
            base_notes=config.base_notes,
        )

    # Evaluate fitness
    try:
        metrics = evaluate_ambient(temp_midi)
        fitness = metrics.composite_score
    except Exception as e:
        print(f"Evaluation error: {e}")
        fitness = 0.0
    finally:
        # Clean up temp file
        if os.path.exists(temp_midi):
            os.remove(temp_midi)

    return EvalResult(
        fitness=fitness,
        spectral_radius=spectral_radius,
        total_firing_events=total_firing_events,
        mean_activation=mean_activation,
        midi_path=midi_filename if save_midi else None,
    )


# =============================================================================
# Evolution
# =============================================================================


@dataclass
class GenerationStats:
    """Statistics for a single generation."""

    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_spectral_radius: float
    mean_spectral_radius: float
    best_firing_events: int
    mean_firing_events: float
    num_degenerate: int  # Networks with no activity


def run_evolution(
    config: EvolutionConfig,
) -> tuple[NetworkGenotype, list[GenerationStats]]:
    """
    Run (μ + λ) evolution strategy.

    Returns:
        best_genotype: The best genotype found
        history: List of GenerationStats for each generation
    """
    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize population
    print(f"Initializing population with {config.mu} individuals...")
    population = [NetworkGenotype.random() for _ in range(config.mu)]

    # Evaluate initial population
    print("Evaluating initial population...")
    results = []
    for geno in tqdm(population, desc="Initial eval", unit="ind"):
        results.append(evaluate_genotype(geno, config))

    fitnesses = [r.fitness for r in results]

    # Sort by fitness (descending)
    sorted_pairs = sorted(
        zip(population, results), key=lambda x: x[1].fitness, reverse=True
    )
    population = [g for g, r in sorted_pairs]
    results = [r for g, r in sorted_pairs]
    fitnesses = [r.fitness for r in results]

    # Track history
    history: list[GenerationStats] = []
    best_ever_fitness = max(fitnesses)
    best_ever_genotype = population[0]

    # Main evolution loop
    print(f"\nStarting evolution: {config.generations} generations")
    print(f"  μ = {config.mu} parents, λ = {config.lambda_} offspring")
    print(f"  Output directory: {config.output_dir}")
    print("-" * 70)

    for gen in range(config.generations):
        # Generate offspring via mutation
        offspring = []
        offspring_per_parent = config.lambda_ // config.mu

        for parent in population:
            for _ in range(offspring_per_parent):
                child = parent.mutate(
                    weight_mutation_rate=config.weight_mutation_rate,
                    weight_mutation_scale=config.weight_mutation_scale,
                    threshold_mutation_rate=config.threshold_mutation_rate,
                    threshold_mutation_scale=config.threshold_mutation_scale,
                    refraction_mutation_rate=config.refraction_mutation_rate,
                )
                offspring.append(child)

        # Evaluate offspring
        offspring_results = []
        for geno in tqdm(offspring, desc=f"Gen {gen+1:3d}", unit="ind", leave=False):
            offspring_results.append(evaluate_genotype(geno, config))

        # Combine parents + offspring
        combined_pop = population + offspring
        combined_results = results + offspring_results

        # Select best μ individuals
        sorted_pairs = sorted(
            zip(combined_pop, combined_results),
            key=lambda x: x[1].fitness,
            reverse=True,
        )
        population = [g for g, r in sorted_pairs[: config.mu]]
        results = [r for g, r in sorted_pairs[: config.mu]]
        fitnesses = [r.fitness for r in results]

        # Compute statistics
        all_fitnesses = [r.fitness for r in combined_results]
        all_spectral = [r.spectral_radius for r in combined_results]
        all_firing = [r.total_firing_events for r in combined_results]
        num_degenerate = sum(1 for r in combined_results if r.total_firing_events == 0)

        stats = GenerationStats(
            generation=gen + 1,
            best_fitness=fitnesses[0],
            mean_fitness=np.mean(fitnesses),
            std_fitness=np.std(fitnesses),
            best_spectral_radius=results[0].spectral_radius,
            mean_spectral_radius=np.mean(all_spectral),
            best_firing_events=results[0].total_firing_events,
            mean_firing_events=np.mean(all_firing),
            num_degenerate=num_degenerate,
        )
        history.append(stats)

        # Update best ever
        if fitnesses[0] > best_ever_fitness:
            best_ever_fitness = fitnesses[0]
            best_ever_genotype = population[0]

        # Progress output
        tqdm.write(
            f"Gen {gen+1:3d} | "
            f"Best: {stats.best_fitness:.4f} | "
            f"Mean: {stats.mean_fitness:.4f} ± {stats.std_fitness:.4f} | "
            f"ρ: {stats.best_spectral_radius:.2f} | "
            f"Fire: {stats.best_firing_events:4d} | "
            f"Dead: {stats.num_degenerate:3d}"
        )

        # Save best MIDI periodically
        if (
            gen + 1
        ) % config.save_every_n_generations == 0 or gen == config.generations - 1:
            midi_path = os.path.join(
                config.output_dir, f"gen{gen+1:03d}_best_{stats.best_fitness:.4f}.mid"
            )
            evaluate_genotype(
                population[0], config, save_midi=True, midi_filename=midi_path
            )
            tqdm.write(f"  → Saved: {midi_path}")

    print("-" * 70)
    print(f"Evolution complete! Best fitness: {best_ever_fitness:.4f}")

    # Save final best
    final_midi_path = os.path.join(
        config.output_dir, f"final_best_{best_ever_fitness:.4f}.mid"
    )
    evaluate_genotype(
        best_ever_genotype, config, save_midi=True, midi_filename=final_midi_path
    )
    print(f"Final best saved to: {final_midi_path}")

    return best_ever_genotype, history


# =============================================================================
# Visualization
# =============================================================================


def plot_evolution_history(
    history: list[GenerationStats], save_path: Optional[str] = None
):
    """Plot evolution progress and network property correlations."""

    generations = [s.generation for s in history]
    best_fitness = [s.best_fitness for s in history]
    mean_fitness = [s.mean_fitness for s in history]
    std_fitness = [s.std_fitness for s in history]
    spectral_radius = [s.best_spectral_radius for s in history]
    firing_events = [s.best_firing_events for s in history]
    num_degenerate = [s.num_degenerate for s in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Fitness over generations
    ax = axes[0, 0]
    ax.plot(generations, best_fitness, "b-", linewidth=2, label="Best")
    ax.fill_between(
        generations,
        np.array(mean_fitness) - np.array(std_fitness),
        np.array(mean_fitness) + np.array(std_fitness),
        alpha=0.3,
        color="blue",
        label="Mean ± std",
    )
    ax.plot(generations, mean_fitness, "b--", alpha=0.7, label="Mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Over Generations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: Spectral radius vs fitness
    ax = axes[0, 1]
    scatter = ax.scatter(
        spectral_radius, best_fitness, c=generations, cmap="viridis", alpha=0.7
    )
    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Spectral Radius vs Fitness")
    plt.colorbar(scatter, ax=ax, label="Generation")
    ax.grid(True, alpha=0.3)

    # Compute correlation
    corr = np.corrcoef(spectral_radius, best_fitness)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"r = {corr:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Bottom left: Firing events over generations
    ax = axes[1, 0]
    ax.plot(generations, firing_events, "g-", linewidth=2, label="Best individual")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Total Firing Events")
    ax.set_title("Network Activity Over Generations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add degenerate count on secondary axis
    ax2 = ax.twinx()
    ax2.plot(generations, num_degenerate, "r-", alpha=0.5, label="Degenerate")
    ax2.set_ylabel("Degenerate Networks", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Bottom right: Network properties correlation heatmap
    ax = axes[1, 1]
    data = np.array([best_fitness, spectral_radius, firing_events]).T
    labels = ["Fitness", "Spectral ρ", "Firing"]
    corr_matrix = np.corrcoef(data.T)

    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Property Correlations")

    # Add correlation values as text
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    plt.colorbar(im, ax=ax, label="Correlation")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Configuration
    config = EvolutionConfig(
        mu=20,
        lambda_=100,
        generations=50,
        random_seed=42,
        save_every_n_generations=5,
    )

    # Run evolution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = f"evolve_midi/{timestamp}"

    best_genotype, history = run_evolution(config)

    # Plot results
    plot_path = os.path.join(config.output_dir, "evolution_history.png")
    plot_evolution_history(history, save_path=plot_path)

    # Save best genotype
    import pickle

    genotype_path = os.path.join(config.output_dir, "best_genotype.pkl")
    with open(genotype_path, "wb") as f:
        pickle.dump(best_genotype, f)
    print(f"Best genotype saved to: {genotype_path}")
