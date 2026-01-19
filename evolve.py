"""
(μ + λ) Evolution Strategy for neural network ambient music generation.

Evolves network genotypes to maximize ambient music fitness score.
"""

import os
import sys
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union
from tqdm import tqdm

from network import NeuralNetwork, NetworkGenotype
from eval_ambient import evaluate_ambient
from eval_basic import evaluate_basic
from typing import Callable, Protocol


class MidiMapper(Protocol):
    """Protocol for converting network outputs to MIDI."""

    def __call__(
        self,
        output_history: np.ndarray,  # (T, num_readouts, n_outputs_per_readout)
        filename: str,
        tempo: int,
    ) -> str:
        """Convert outputs to MIDI file, return path."""
        ...


# =============================================================================
# Lineage Tracking
# =============================================================================


def compute_genotype_hash(genotype: NetworkGenotype) -> str:
    """Compute a short hash from genotype weights for identification."""
    # Combine key arrays into bytes
    data = (
        genotype.network_weights.tobytes()
        + genotype.output_weights.tobytes()
        + genotype.thresholds.tobytes()
    )
    return hashlib.md5(data).hexdigest()[:6]


@dataclass
class Individual:
    """Wrapper for genotype with tracking."""

    genotype: NetworkGenotype
    id: str = field(default_factory=lambda: "")
    parent_id: Optional[str] = None  # None = fresh random, set = mutation of parent
    generation_born: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = compute_genotype_hash(self.genotype)

    def mutate_to_child(
        self, current_generation: int, **mutation_kwargs
    ) -> "Individual":
        """Create a mutated child with tracking."""
        child_genotype = self.genotype.mutate(**mutation_kwargs)
        return Individual(
            genotype=child_genotype,
            parent_id=self.id,
            generation_born=current_generation,
        )


@dataclass
class Checkpoint:
    """Checkpoint for resuming evolution."""

    generation: int
    population: list[Individual]
    results: list["EvalResult"]
    history: list["GenerationStats"]
    best_ever_fitness: float
    best_ever_individual: Individual
    config: "EvolutionConfig"

    def save(self, path: str, quiet: bool = True):
        """Save checkpoint to file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        if not quiet:
            print(f"Checkpoint saved: {path}")

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        """Load checkpoint from file."""
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"Checkpoint loaded: {path} (generation {checkpoint.generation})")
        return checkpoint


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


def _save_generation_plot(
    parent_fitnesses: list[float],
    parent_from_random: list[bool],
    offspring_fitnesses: list[float],
    random_fitnesses: list[float],
    gen: int,
    output_dir: str,
    best_fitness: float,
):
    """Save PNG scatter plot of generation fitness distribution."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    n_parents = len(parent_fitnesses)
    n_offspring = len(offspring_fitnesses)
    n_randoms = len(random_fitnesses)

    # Deterministic X positions: 3 groups
    parent_x = np.linspace(0.05, 0.25, n_parents) if n_parents > 0 else []
    offspring_x = np.linspace(0.35, 0.6, n_offspring) if n_offspring > 0 else []
    random_x = np.linspace(0.7, 0.95, n_randoms) if n_randoms > 0 else []

    # Parents - color by lineage (green = from mutation, purple = from random)
    n_from_random = sum(parent_from_random)
    n_from_mutation = n_parents - n_from_random
    parent_colors = ["#9b59b6" if rnd else "#2ecc71" for rnd in parent_from_random]

    if n_parents > 0:
        ax.scatter(
            parent_x,
            parent_fitnesses,
            c=parent_colors,
            s=120,
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
        )

    # Offspring (mutations) - blue
    if n_offspring > 0:
        ax.scatter(
            offspring_x,
            offspring_fitnesses,
            c="#3498db",
            s=40,
            alpha=0.6,
            label=f"Offspring ({n_offspring})",
            zorder=2,
        )

    # Randoms - red
    if n_randoms > 0:
        ax.scatter(
            random_x,
            random_fitnesses,
            c="#e74c3c",
            s=40,
            alpha=0.6,
            label=f"Randoms ({n_randoms})",
            zorder=2,
        )

    # Median lines for offspring and randoms
    if n_offspring > 0:
        offspring_median = np.median(offspring_fitnesses)
        ax.hlines(
            offspring_median,
            0.35,
            0.6,
            colors="#3498db",
            linestyles="--",
            linewidth=2,
            alpha=0.8,
            zorder=3,
        )
        ax.text(
            0.61,
            offspring_median,
            f"{offspring_median:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#3498db",
        )

    if n_randoms > 0:
        random_median = np.median(random_fitnesses)
        ax.hlines(
            random_median,
            0.7,
            0.95,
            colors="#e74c3c",
            linestyles="--",
            linewidth=2,
            alpha=0.8,
            zorder=3,
        )
        ax.text(
            0.96,
            random_median,
            f"{random_median:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#e74c3c",
        )

    # Dummy scatters for legend (parents)
    ax.scatter(
        [],
        [],
        c="#2ecc71",
        s=120,
        label=f"Parents←Mut ({n_from_mutation})",
        edgecolors="black",
    )
    ax.scatter(
        [],
        [],
        c="#9b59b6",
        s=120,
        label=f"Parents←Rnd ({n_from_random})",
        edgecolors="black",
    )

    # Formatting - fixed axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.15, 0.475, 0.825])
    ax.set_xticklabels(["Parents", "Offspring", "Randoms"])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Fitness Score")
    ax.set_title(f"Generation {gen} | Best: {best_fitness:.4f}")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=best_fitness, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(axis="y", alpha=0.3)

    # Save
    plot_dir = os.path.join(output_dir, "generation_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"gen_{gen:04d}.png")
    plt.savefig(plot_path, dpi=100)
    plt.close(fig)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""

    # Evolution parameters
    mu: int = 20  # Number of parents to keep
    num_offspring: int = 50  # Number of mutated offspring per generation
    num_randoms: int = 50  # Number of fresh randoms per generation
    generations: int = 50

    # Simulation parameters
    sim_steps: int = 128
    tempo: int = 60

    # Mutation parameters
    weight_mutation_rate: float = 0.1
    weight_mutation_scale: float = 0.001
    threshold_mutation_rate: float = 0.1
    threshold_mutation_scale: float = 0.001
    refraction_mutation_rate: float = 0

    # Output
    output_dir: str = "evolve_midi"
    save_every_n_generations: int = 5  # Save best MIDI every N generations
    random_seed: Optional[int] = None

    # Output encoding - determines how network outputs map to MIDI
    # "pitch" = 12 chromatic outputs, "motion" = 8 motion outputs
    encoding: str = "pitch"

    # Evaluator: "ambient" (full) or "basic" (just modal + activity)
    evaluator: str = "basic"

    # Transient (not pickled) - recreated from encoding on load
    _midi_mapper: Optional[Callable[..., str]] = field(default=None, repr=False)

    @property
    def midi_mapper(self) -> Callable[..., str]:
        """Get the MIDI mapper, creating it if needed."""
        if self._midi_mapper is None:
            self._midi_mapper = _create_mapper_for_encoding(self.encoding)
        return self._midi_mapper

    @midi_mapper.setter
    def midi_mapper(self, value):
        self._midi_mapper = value

    def __getstate__(self):
        """Exclude _midi_mapper from pickle (can't pickle closures)."""
        state = self.__dict__.copy()
        state["_midi_mapper"] = None
        return state


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
    activity_trend: float  # ratio of 2nd/1st half firing (used for culling)
    note_count: int = 0  # number of MIDI notes generated
    note_density: float = 0.0  # notes per beat
    modal_consistency: float = 0.0  # 0-1, how well notes fit a scale
    activity: float = 0.0  # 0-1, based on note density
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

    # Compute activity trend: ratio of 2nd half to 1st half firing
    midpoint = config.sim_steps // 2
    first_half_firing = np.sum(firing_history[:midpoint])
    second_half_firing = np.sum(firing_history[midpoint:])
    # Avoid division by zero; if first half is 0, use a small number
    activity_trend = second_half_firing / max(first_half_firing, 1)

    # Check for degenerate cases: no activity, fizzling, or exploding
    # These get fitness=0 and skip expensive MIDI evaluation
    is_degenerate = (
        total_firing_events == 0
        or mean_activation < 1e-6
        or activity_trend < 0.5  # Fizzling: activity drops to less than half
        or activity_trend > 2.0  # Exploding: activity more than doubles
    )
    if is_degenerate:
        return EvalResult(
            fitness=0.0,
            spectral_radius=spectral_radius,
            total_firing_events=total_firing_events,
            mean_activation=mean_activation,
            activity_trend=activity_trend,
            midi_path=None,
        )

    # Convert outputs to MIDI using mapper (auto-created from config.encoding)
    # Save MIDI (suppress verbose output)
    if save_midi and midi_filename:
        os.makedirs(os.path.dirname(midi_filename), exist_ok=True)
        with suppress_stdout():
            config.midi_mapper(output_history, midi_filename, config.tempo)

    # Create temporary MIDI for evaluation (suppress verbose output)
    temp_midi = f"/tmp/evolve_eval_{os.getpid()}_{id(genotype)}.mid"
    with suppress_stdout():
        config.midi_mapper(output_history, temp_midi, config.tempo)

    # Evaluate fitness using chosen evaluator
    note_count = 0
    note_density = 0.0
    modal_consistency = 0.0
    activity = 0.0
    try:
        if config.evaluator == "basic":
            metrics = evaluate_basic(temp_midi)
            fitness = metrics.composite_score
            note_count = metrics.note_count
            note_density = metrics.note_density
            modal_consistency = metrics.modal_consistency
            activity = metrics.activity
        else:
            # Default to ambient
            metrics = evaluate_ambient(temp_midi)
            fitness = metrics.composite_score
            note_count = metrics.note_count
            note_density = metrics.note_density
            modal_consistency = metrics.modal_consistency
            activity = metrics.activity
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
        activity_trend=activity_trend,
        note_count=note_count,
        note_density=note_density,
        modal_consistency=modal_consistency,
        activity=activity,
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
    # Lineage tracking
    best_id: str = ""
    best_parent_id: Optional[str] = None
    best_age: int = 0  # generations since birth
    # Culling stats (degenerate networks removed from pool)
    num_culled: int = 0  # networks with fitness=0 (fizzled, exploded, or dead)
    # Population diversity
    unique_lineages: int = 0  # unique parent IDs among surviving parents


def run_evolution(
    config: EvolutionConfig,
    resume_from: Optional[Union[str, Checkpoint]] = None,
    additional_generations: Optional[int] = None,
    initial_population: Optional[list[Individual]] = None,
) -> tuple[NetworkGenotype, list[GenerationStats]]:
    """
    Run (μ + λ) evolution strategy.

    Args:
        config: Evolution configuration
        resume_from: Path to checkpoint file or Checkpoint object to resume from
        additional_generations: If resuming, run this many more generations
                               (overrides config.generations)
        initial_population: Optional pre-created population (for custom genotype configs)

    Returns:
        best_genotype: The best genotype found
        history: List of GenerationStats for each generation
    """
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Resume from checkpoint or start fresh
    if resume_from:
        if isinstance(resume_from, str):
            checkpoint = Checkpoint.load(resume_from)
        else:
            checkpoint = resume_from
        population = checkpoint.population
        results = checkpoint.results
        history = checkpoint.history
        best_ever_fitness = checkpoint.best_ever_fitness
        best_ever_individual = checkpoint.best_ever_individual
        start_gen = checkpoint.generation

        # Determine total generations
        if additional_generations:
            total_generations = start_gen + additional_generations
        else:
            total_generations = max(config.generations, start_gen + 10)

        print(f"Resuming from generation {start_gen}")
        print(f"  Will run until generation {total_generations}")
    else:
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Use provided population or create default
        if initial_population is not None:
            population = initial_population
            print(f"Using provided population with {len(population)} individuals")
        else:
            # Initialize population with lineage tracking (default: 12 outputs)
            print(f"Initializing population with {config.mu} individuals...")
            population = [
                Individual(genotype=NetworkGenotype.random(), generation_born=0)
                for _ in range(config.mu)
            ]

        # Evaluate initial population
        print("Evaluating initial population...")
        results = []
        for ind in tqdm(population, desc="Initial eval", unit="ind"):
            results.append(evaluate_genotype(ind.genotype, config))

        fitnesses = [r.fitness for r in results]

        # Sort by fitness (descending)
        sorted_pairs = sorted(
            zip(population, results), key=lambda x: x[1].fitness, reverse=True
        )
        population = [ind for ind, r in sorted_pairs]
        results = [r for ind, r in sorted_pairs]

        # Show initial population stats
        best_result = results[0]
        print(
            f"Initial best: {best_result.fitness:.4f} | "
            f"modal:{best_result.modal_consistency:.2f} act:{best_result.activity:.2f} | "
            f"notes:{best_result.note_count}"
        )

        # Track history
        history = []
        best_ever_fitness = max(r.fitness for r in results)
        best_ever_individual = population[0]
        start_gen = 0
        total_generations = config.generations

    # Main evolution loop
    print(f"\nRunning evolution: generations {start_gen + 1} to {total_generations}")
    print(
        f"  μ = {config.mu} parents, {config.num_offspring} offspring, {config.num_randoms} randoms"
    )
    print(f"  Output directory: {config.output_dir}")
    print("-" * 100)

    # Track improvement sources
    improvements_from_mutation = 0  # Improvements from mutation lineage
    improvements_from_random = 0  # Improvements from random lineage
    last_saved_fitness = 0.0  # Track to avoid duplicate saves

    # Get network params for generating randoms
    ref_geno = population[0].genotype

    for current_gen in range(start_gen + 1, total_generations + 1):
        prev_best_id = population[0].id if population else None
        prev_best_fitness = results[0].fitness if results else 0.0

        # Generate offspring via mutation (round-robin until we hit target)
        offspring = []
        parent_idx = 0
        while len(offspring) < config.num_offspring:
            parent = population[parent_idx % len(population)]
            child = parent.mutate_to_child(
                current_generation=current_gen,
                weight_mutation_rate=config.weight_mutation_rate,
                weight_mutation_scale=config.weight_mutation_scale,
                threshold_mutation_rate=config.threshold_mutation_rate,
                threshold_mutation_scale=config.threshold_mutation_scale,
                refraction_mutation_rate=config.refraction_mutation_rate,
            )
            offspring.append(child)
            parent_idx += 1

        # Generate fresh randoms (parent_id=None by default marks them as random)
        randoms = []
        for _ in range(config.num_randoms):
            fresh = Individual(
                genotype=NetworkGenotype.random(
                    num_neurons=ref_geno.num_neurons,
                    num_readouts=ref_geno.num_readouts,
                    n_outputs_per_readout=ref_geno.n_outputs_per_readout,
                ),
                generation_born=current_gen,
            )
            randoms.append(fresh)

        # Evaluate offspring and randoms separately
        offspring_results = []
        random_results = []

        # Evaluate offspring
        for ind in tqdm(
            offspring, desc=f"Gen {current_gen:3d} offspring", unit="ind", leave=False
        ):
            offspring_results.append(evaluate_genotype(ind.genotype, config))

        # Evaluate randoms
        for ind in tqdm(
            randoms, desc=f"Gen {current_gen:3d} randoms", unit="ind", leave=False
        ):
            random_results.append(evaluate_genotype(ind.genotype, config))

        # Track previous generation's parent IDs for survival visualization
        prev_parent_ids = {ind.id for ind in population}

        # Capture fitnesses for visualization BEFORE combining
        parent_fitnesses = [r.fitness for r in results]
        # parent_id is None means it was a fresh random, not None means it was a mutation
        parent_from_random = [ind.parent_id is None for ind in population]
        offspring_fitnesses_for_plot = [r.fitness for r in offspring_results]
        random_fitnesses_for_plot = [r.fitness for r in random_results]

        # Combine parents + offspring + randoms
        combined_pop = population + offspring + randoms
        combined_results = results + offspring_results + random_results

        # Select best μ individuals
        sorted_pairs = sorted(
            zip(combined_pop, combined_results),
            key=lambda x: x[1].fitness,
            reverse=True,
        )
        population = [ind for ind, r in sorted_pairs[: config.mu]]
        results = [r for ind, r in sorted_pairs[: config.mu]]
        fitnesses = [r.fitness for r in results]

        # Compute parent survival visualization: ● = old parent survived, ○ = new offspring
        survival_str = "".join(
            "●" if ind.id in prev_parent_ids else "○" for ind in population
        )
        num_parents_survived = sum(1 for ind in population if ind.id in prev_parent_ids)

        # Compute statistics
        all_spectral = [r.spectral_radius for r in combined_results]
        all_firing = [r.total_firing_events for r in combined_results]
        num_degenerate = sum(1 for r in combined_results if r.total_firing_events == 0)
        # Culled = degenerate networks (fitness=0 due to dead, fizzling, or exploding)
        num_culled = sum(1 for r in combined_results if r.fitness == 0.0)

        best_ind = population[0]
        best_result = results[0]

        # Count unique lineages in surviving population (by parent_id, or own id if root)
        lineage_ids = set()
        for ind in population:
            lineage_ids.add(ind.parent_id if ind.parent_id else ind.id)
        unique_lineages = len(lineage_ids)

        stats = GenerationStats(
            generation=current_gen,
            best_fitness=fitnesses[0],
            mean_fitness=np.mean(fitnesses),
            std_fitness=np.std(fitnesses),
            best_spectral_radius=best_result.spectral_radius,
            mean_spectral_radius=np.mean(all_spectral),
            best_firing_events=best_result.total_firing_events,
            mean_firing_events=np.mean(all_firing),
            num_degenerate=num_degenerate,
            best_id=best_ind.id,
            best_parent_id=best_ind.parent_id,
            best_age=current_gen - best_ind.generation_born,
            num_culled=num_culled,
            unique_lineages=unique_lineages,
        )
        history.append(stats)

        # Track if best improved and from what source
        best_improved = fitnesses[0] > prev_best_fitness
        improvement_source = ""
        if best_improved:
            new_best = population[0]
            if new_best.id != prev_best_id:
                # New individual became best - was it created by mutation or fresh random?
                # parent_id is None for fresh randoms, set for mutations
                if new_best.parent_id is None:
                    improvements_from_random += 1
                    improvement_source = "RND"
                else:
                    improvements_from_mutation += 1
                    improvement_source = "MUT"

        # Update best ever
        if fitnesses[0] > best_ever_fitness:
            best_ever_fitness = fitnesses[0]
            best_ever_individual = population[0]

        # Count how many parents were created as fresh randoms vs mutations
        n_parents_from_random = sum(1 for ind in population if ind.parent_id is None)
        n_parents_from_mutation = config.mu - n_parents_from_random

        # Progress output
        src_str = f" [{improvement_source}]" if improvement_source else ""
        tqdm.write(
            f"Gen {current_gen:3d} | "
            f"Best: {stats.best_fitness:.4f} | "
            f"modal:{best_result.modal_consistency:.2f} act:{best_result.activity:.2f} | "
            f"notes:{best_result.note_count:3d} | "
            f"[{survival_str}] | "
            f"age:{stats.best_age:2d} | "
            f"parents: mut={n_parents_from_mutation} rnd={n_parents_from_random} | "
            f"wins: mut={improvements_from_mutation} rnd={improvements_from_random}{src_str}"
        )

        # Save generation plot (for animation)
        _save_generation_plot(
            parent_fitnesses=parent_fitnesses,
            parent_from_random=parent_from_random,
            offspring_fitnesses=offspring_fitnesses_for_plot,
            random_fitnesses=random_fitnesses_for_plot,
            gen=current_gen,
            output_dir=config.output_dir,
            best_fitness=best_ever_fitness,
        )

        # Save best MIDI and checkpoint periodically (only if fitness improved)
        is_last_gen = current_gen == total_generations
        current_best_fitness = best_ever_fitness
        if current_gen % config.save_every_n_generations == 0 or is_last_gen:
            # Only save MIDI if fitness actually improved since last save
            if current_best_fitness > last_saved_fitness:
                midi_path = os.path.join(
                    config.output_dir,
                    f"gen{current_gen:03d}_best_{current_best_fitness:.4f}.mid",
                )
                evaluate_genotype(
                    best_ever_individual.genotype,
                    config,
                    save_midi=True,
                    midi_filename=midi_path,
                )
                tqdm.write(f"  → Saved: {midi_path}")
                last_saved_fitness = current_best_fitness

            # Save checkpoint
            checkpoint = Checkpoint(
                generation=current_gen,
                population=population,
                results=results,
                history=history,
                best_ever_fitness=best_ever_fitness,
                best_ever_individual=best_ever_individual,
                config=config,
            )
            checkpoint_path = os.path.join(config.output_dir, "checkpoint.pkl")
            checkpoint.save(checkpoint_path)

    print("-" * 100)
    print(
        f"Evolution complete! Best fitness: {best_ever_fitness:.4f} (ID: {best_ever_individual.id})"
    )

    # Save final best (only if improved since last periodic save)
    if best_ever_fitness > last_saved_fitness:
        final_midi_path = os.path.join(
            config.output_dir, f"final_best_{best_ever_fitness:.4f}.mid"
        )
        evaluate_genotype(
            best_ever_individual.genotype,
            config,
            save_midi=True,
            midi_filename=final_midi_path,
        )
        print(f"Final best saved to: {final_midi_path}")
    else:
        print(f"Final best unchanged since last save")

    return best_ever_individual.genotype, history


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

    # Bottom left: Diversity and culling over generations
    ax = axes[1, 0]
    unique_lineages = [s.unique_lineages for s in history]
    num_culled = [s.num_culled for s in history]

    ax.plot(generations, unique_lineages, "g-", linewidth=2, label="Unique lineages")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity (unique parent lineages)")
    ax.set_title("Population Diversity & Culling")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add culled count on secondary axis
    ax2 = ax.twinx()
    ax2.fill_between(
        generations, 0, num_culled, alpha=0.3, color="red", label="Culled (degenerate)"
    )
    ax2.set_ylabel("Culled networks", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right")

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
# Mapper Factories
# =============================================================================


def _create_mapper_for_encoding(encoding: str) -> Callable[..., str]:
    """Create the appropriate mapper for an encoding type."""
    if encoding == "pitch":
        return create_pitch_class_mapper()
    elif encoding == "motion":
        return create_motion_mapper()
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def create_pitch_class_mapper(
    base_notes: list[int] = [48, 60, 72, 84],
    percentile: int = 95,
) -> Callable[..., str]:
    """
    Create a mapper for pitch-class encoding (original 12-output scheme).

    Each of 12 outputs per voice maps to a chromatic pitch class.
    Notes triggered when output exceeds percentile-based threshold.
    """
    from utils_sonic import save_readout_outputs_as_midi

    def mapper(output_history: np.ndarray, filename: str, tempo: int) -> str:
        num_readouts = output_history.shape[1]
        # Compute voice thresholds (percentile-based, capped at 0.99)
        voice_thresholds = np.zeros(num_readouts)
        for r in range(num_readouts):
            voice_data = output_history[:, r, :]
            voice_thresholds[r] = min(np.percentile(voice_data, percentile), 0.99)

        save_readout_outputs_as_midi(
            output_history,
            filename=filename,
            tempo=tempo,
            threshold=voice_thresholds,
            base_notes=base_notes,
        )
        return filename

    return mapper


def create_motion_mapper(
    start_pitches: list[int] = [48, 60, 72, 84],
    velocity_percentile: int = 50,
    velocity_range: tuple[int, int] = (70, 100),
) -> Callable[..., str]:
    """
    Create a mapper for motion encoding (8-output scheme).

    Outputs per voice: [u1, u4, u7, d1, d3, d8, v1, v2]
    Motion = (u1 + u4×4 + u7×7) - (d1 + d3×3 + d8×8)
    Velocity = |v1 - v2| (soft XOR)

    Args:
        start_pitches: Starting MIDI note per voice
        velocity_percentile: Threshold as % of peak velocity (e.g., 50 = half of peak)
        velocity_range: (min, max) MIDI velocity range for dynamics

    Voices start at unison (C) in their respective octaves.
    Pitch wraps modulo 12 within each octave.
    No note until first motion; sustain when velocity high but motion = 0.
    """
    from utils_sonic import save_motion_outputs_as_midi

    def mapper(output_history: np.ndarray, filename: str, tempo: int) -> str:
        # Compute velocity values (v1 * v2) per voice
        # Outputs 6 and 7 are v1 and v2
        num_voices = output_history.shape[1]

        # Compute adaptive threshold PER VOICE using peak-relative method
        # threshold = X% of peak velocity (not percentile of occurrences)
        # This lets the network control note density, not the threshold
        velocity_thresholds = np.zeros(num_voices)
        for v in range(num_voices):
            vel_values = np.abs(output_history[:, v, 6] - output_history[:, v, 7])
            peak_vel = np.max(vel_values) if len(vel_values) > 0 else 1.0
            velocity_thresholds[v] = peak_vel * (velocity_percentile / 100.0)

        save_motion_outputs_as_midi(
            output_history,
            filename=filename,
            tempo=tempo,
            velocity_threshold=velocity_thresholds,  # now per-voice array
            start_pitches=start_pitches,
            velocity_range=velocity_range,
        )
        return filename

    return mapper


# =============================================================================
# Main
# =============================================================================


def resume_evolution(
    checkpoint_path: str,
    additional_generations: int = 10,
) -> tuple[NetworkGenotype, list[GenerationStats]]:
    """
    Resume evolution from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint.pkl file
        additional_generations: How many more generations to run

    Returns:
        best_genotype: The best genotype found
        history: Full history including resumed generations
    """
    checkpoint = Checkpoint.load(checkpoint_path)

    return run_evolution(
        checkpoint.config,
        resume_from=checkpoint,
        additional_generations=additional_generations,
    )


def get_n_outputs_for_encoding(encoding: str) -> int:
    """Get n_outputs_per_readout for encoding type."""
    if encoding == "pitch":
        return 12
    elif encoding == "motion":
        return 8
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evolve neural networks for ambient music"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint.pkl to resume from",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations (or additional generations if resuming)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["pitch", "motion"],
        default="pitch",
        help="Output encoding: 'pitch' (12 chromatic outputs) or 'motion' (8 motion outputs)",
    )
    parser.add_argument(
        "--eval",
        type=str,
        choices=["basic", "ambient"],
        default="basic",
        help="Evaluator: 'basic' (modal + activity only) or 'ambient' (full heuristics)",
    )
    args = parser.parse_args()

    if args.resume:
        # Resume from checkpoint
        checkpoint = Checkpoint.load(args.resume)
        config = checkpoint.config

        # Handle old checkpoints without encoding/evaluator fields
        if not hasattr(config, "encoding") or config.encoding is None:
            config.encoding = "pitch"
        if not hasattr(config, "evaluator") or config.evaluator is None:
            config.evaluator = "basic"

        best_genotype, history = run_evolution(
            config,
            resume_from=checkpoint,
            additional_generations=args.generations,
        )
    else:
        # Get outputs per voice for this encoding
        n_outputs = get_n_outputs_for_encoding(args.encoding)

        # Fresh run
        config = EvolutionConfig(
            mu=20,
            num_offspring=50,
            num_randoms=50,
            generations=args.generations,
            random_seed=45,
            save_every_n_generations=5,
            encoding=args.encoding,
            evaluator=args.eval,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = f"evolve_midi/{timestamp}_{args.encoding}"

        # Create initial population with correct output size
        print(f"Encoding: {args.encoding} ({n_outputs} outputs per voice)")
        print(f"Evaluator: {args.eval}")
        np.random.seed(config.random_seed)

        initial_population = [
            Individual(
                genotype=NetworkGenotype.random(n_outputs_per_readout=n_outputs),
                generation_born=0,
            )
            for _ in range(config.mu)
        ]

        best_genotype, history = run_evolution(
            config, initial_population=initial_population
        )

    # Plot results
    plot_path = os.path.join(config.output_dir, "evolution_history.png")
    plot_evolution_history(history, save_path=plot_path)

    # Save best genotype
    genotype_path = os.path.join(config.output_dir, "best_genotype.pkl")
    with open(genotype_path, "wb") as f:
        pickle.dump(best_genotype, f)
    print(f"Best genotype saved to: {genotype_path}")
