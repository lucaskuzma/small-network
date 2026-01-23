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

from enum import Enum
from network import NeuralNetwork, NetworkGenotype
from eval_ambient import evaluate_ambient
from eval_basic import evaluate_basic
from utils_sonic import save_piano_roll_png
from typing import Callable, Protocol


class AnnealStrategy(Enum):
    """Strategy for annealing mutation parameters based on parent age."""

    NONE = "none"  # No annealing (baseline)
    SCALE_UP = "scale_up"  # Increase mutation scale with age (explore more)
    SCALE_DOWN = "scale_down"  # Decrease mutation scale with age (fine-tune)
    RATE_UP = "rate_up"  # Increase mutation rate with age (mutate more genes)
    RATE_DOWN = "rate_down"  # Decrease mutation rate with age (preserve more)


@dataclass
class MutationParams:
    """Records the actual mutation parameters used to create an individual."""

    weight_rate: float = 0.0
    weight_scale: float = 0.0
    threshold_rate: float = 0.0
    threshold_scale: float = 0.0
    parent_age: int = 0
    strategy: AnnealStrategy = AnnealStrategy.NONE


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
    mutation_params: Optional[MutationParams] = (
        None  # Params used to create this individual
    )
    root_id: Optional[str] = (
        None  # ID of the original gen-0 ancestor (for lineage tracking)
    )

    def __post_init__(self):
        if not self.id:
            self.id = compute_genotype_hash(self.genotype)
        # If no root_id set and no parent, this is a root
        if self.root_id is None and self.parent_id is None:
            self.root_id = self.id

    def mutate_to_child(
        self,
        current_generation: int,
        strategy: AnnealStrategy = AnnealStrategy.NONE,
        **mutation_kwargs,
    ) -> "Individual":
        """Create a mutated child with tracking."""
        child_genotype = self.genotype.mutate(**mutation_kwargs)

        # Record the actual mutation params used
        params = MutationParams(
            weight_rate=mutation_kwargs.get("weight_mutation_rate", 0.0),
            weight_scale=mutation_kwargs.get("weight_mutation_scale", 0.0),
            threshold_rate=mutation_kwargs.get("threshold_mutation_rate", 0.0),
            threshold_scale=mutation_kwargs.get("threshold_mutation_scale", 0.0),
            parent_age=current_generation - self.generation_born,
            strategy=strategy,
        )

        return Individual(
            genotype=child_genotype,
            parent_id=self.id,
            generation_born=current_generation,
            mutation_params=params,
            root_id=self.root_id,  # Inherit lineage from parent
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
    parent_ages: list[int],
    offspring_fitnesses: list[float],
    random_fitnesses: list[float],
    offspring_modality: list[float],
    offspring_activity: list[float],
    offspring_diversity: list[float],
    random_modality: list[float],
    random_activity: list[float],
    random_diversity: list[float],
    gen: int,
    output_dir: str,
    best_fitness: float,
    encoding: str = "pitch",
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
    # Opacity by age (older = more opaque, capped at age 50)
    n_from_random = sum(parent_from_random)
    n_from_mutation = n_parents - n_from_random

    # Convert hex colors to RGBA with age-based alpha
    import matplotlib.colors as mcolors

    max_age = 50
    min_alpha, max_alpha = 0.3, 1.0

    parent_colors_rgba = []
    for rnd, age in zip(parent_from_random, parent_ages):
        base_color = "#9b59b6" if rnd else "#2ecc71"
        rgb = mcolors.to_rgb(base_color)
        # Alpha increases with age: min_alpha at age 0, max_alpha at age >= max_age
        alpha = min_alpha + (max_alpha - min_alpha) * min(age / max_age, 1.0)
        parent_colors_rgba.append((*rgb, alpha))

    if n_parents > 0:
        ax.scatter(
            parent_x,
            parent_fitnesses,
            c=parent_colors_rgba,
            s=120,
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

    # Median lines for offspring and randoms (fitness, modality, activity)
    # Colors: fitness=solid, modality=dashed, activity=dotted
    if n_offspring > 0:
        offspring_fit_med = np.median(offspring_fitnesses)
        offspring_mod_med = np.median(offspring_modality) if offspring_modality else 0
        offspring_act_med = np.median(offspring_activity) if offspring_activity else 0
        offspring_div_med = np.median(offspring_diversity) if offspring_diversity else 0
        # Fitness - solid blue
        ax.hlines(
            offspring_fit_med,
            0.35,
            0.6,
            colors="#3498db",
            linestyles="-",
            linewidth=2,
            alpha=0.9,
            zorder=3,
        )
        # Modality - dashed purple
        ax.hlines(
            offspring_mod_med,
            0.35,
            0.6,
            colors="#9b59b6",
            linestyles="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
        )
        # Activity - dotted green
        ax.hlines(
            offspring_act_med,
            0.35,
            0.6,
            colors="#27ae60",
            linestyles=":",
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
        )
        # Diversity - dash-dot orange
        ax.hlines(
            offspring_div_med,
            0.35,
            0.6,
            colors="#e67e22",
            linestyles="-.",
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
        )

    if n_randoms > 0:
        random_fit_med = np.median(random_fitnesses)
        random_mod_med = np.median(random_modality) if random_modality else 0
        random_act_med = np.median(random_activity) if random_activity else 0
        random_div_med = np.median(random_diversity) if random_diversity else 0
        # Fitness - solid red
        ax.hlines(
            random_fit_med,
            0.7,
            0.95,
            colors="#e74c3c",
            linestyles="-",
            linewidth=2,
            alpha=0.9,
            zorder=3,
        )
        # Modality - dashed purple
        ax.hlines(
            random_mod_med,
            0.7,
            0.95,
            colors="#9b59b6",
            linestyles="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
        )
        # Activity - dotted green
        ax.hlines(
            random_act_med,
            0.7,
            0.95,
            colors="#27ae60",
            linestyles=":",
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
        )
        # Diversity - dash-dot orange
        ax.hlines(
            random_div_med,
            0.7,
            0.95,
            colors="#e67e22",
            linestyles="-.",
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
        )

    # Legend entries for median lines
    ax.hlines(
        [],
        [],
        [],
        colors="#9b59b6",
        linestyles="--",
        linewidth=1.5,
        label="Modality median",
    )
    ax.hlines(
        [],
        [],
        [],
        colors="#27ae60",
        linestyles=":",
        linewidth=1.5,
        label="Activity median",
    )
    ax.hlines(
        [],
        [],
        [],
        colors="#e67e22",
        linestyles="-.",
        linewidth=1.5,
        label="Diversity median",
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
    ax.set_title(f"Generation {gen} | Best: {best_fitness:.4f} | Encoding: {encoding}")
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
    num_offspring: int = 100  # Number of mutated offspring per generation
    num_randoms: int = (
        0  # Number of fresh randoms per generation (mutations proven more effective)
    )
    generations: int = 50

    # Simulation parameters
    sim_steps: int = 128
    tempo: int = 60

    # Mutation parameters
    weight_mutation_rate: float = 0.1
    weight_mutation_scale: float = 0.001
    threshold_mutation_rate: float = 0.1
    threshold_mutation_scale: float = 0.001
    refraction_mutation_rate: float = 0.1

    # Annealing parameters: mutation params grow with parent age to escape stagnation
    # annealed_value = base_value * (1 + anneal_factor * parent_age)
    # Capped at anneal_max_multiplier * base_value
    anneal_factor: float = 0.1  # How much to increase per generation of age
    anneal_max_multiplier: float = 10.0  # Cap at 5x base value

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
    total_firing_events: int
    mean_activation: float
    activity_trend: float  # ratio of 2nd/1st half firing (used for culling)
    note_count: int = 0  # number of MIDI notes generated
    note_density: float = 0.0  # notes per beat
    modal_consistency: float = 0.0  # 0-1, how well notes fit a scale
    activity: float = 0.0  # 0-1, based on note density
    diversity: float = 0.0  # 0-1, pitch variety + anti-repetition
    midi_path: Optional[str] = None
    # Output statistics for debugging activity ceiling
    output_max: float = 0.0  # max output value across all timesteps
    output_min: float = 0.0  # min output value
    outputs_above_threshold: float = (
        0.0  # fraction of (timestep, voice) pairs above 0.3
    )


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

    # Output statistics for debugging activity ceiling
    output_max = float(np.max(output_history))
    output_min = float(np.min(output_history))
    # For argmax: check how many (timestep, voice) have max output > 0.3
    # Shape is (T, num_voices, n_pitches) - take max over pitches for each (t, voice)
    max_per_voice_per_step = np.max(output_history, axis=2)  # (T, num_voices)
    outputs_above_threshold = float(np.mean(max_per_voice_per_step > 0.3))

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
            total_firing_events=total_firing_events,
            mean_activation=mean_activation,
            activity_trend=activity_trend,
            midi_path=None,
            output_max=output_max,
            output_min=output_min,
            outputs_above_threshold=outputs_above_threshold,
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
    diversity = 0.0
    try:
        if config.evaluator == "basic":
            metrics = evaluate_basic(temp_midi, target_notes=config.sim_steps)
            fitness = metrics.composite_score
            note_count = metrics.note_count
            note_density = metrics.note_density
            modal_consistency = metrics.modal_consistency
            activity = metrics.activity
            diversity = metrics.diversity
        else:
            # Default to ambient
            metrics = evaluate_ambient(temp_midi)
            fitness = metrics.composite_score
            note_count = metrics.note_count
            note_density = metrics.note_density
            modal_consistency = metrics.modal_consistency
            activity = metrics.activity
            # ambient evaluator doesn't have diversity yet
    except Exception as e:
        print(f"Evaluation error: {e}")
        fitness = 0.0
    finally:
        # Clean up temp file
        if os.path.exists(temp_midi):
            os.remove(temp_midi)

    return EvalResult(
        fitness=fitness,
        total_firing_events=total_firing_events,
        mean_activation=mean_activation,
        activity_trend=activity_trend,
        note_count=note_count,
        note_density=note_density,
        modal_consistency=modal_consistency,
        activity=activity,
        diversity=diversity,
        midi_path=midi_filename if save_midi else None,
        output_max=output_max,
        output_min=output_min,
        outputs_above_threshold=outputs_above_threshold,
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
    # Best individual's evaluation metrics
    best_modal_consistency: float = 0.0
    best_activity: float = 0.0
    best_diversity: float = 0.0
    best_note_count: int = 0
    # Output statistics for debugging activity ceiling
    best_output_max: float = 0.0
    mean_output_max: float = 0.0
    mean_outputs_above_threshold: float = 0.0  # across all individuals
    # Root lineage counts (how many parents descend from each original ancestor)
    lineage_counts: dict = field(default_factory=dict)  # root_id -> count in top μ


@dataclass
class SuccessfulMutation:
    """Records a mutation that led to fitness improvement."""

    generation: int
    fitness_before: float
    fitness_after: float
    params: MutationParams


def compute_annealed_value(
    base_value: float,
    parent_age: int,
    anneal_factor: float,
    max_multiplier: float,
    increase: bool = True,
) -> float:
    """Compute annealed mutation parameter based on parent age.

    Args:
        base_value: Base mutation parameter value
        parent_age: How many generations since parent was born
        anneal_factor: How much to change per generation
        max_multiplier: Maximum/minimum multiplier to apply
        increase: If True, increase value with age; if False, decrease

    Returns:
        Annealed value, capped at max_multiplier (or min 1/max_multiplier if decreasing)
    """
    if increase:
        multiplier = min(1.0 + anneal_factor * parent_age, max_multiplier)
    else:
        # Decrease: multiplier goes from 1.0 down to 1/max_multiplier
        multiplier = max(1.0 / (1.0 + anneal_factor * parent_age), 1.0 / max_multiplier)
    return base_value * multiplier


def _plot_mutation_stats(
    successful_mutations: list[SuccessfulMutation],
    output_dir: str,
) -> None:
    """Plot summary statistics for successful mutations as histograms."""
    if not successful_mutations:
        print("\nNo successful mutations recorded.")
        return

    # Split by strategy
    by_strategy: dict[AnnealStrategy, list[SuccessfulMutation]] = {
        AnnealStrategy.NONE: [],
        AnnealStrategy.SCALE_UP: [],
        AnnealStrategy.SCALE_DOWN: [],
        AnnealStrategy.RATE_UP: [],
        AnnealStrategy.RATE_DOWN: [],
    }
    for m in successful_mutations:
        if m.params.strategy in by_strategy:
            by_strategy[m.params.strategy].append(m)

    # Colors for each strategy
    colors = {
        AnnealStrategy.NONE: "#7f8c8d",  # Gray - baseline
        AnnealStrategy.SCALE_UP: "#e74c3c",  # Red - aggressive
        AnnealStrategy.SCALE_DOWN: "#3498db",  # Blue - conservative
        AnnealStrategy.RATE_UP: "#e67e22",  # Orange - spread mutations
        AnnealStrategy.RATE_DOWN: "#27ae60",  # Green - focused mutations
    }

    strategy_labels = {
        AnnealStrategy.NONE: "None",
        AnnealStrategy.SCALE_UP: "Scale ↑",
        AnnealStrategy.SCALE_DOWN: "Scale ↓",
        AnnealStrategy.RATE_UP: "Rate ↑",
        AnnealStrategy.RATE_DOWN: "Rate ↓",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ===== Plot 1: Success count by strategy (bar chart) =====
    ax = axes[0, 0]
    strategies = list(by_strategy.keys())
    counts = [len(by_strategy[s]) for s in strategies]
    bars = ax.bar(
        [strategy_labels[s] for s in strategies],
        counts,
        color=[colors[s] for s in strategies],
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylabel("Number of Successful Mutations")
    ax.set_title("Success Count by Annealing Strategy")
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
    ax.set_ylim(0, max(counts) * 1.15 if counts and max(counts) > 0 else 1)

    # ===== Plot 2: Parent age distribution by strategy (overlapping histograms) =====
    ax = axes[0, 1]
    max_age = max((m.params.parent_age for m in successful_mutations), default=1)
    bins = np.linspace(0, max_age, min(20, max_age + 1))
    for strategy in strategies:
        ages = [m.params.parent_age for m in by_strategy[strategy]]
        if ages:
            ax.hist(
                ages,
                bins=bins,
                alpha=0.5,
                label=f"{strategy_labels[strategy]} (n={len(ages)})",
                color=colors[strategy],
                edgecolor=colors[strategy],
                linewidth=1.5,
            )
    ax.set_xlabel("Parent Age (generations)")
    ax.set_ylabel("Count")
    ax.set_title("Parent Age at Successful Mutation")
    ax.legend(loc="upper right")

    # ===== Plot 3: Fitness gain distribution by strategy =====
    ax = axes[1, 0]
    for strategy in strategies:
        gains = [m.fitness_after - m.fitness_before for m in by_strategy[strategy]]
        if gains:
            ax.hist(
                gains,
                bins=15,
                alpha=0.5,
                label=f"{strategy_labels[strategy]} (μ={np.mean(gains):.4f})",
                color=colors[strategy],
                edgecolor=colors[strategy],
                linewidth=1.5,
            )
    ax.set_xlabel("Fitness Gain")
    ax.set_ylabel("Count")
    ax.set_title("Fitness Gain Distribution by Strategy")
    ax.legend(loc="upper right")

    # ===== Plot 4: Success rate over time (cumulative by generation) =====
    ax = axes[1, 1]
    if successful_mutations:
        max_gen = max(m.generation for m in successful_mutations)
        gen_bins = np.linspace(0, max_gen, 10)

        for strategy in strategies:
            gens = [m.generation for m in by_strategy[strategy]]
            if gens:
                # Cumulative histogram
                ax.hist(
                    gens,
                    bins=gen_bins,
                    alpha=0.5,
                    label=strategy_labels[strategy],
                    color=colors[strategy],
                    edgecolor=colors[strategy],
                    linewidth=1.5,
                    cumulative=False,
                )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Successes in Period")
        ax.set_title("Success Timing by Strategy")
        ax.legend(loc="upper left")

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "mutation_strategy_stats.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMutation strategy stats saved to: {plot_path}")

    # Also print a brief summary
    print(f"\n{'='*60}")
    print(f"MUTATION STRATEGY SUMMARY ({len(successful_mutations)} total successes)")
    print("=" * 60)
    for strategy in strategies:
        muts = by_strategy[strategy]
        if muts:
            gains = [m.fitness_after - m.fitness_before for m in muts]
            ages = [m.params.parent_age for m in muts]
            print(
                f"  {strategy_labels[strategy]:12s}: {len(muts):3d} wins | "
                f"mean gain={np.mean(gains):.4f} | mean age={np.mean(ages):.1f}"
            )
        else:
            print(f"  {strategy_labels[strategy]:12s}:   0 wins")
    print("=" * 60)


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
            f"modal:{best_result.modal_consistency:.2f} act:{best_result.activity:.2f} div:{best_result.diversity:.2f} | "
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

    # Track successful mutation params for end-of-run analysis
    successful_mutations: list[SuccessfulMutation] = []

    # Get network params for generating randoms
    ref_geno = population[0].genotype

    for current_gen in range(start_gen + 1, total_generations + 1):
        prev_best_id = population[0].id if population else None
        prev_best_fitness = results[0].fitness if results else 0.0

        # Generate offspring via mutation (round-robin until we hit target)
        # Split into 5 fifths to test different annealing strategies:
        #   Q1: NONE       - baseline, no annealing (control group)
        #   Q2: SCALE_UP   - increase mutation magnitude with age (explore)
        #   Q3: SCALE_DOWN - decrease mutation magnitude with age (fine-tune)
        #   Q4: RATE_UP    - mutate more genes with age
        #   Q5: RATE_DOWN  - preserve more genes with age
        offspring = []
        parent_idx = 0
        fifth = config.num_offspring // 5

        while len(offspring) < config.num_offspring:
            parent = population[parent_idx % len(population)]
            parent_age = current_gen - parent.generation_born
            idx = len(offspring)

            # Determine annealing strategy based on offspring index (5 fifths)
            if idx < fifth:
                # Q1: NONE - baseline, no annealing
                strategy = AnnealStrategy.NONE
                weight_rate = config.weight_mutation_rate
                weight_scale = config.weight_mutation_scale
                threshold_rate = config.threshold_mutation_rate
                threshold_scale = config.threshold_mutation_scale
            elif idx < 2 * fifth:
                # Q2: SCALE_UP - increase mutation magnitude with age
                strategy = AnnealStrategy.SCALE_UP
                weight_scale = compute_annealed_value(
                    config.weight_mutation_scale,
                    parent_age,
                    config.anneal_factor,
                    config.anneal_max_multiplier,
                    increase=True,
                )
                threshold_scale = compute_annealed_value(
                    config.threshold_mutation_scale,
                    parent_age,
                    config.anneal_factor,
                    config.anneal_max_multiplier,
                    increase=True,
                )
                weight_rate = config.weight_mutation_rate
                threshold_rate = config.threshold_mutation_rate
            elif idx < 3 * fifth:
                # Q3: SCALE_DOWN - decrease mutation magnitude with age (fine-tune)
                strategy = AnnealStrategy.SCALE_DOWN
                weight_scale = compute_annealed_value(
                    config.weight_mutation_scale,
                    parent_age,
                    config.anneal_factor,
                    config.anneal_max_multiplier,
                    increase=False,
                )
                threshold_scale = compute_annealed_value(
                    config.threshold_mutation_scale,
                    parent_age,
                    config.anneal_factor,
                    config.anneal_max_multiplier,
                    increase=False,
                )
                weight_rate = config.weight_mutation_rate
                threshold_rate = config.threshold_mutation_rate
            elif idx < 4 * fifth:
                # Q4: RATE_UP - mutate more genes with age
                strategy = AnnealStrategy.RATE_UP
                weight_rate = compute_annealed_value(
                    config.weight_mutation_rate,
                    parent_age,
                    config.anneal_factor,
                    config.anneal_max_multiplier,
                    increase=True,
                )
                threshold_rate = compute_annealed_value(
                    config.threshold_mutation_rate,
                    parent_age,
                    config.anneal_factor,
                    config.anneal_max_multiplier,
                    increase=True,
                )
                # Cap rates at 1.0 (they're probabilities)
                weight_rate = min(weight_rate, 1.0)
                threshold_rate = min(threshold_rate, 1.0)
                weight_scale = config.weight_mutation_scale
                threshold_scale = config.threshold_mutation_scale
            else:
                # Q5: RATE_DOWN - preserve more genes with age
                strategy = AnnealStrategy.RATE_DOWN
                weight_rate = compute_annealed_value(
                    config.weight_mutation_rate,
                    parent_age,
                    config.anneal_factor,
                    config.anneal_max_multiplier,
                    increase=False,
                )
                threshold_rate = compute_annealed_value(
                    config.threshold_mutation_rate,
                    parent_age,
                    config.anneal_factor,
                    config.anneal_max_multiplier,
                    increase=False,
                )
                weight_scale = config.weight_mutation_scale
                threshold_scale = config.threshold_mutation_scale

            child = parent.mutate_to_child(
                current_generation=current_gen,
                strategy=strategy,
                weight_mutation_rate=weight_rate,
                weight_mutation_scale=weight_scale,
                threshold_mutation_rate=threshold_rate,
                threshold_mutation_scale=threshold_scale,
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

        # Capture metrics for visualization BEFORE combining
        parent_fitnesses = [r.fitness for r in results]
        # parent_id is None means it was a fresh random, not None means it was a mutation
        parent_from_random = [ind.parent_id is None for ind in population]
        parent_ages = [current_gen - ind.generation_born for ind in population]
        offspring_fitnesses_for_plot = [r.fitness for r in offspring_results]
        offspring_modality_for_plot = [r.modal_consistency for r in offspring_results]
        offspring_activity_for_plot = [r.activity for r in offspring_results]
        offspring_diversity_for_plot = [r.diversity for r in offspring_results]
        random_fitnesses_for_plot = [r.fitness for r in random_results]
        random_modality_for_plot = [r.modal_consistency for r in random_results]
        random_activity_for_plot = [r.activity for r in random_results]
        random_diversity_for_plot = [r.diversity for r in random_results]

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

        # Compute output statistics across all evaluated individuals
        all_output_max = [r.output_max for r in combined_results]
        all_above_thresh = [r.outputs_above_threshold for r in combined_results]

        # Compute lineage counts (how many of top μ descend from each original ancestor)
        lineage_counts: dict[str, int] = {}
        for ind in population:
            root = ind.root_id or ind.id  # Fallback for old individuals without root_id
            lineage_counts[root] = lineage_counts.get(root, 0) + 1

        stats = GenerationStats(
            generation=current_gen,
            best_fitness=fitnesses[0],
            mean_fitness=np.mean(fitnesses),
            std_fitness=np.std(fitnesses),
            best_firing_events=best_result.total_firing_events,
            mean_firing_events=np.mean(all_firing),
            num_degenerate=num_degenerate,
            best_id=best_ind.id,
            best_parent_id=best_ind.parent_id,
            best_age=current_gen - best_ind.generation_born,
            num_culled=num_culled,
            unique_lineages=unique_lineages,
            best_modal_consistency=best_result.modal_consistency,
            best_activity=best_result.activity,
            best_diversity=best_result.diversity,
            best_note_count=best_result.note_count,
            best_output_max=best_result.output_max,
            mean_output_max=np.mean(all_output_max),
            mean_outputs_above_threshold=np.mean(all_above_thresh),
            lineage_counts=lineage_counts,
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
                    # Record successful mutation params
                    if new_best.mutation_params is not None:
                        successful_mutations.append(
                            SuccessfulMutation(
                                generation=current_gen,
                                fitness_before=prev_best_fitness,
                                fitness_after=fitnesses[0],
                                params=new_best.mutation_params,
                            )
                        )

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
            f"modal:{best_result.modal_consistency:.2f} act:{best_result.activity:.2f} div:{best_result.diversity:.2f} | "
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
            parent_ages=parent_ages,
            offspring_fitnesses=offspring_fitnesses_for_plot,
            random_fitnesses=random_fitnesses_for_plot,
            offspring_modality=offspring_modality_for_plot,
            offspring_activity=offspring_activity_for_plot,
            offspring_diversity=offspring_diversity_for_plot,
            random_modality=random_modality_for_plot,
            random_activity=random_activity_for_plot,
            random_diversity=random_diversity_for_plot,
            gen=current_gen,
            output_dir=config.output_dir,
            best_fitness=best_ever_fitness,
            encoding=config.encoding,
        )

        # Save best MIDI and checkpoint periodically (only if fitness improved)
        is_last_gen = current_gen == total_generations
        current_best_fitness = best_ever_fitness
        if current_gen % config.save_every_n_generations == 0 or is_last_gen:
            # Only save MIDI if fitness actually improved since last save
            if current_best_fitness > last_saved_fitness:
                # Create subfolders for midi and piano roll graphs
                midi_dir = os.path.join(config.output_dir, "midi")
                graph_dir = os.path.join(config.output_dir, "midi_graph")
                os.makedirs(midi_dir, exist_ok=True)
                os.makedirs(graph_dir, exist_ok=True)

                filename = f"gen{current_gen:03d}_best_{current_best_fitness:.4f}"
                midi_path = os.path.join(midi_dir, f"{filename}.mid")
                graph_path = os.path.join(graph_dir, f"{filename}.png")

                evaluate_genotype(
                    best_ever_individual.genotype,
                    config,
                    save_midi=True,
                    midi_filename=midi_path,
                )
                # Save piano roll visualization
                save_piano_roll_png(
                    midi_path,
                    png_path=graph_path,
                    duration_beats=config.sim_steps / 4,  # 16th notes to beats
                    tempo=config.tempo,
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

    # Plot successful mutation statistics
    _plot_mutation_stats(successful_mutations, config.output_dir)

    # Save final best (only if improved since last periodic save)
    if best_ever_fitness > last_saved_fitness:
        # Create subfolders for midi and piano roll graphs
        midi_dir = os.path.join(config.output_dir, "midi")
        graph_dir = os.path.join(config.output_dir, "midi_graph")
        os.makedirs(midi_dir, exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)

        filename = f"final_best_{best_ever_fitness:.4f}"
        final_midi_path = os.path.join(midi_dir, f"{filename}.mid")
        final_graph_path = os.path.join(graph_dir, f"{filename}.png")

        evaluate_genotype(
            best_ever_individual.genotype,
            config,
            save_midi=True,
            midi_filename=final_midi_path,
        )
        # Save piano roll visualization
        save_piano_roll_png(
            final_midi_path,
            png_path=final_graph_path,
            duration_beats=config.sim_steps / 4,  # 16th notes to beats
            tempo=config.tempo,
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
    """Plot evolution progress: fitness charts on top, lineage chart on bottom."""

    generations = [s.generation for s in history]
    best_fitness = [s.best_fitness for s in history]
    mean_fitness = [s.mean_fitness for s in history]
    std_fitness = [s.std_fitness for s in history]
    modal_consistency = [s.best_modal_consistency for s in history]
    activity = [s.best_activity for s in history]
    diversity = [s.best_diversity for s in history]

    # Create figure with 2 rows: top row has 2 columns, bottom row spans full width
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # Top left: Fitness over generations
    ax = fig.add_subplot(gs[0, 0])
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

    # Top right: Modality, Activity, Diversity, and Fitness over generations
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(generations, best_fitness, "b-", linewidth=2, label="Fitness", alpha=0.9)
    ax.plot(
        generations,
        modal_consistency,
        "--",
        color="#9b59b6",
        linewidth=2,
        label="Modality",
        alpha=0.8,
    )
    ax.plot(
        generations,
        activity,
        ":",
        color="#27ae60",
        linewidth=2,
        label="Activity",
        alpha=0.8,
    )
    ax.plot(
        generations,
        diversity,
        "-.",
        color="#e67e22",
        linewidth=2,
        label="Diversity",
        alpha=0.8,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Fitness Components Over Time")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Bottom: Lineage survival (spans full width)
    ax = fig.add_subplot(gs[1, :])

    if history and history[0].lineage_counts:
        # Collect all root IDs that ever appear
        all_roots = set()
        for s in history:
            all_roots.update(s.lineage_counts.keys())
        all_roots = sorted(all_roots)

        # Build data matrix: (num_generations, num_roots)
        lineage_data = np.zeros((len(history), len(all_roots)))
        for i, s in enumerate(history):
            for j, root in enumerate(all_roots):
                lineage_data[i, j] = s.lineage_counts.get(root, 0)

        # Generate colors for each lineage
        cmap = plt.cm.get_cmap("tab20", len(all_roots))
        colors = [cmap(i) for i in range(len(all_roots))]

        # Stacked area chart
        ax.stackplot(
            generations,
            lineage_data.T,
            colors=colors,
            alpha=0.8,
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Parents per Lineage")
        ax.set_title("Lineage Survival Over Generations")
        ax.set_xlim(generations[0], generations[-1])
        ax.set_ylim(0, lineage_data.sum(axis=1).max())

        # Note how many lineages survive
        surviving_at_end = [
            root for root in all_roots if history[-1].lineage_counts.get(root, 0) > 0
        ]
        ax.text(
            0.98,
            0.98,
            f"{len(surviving_at_end)}/{len(all_roots)} lineages survive",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    else:
        ax.text(0.5, 0.5, "No lineage data", ha="center", va="center")
        ax.set_title("Lineage Survival")

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
    min_threshold: float = 0.3,
) -> Callable[..., str]:
    """
    Create a mapper for pitch-class encoding (original 12-output scheme).

    Each of 12 outputs per voice maps to a chromatic pitch class.
    Uses argmax: only the highest output per voice per timestep triggers a note,
    and only if it exceeds min_threshold.

    Args:
        base_notes: Base MIDI note per voice
        min_threshold: Minimum output value for argmax winner to trigger note
    """
    from utils_sonic import save_argmax_outputs_as_midi

    def mapper(output_history: np.ndarray, filename: str, tempo: int) -> str:
        save_argmax_outputs_as_midi(
            output_history,
            filename=filename,
            tempo=tempo,
            min_threshold=min_threshold,
            base_notes=base_notes,
        )
        return filename

    return mapper


def create_motion_mapper(
    start_pitches: list[int] = [48, 60, 72, 84],
    velocity_threshold: float = 0.3,
    velocity_range: tuple[int, int] = (70, 100),
) -> Callable[..., str]:
    """
    Create a mapper for motion encoding (8-output scheme).

    Outputs per voice: [u1, u4, u7, d1, d3, d8, v1, v2]
    Motion = (u1 + u4×4 + u7×7) - (d1 + d3×3 + d8×8)
    Velocity = |v1 - v2| (soft XOR)

    Args:
        start_pitches: Starting MIDI note per voice
        velocity_threshold: Fixed threshold for |v1-v2| to trigger notes (0-1)
        velocity_range: (min, max) MIDI velocity range for dynamics

    Voices start at unison (C) in their respective octaves.
    Pitch wraps modulo 12 within each octave.
    No note until first motion; sustain when velocity high but motion = 0.
    """
    from utils_sonic import save_motion_outputs_as_midi

    def mapper(output_history: np.ndarray, filename: str, tempo: int) -> str:
        save_motion_outputs_as_midi(
            output_history,
            filename=filename,
            tempo=tempo,
            velocity_threshold=velocity_threshold,  # Fixed threshold for all voices
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
        return 7  # 6 motion bits + 1 velocity gate
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
            num_offspring=100,  # All offspring from mutation (4 strategies, 25 each)
            num_randoms=0,  # Mutations proven more effective than randoms
            generations=args.generations,
            random_seed=42,
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
