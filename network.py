# %%

from typing import Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NeuralNetworkState:
    num_neurons: int = 16
    num_readouts: int = 1  # Number of readout voices
    n_outputs_per_readout: int = 12  # Chromatic scale per readout
    num_outputs: int = field(init=False)  # Calculated from readouts
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

    # Weight threshold: weights with |w| < threshold are treated as 0
    weight_threshold: float = 0.0  # 0 = disabled

    def __post_init__(self):
        self.num_outputs = self.num_readouts * self.n_outputs_per_readout

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

    def get_readout_outputs(self) -> np.ndarray:
        """Return outputs reshaped as (num_readouts, n_outputs_per_readout)."""
        return self.outputs.reshape(self.num_readouts, self.n_outputs_per_readout)


@dataclass
class NetworkGenotype:
    """Simple direct encoding genotype for the spiking neural network.

    Contains only the evolvable parameters:
    - network_weights: neuron-to-neuron connection weights
    - output_weights: neuron-to-output connection weights
    - thresholds: firing threshold per neuron
    - refraction_period: refractory period per neuron

    Hyperparameters like activation_leak, refraction_leak are NOT part of the
    genotype - they define the simulation physics, not the individual.
    """

    num_neurons: int
    num_readouts: int
    n_outputs_per_readout: int
    network_weights: np.ndarray  # (num_neurons, num_neurons)
    output_weights: np.ndarray  # (num_neurons, num_outputs)
    thresholds: np.ndarray  # (num_neurons,)
    refraction_period: np.ndarray  # (num_neurons,) integers

    @property
    def num_outputs(self) -> int:
        return self.num_readouts * self.n_outputs_per_readout

    @classmethod
    def random(
        cls,
        num_neurons: int = 256,
        num_readouts: int = 4,
        n_outputs_per_readout: int = 12,
    ) -> "NetworkGenotype":
        """Create a random genotype using the same setup as exp_outputs.py."""
        # Create network and use existing methods (same as exp_outputs.py)
        net = NeuralNetwork(
            num_neurons=num_neurons,
            num_readouts=num_readouts,
            n_outputs_per_readout=n_outputs_per_readout,
        )
        net.randomize_weights(sparsity=0.025, scale=0.4)
        net.randomize_output_weights(sparsity=0.025, scale=0.2)
        net.randomize_thresholds()
        net.set_diagonal_weights(0)
        net.enable_refraction_decay(2, 0.75, 32)

        # Extract genotype from the configured network
        return cls.from_network(net)

    @classmethod
    def from_network(cls, net: "NeuralNetwork") -> "NetworkGenotype":
        """Extract genotype from an existing network."""
        return cls(
            num_neurons=net.state.num_neurons,
            num_readouts=net.state.num_readouts,
            n_outputs_per_readout=net.state.n_outputs_per_readout,
            network_weights=net.state.network_weights.copy(),
            output_weights=net.state.output_weights.copy(),
            thresholds=net.state.thresholds.copy(),
            refraction_period=net.state.refraction_period.copy(),
        )

    def to_network(
        self,
        activation_leak: float = 0.9,
        refraction_leak: float = 0.75,
        weight_threshold: float = 0.05,
    ) -> "NeuralNetwork":
        """Create a NeuralNetwork from this genotype.

        Args:
            activation_leak: Leak factor for activations (0-1). Default from exp_outputs.py.
            refraction_leak: Leak factor during refractory period (0-1). Default from exp_outputs.py.
            weight_threshold: Minimum |weight| to have effect (0 = disabled). Default 0.05.
        """
        net = NeuralNetwork(
            num_neurons=self.num_neurons,
            num_readouts=self.num_readouts,
            n_outputs_per_readout=self.n_outputs_per_readout,
        )
        # Set genotype parameters
        net.state.network_weights = self.network_weights.copy()
        net.state.output_weights = self.output_weights.copy()
        net.state.thresholds = self.thresholds.copy()
        net.state.thresholds_current = self.thresholds.copy()
        net.state.refraction_period = self.refraction_period.copy()

        # Configure hyperparameters (don't overwrite refraction_period from genotype)
        net.state.use_activation_leak = True
        net.state.activation_leak = activation_leak
        net.state.use_refraction_decay = True
        net.state.refraction_leak = refraction_leak
        net.state.weight_threshold = weight_threshold

        return net

    def mutate(
        self,
        weight_mutation_rate: float = 0.1,
        weight_mutation_scale: float = 0.1,
        threshold_mutation_rate: float = 0.1,
        threshold_mutation_scale: float = 0.1,
        refraction_mutation_rate: float = 0.05,
    ) -> "NetworkGenotype":
        """Return a mutated copy of this genotype.

        Args:
            weight_mutation_rate: Probability of mutating each weight
            weight_mutation_scale: Std dev of gaussian noise added to weights
            threshold_mutation_rate: Probability of mutating each threshold
            threshold_mutation_scale: Std dev of gaussian noise added to thresholds
            refraction_mutation_rate: Probability of mutating each refraction period
        """
        # Copy arrays
        new_network_weights = self.network_weights.copy()
        new_output_weights = self.output_weights.copy()
        new_thresholds = self.thresholds.copy()
        new_refraction = self.refraction_period.copy()

        # Mutate network weights (all weights, including zeros - topology can evolve)
        mask = np.random.random(new_network_weights.shape) < weight_mutation_rate
        new_network_weights += (
            mask * np.random.randn(*new_network_weights.shape) * weight_mutation_scale
        )
        new_network_weights = np.clip(new_network_weights, -1, 1)
        np.fill_diagonal(new_network_weights, 0)  # Preserve no self-connections

        # Mutate output weights (all weights, including zeros)
        mask = np.random.random(new_output_weights.shape) < weight_mutation_rate
        new_output_weights += (
            mask * np.random.randn(*new_output_weights.shape) * weight_mutation_scale
        )
        new_output_weights = np.clip(new_output_weights, -1, 1)

        # Mutate thresholds
        mask = np.random.random(new_thresholds.shape) < threshold_mutation_rate
        new_thresholds += (
            mask * np.random.randn(*new_thresholds.shape) * threshold_mutation_scale
        )
        new_thresholds = np.clip(new_thresholds, 0, 1)

        # Mutate refraction periods (add/subtract 1)
        mask = np.random.random(new_refraction.shape) < refraction_mutation_rate
        new_refraction = new_refraction + mask * np.random.choice(
            [-1, 1], size=new_refraction.shape
        )
        new_refraction = np.clip(new_refraction, 2, 33).astype(
            int
        )  # Match random() range

        return NetworkGenotype(
            num_neurons=self.num_neurons,
            num_readouts=self.num_readouts,
            n_outputs_per_readout=self.n_outputs_per_readout,
            network_weights=new_network_weights,
            output_weights=new_output_weights,
            thresholds=new_thresholds,
            refraction_period=new_refraction,
        )

    def crossover(self, other: "NetworkGenotype") -> "NetworkGenotype":
        """Uniform crossover: randomly pick each gene from either parent."""
        assert self.num_neurons == other.num_neurons
        assert self.num_outputs == other.num_outputs

        # For weights: element-wise random choice
        mask_net = np.random.random(self.network_weights.shape) < 0.5
        new_network_weights = np.where(
            mask_net, self.network_weights, other.network_weights
        )

        mask_out = np.random.random(self.output_weights.shape) < 0.5
        new_output_weights = np.where(
            mask_out, self.output_weights, other.output_weights
        )

        # For per-neuron params: element-wise random choice
        mask_thresh = np.random.random(self.thresholds.shape) < 0.5
        new_thresholds = np.where(mask_thresh, self.thresholds, other.thresholds)

        mask_refrac = np.random.random(self.refraction_period.shape) < 0.5
        new_refraction = np.where(
            mask_refrac, self.refraction_period, other.refraction_period
        )

        return NetworkGenotype(
            num_neurons=self.num_neurons,
            num_readouts=self.num_readouts,
            n_outputs_per_readout=self.n_outputs_per_readout,
            network_weights=new_network_weights,
            output_weights=new_output_weights,
            thresholds=new_thresholds,
            refraction_period=new_refraction,
        )


class NeuralNetwork:
    def __init__(
        self,
        num_neurons: int = 64,
        num_outputs: Optional[int] = None,
        num_readouts: int = 1,
        n_outputs_per_readout: int = 12,
        initial_state: Optional[NeuralNetworkState] = None,
    ):
        if initial_state is None:
            # Backward compatibility: if num_outputs is specified, use it directly
            if num_outputs is not None:
                num_readouts = 1
                n_outputs_per_readout = num_outputs

            self.state = NeuralNetworkState(
                num_neurons=num_neurons,
                num_readouts=num_readouts,
                n_outputs_per_readout=n_outputs_per_readout,
            )
        else:
            self.state = initial_state

    def get_readout_outputs(self) -> np.ndarray:
        """Return outputs reshaped as (num_readouts, n_outputs_per_readout)."""
        return self.state.get_readout_outputs()

    def _apply_weight_threshold(self, weights: np.ndarray) -> np.ndarray:
        """Apply weight threshold: weights with |w| < threshold contribute 0."""
        if self.state.weight_threshold <= 0:
            return weights
        # Hard threshold: zero out small weights
        return np.where(np.abs(weights) >= self.state.weight_threshold, weights, 0)

    def tick(self, step: int):
        """Vectorized network tick - update all neurons in parallel."""
        state = self.state

        # === 1. Compute incoming activation (matrix-vector multiply) ===
        # incoming[i] = sum of weights from all firing neurons j to neuron i
        # Apply weight threshold so small weights don't contribute
        effective_weights = self._apply_weight_threshold(state.network_weights)
        incoming = effective_weights.T @ state.firing.astype(np.float64)

        # === 2. Determine which neurons can receive input (not refractory) ===
        if state.use_refraction_decay:
            can_receive = state.refractory_counters == 0
        else:
            can_receive = np.ones(state.num_neurons, dtype=bool)

        # === 3. Update activations for non-refractory neurons ===
        new_activations = state.activations.copy()
        if state.use_tanh_activation:
            updated = (np.tanh(state.activations + incoming) + 1) / 2
        else:
            updated = np.clip(state.activations + incoming, 0, 1)
        new_activations = np.where(can_receive, updated, new_activations)

        # === 4. Determine which neurons fire ===
        new_firing = can_receive & (new_activations >= state.thresholds_current)

        # === 5. Handle post-fire state ===
        if state.use_refraction_decay:
            # Set refractory counters for neurons that just fired
            new_refractory_counters = state.refractory_counters.copy()
            new_refractory_counters = np.where(
                new_firing, state.refraction_period, new_refractory_counters
            )
        else:
            # Reset activation to 0 for neurons that fired
            new_activations = np.where(new_firing, 0, new_activations)
            new_refractory_counters = state.refractory_counters

        # === 6. Apply refraction leak to neurons that WERE refractory (old counter) ===
        if state.use_refraction_decay:
            was_refractory = state.refractory_counters > 0
            new_activations = np.where(
                was_refractory,
                new_activations * state.refraction_leak,
                new_activations,
            )

        # === 7. Update threshold variations (for neurons with period > 0) ===
        has_variation = state.threshold_variation_periods > 0
        if np.any(has_variation):
            # Vectorized sinusoidal threshold update
            phase = (
                step
                * 2
                * np.pi
                / np.where(has_variation, state.threshold_variation_periods, 1)
            )
            variation = np.sin(phase) * state.threshold_variation_ranges
            new_thresholds = state.thresholds + variation
            state.thresholds_current = np.where(
                has_variation,
                np.clip(new_thresholds, 0, 1),
                state.thresholds_current,
            )

        # === 8. Calculate outputs (matrix-vector multiply) ===
        new_outputs = state.outputs * state.refraction_leak
        effective_output_weights = self._apply_weight_threshold(state.output_weights)
        new_outputs += effective_output_weights.T @ new_firing.astype(np.float64)

        # === 9. Apply activation leak ===
        if state.use_activation_leak:
            new_activations *= state.activation_leak

        # === 10. Decrement refractory counters ===
        if state.use_refraction_decay:
            new_refractory_counters = np.maximum(0, new_refractory_counters - 1)

        # === Update state ===
        state.activations = new_activations
        state.firing = new_firing
        state.outputs = np.clip(new_outputs, 0, 1)
        if state.use_refraction_decay:
            state.refractory_counters = new_refractory_counters

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

    def manual_activate_most_weighted_per_module(self, value: float):
        """Activate the most connected neuron in each module.

        Requires randomize_modular_weights() to have been called first.
        """
        if not hasattr(self, "module_assignments"):
            raise ValueError(
                "No module assignments found. Call randomize_modular_weights() first."
            )

        total_network_weights = np.sum(self.state.network_weights, axis=1)
        activated = []

        for m in range(int(np.max(self.module_assignments)) + 1):
            # Get neurons in this module
            module_neurons = np.where(self.module_assignments == m)[0]

            # Find most weighted neuron within this module
            module_weights = total_network_weights[module_neurons]
            best_idx = np.argmax(module_weights)
            best_neuron = module_neurons[best_idx]

            self.manual_activate(best_neuron, value)
            activated.append(best_neuron)
            print(
                f"Module {m}: activated neuron {best_neuron} (weight: {total_network_weights[best_neuron]:.3f})"
            )

        return activated

    def clear_firing(self):
        """Clear all firing states and outputs."""
        self.state.firing = np.zeros(self.state.num_neurons, dtype=bool)
        self.state.outputs = np.zeros(self.state.num_outputs)

    def enable_activation_leak(self, leak_factor: float = 0.95):
        self.state.use_activation_leak = True
        self.state.activation_leak = np.clip(leak_factor, 0, 1)

    def disable_activation_leak(self):
        self.state.use_activation_leak = False

    def set_weight_threshold(self, threshold: float = 0.05):
        """Set minimum weight magnitude to have effect. Weights with |w| < threshold are ignored."""
        self.state.weight_threshold = max(0, threshold)

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

    def randomize_modular_weights(
        self, n_modules=4, intra_sparsity=0.3, inter_sparsity=0.02, scale=0.4
    ):
        """Randomize weights with modular/block structure.

        Creates n_modules groups of neurons with high intra-module connectivity
        and low inter-module connectivity. This encourages independent clusters.

        Args:
            n_modules: Number of modules to create
            intra_sparsity: Fraction of connections within each module (0-1)
            inter_sparsity: Fraction of connections between modules (0-1)
            scale: Standard deviation of the gaussian distribution
        """
        N = self.state.num_neurons
        module_size = N // n_modules

        # Gaussian weights, clipped to [-1, 1]
        self.state.network_weights = np.clip(np.random.randn(N, N) * scale, -1, 1)

        # Build sparsity mask with block structure
        mask = np.zeros((N, N))

        for m in range(n_modules):
            start = m * module_size
            # Handle last module getting any remainder neurons
            end = (m + 1) * module_size if m < n_modules - 1 else N

            # Intra-module connections (dense)
            intra_mask = np.random.random((end - start, end - start)) < intra_sparsity
            mask[start:end, start:end] = intra_mask

            # Inter-module connections (sparse)
            for m2 in range(n_modules):
                if m2 != m:
                    start2 = m2 * module_size
                    end2 = (m2 + 1) * module_size if m2 < n_modules - 1 else N
                    inter_mask = (
                        np.random.random((end - start, end2 - start2)) < inter_sparsity
                    )
                    mask[start:end, start2:end2] = inter_mask

        self.state.network_weights *= mask

        # Store module assignments for reference
        self.module_assignments = np.zeros(N, dtype=int)
        for m in range(n_modules):
            start = m * module_size
            end = (m + 1) * module_size if m < n_modules - 1 else N
            self.module_assignments[start:end] = m

        print(
            f"Created {n_modules} modules: intra={intra_sparsity:.0%}, inter={inter_sparsity:.0%}"
        )
        for m in range(n_modules):
            count = np.sum(self.module_assignments == m)
            print(
                f"  Module {m}: neurons {np.where(self.module_assignments == m)[0][0]}-{np.where(self.module_assignments == m)[0][-1]} ({count} neurons)"
            )

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
            num_neurons=self.state.num_neurons,
            num_readouts=self.state.num_readouts,
            n_outputs_per_readout=self.state.n_outputs_per_readout,
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
