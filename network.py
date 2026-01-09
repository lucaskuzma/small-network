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
