import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class NeuralNetworkState:
    network_weights: np.ndarray
    thresholds: np.ndarray
    output_weights: np.ndarray
    activations: np.ndarray
    firing: np.ndarray
    outputs: np.ndarray

    use_activation_leak: bool = False
    activation_leak: float = 0.95
    use_refraction_decay: bool = False
    refraction_period: int = 3
    refractory_counters: np.ndarray = None

    def __init__(self, **kwargs):
        self.network_weights = kwargs.get("network_weights", np.zeros((4, 4)))
        self.thresholds = kwargs.get("thresholds", np.full((4,), 0.5))
        self.output_weights = kwargs.get("output_weights", np.eye(4))
        self.activations = kwargs.get("activations", np.zeros(4))
        self.firing = kwargs.get("firing", np.zeros(4, dtype=bool))
        self.outputs = kwargs.get("outputs", np.zeros(4))
        # dynamics
        self.use_activation_leak = kwargs.get("use_activation_leak", False)
        self.activation_leak = kwargs.get("activation_leak", 0.95)
        self.use_refraction_decay = kwargs.get("use_refraction_decay", False)
        self.refraction_period = kwargs.get("refraction_period", 3)
        # refractory counters
        self.refractory_counters = kwargs.get(
            "refractory_counters", np.zeros(4, dtype=int)
        )


class NeuralNetwork:
    def __init__(self, initial_state: Optional[NeuralNetworkState] = None):
        if initial_state is None:
            self.state = NeuralNetworkState()
        else:
            self.state = initial_state

    def tick(self):
        # Calculate new activations from current firing neurons
        new_activations = self.state.activations.copy()
        new_firing = np.zeros(4, dtype=bool)
        new_refractory_counters = self.state.refractory_counters.copy()

        for i in range(4):
            activation = 0
            for j in range(4):
                if self.state.firing[j]:
                    activation += self.state.network_weights[j, i]

            new_activations[i] = np.clip(new_activations[i] + activation, 0, 1)

            # Check if neuron should fire (only if not in refractory period)
            if new_activations[i] >= self.state.thresholds[i] and (
                not self.state.use_refraction_decay or new_refractory_counters[i] == 0
            ):
                new_firing[i] = True
                # Set refractory counter if enabled
                if self.state.use_refraction_decay:
                    new_refractory_counters[i] = self.state.refraction_period
                else:
                    new_activations[i] = 0

        # Calculate outputs based on firing neurons
        new_outputs = np.zeros(4)
        for i in range(4):
            if new_firing[i]:
                for j in range(4):
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
        if not 0 <= neuron_index < 4:
            raise ValueError("Neuron index must be between 0 and 3")

        # Set the neuron to fire
        # self.state.firing[neuron_index] = True

        # Set activation to 1
        self.state.activations[neuron_index] = 1

    def manual_activate(self, neuron_index: int, value: float):
        self.state.activations[neuron_index] += value
        self.state.activations[neuron_index] = np.clip(
            self.state.activations[neuron_index], 0, 1
        )

    def clear_firing(self):
        """Clear all firing states and outputs."""
        self.state.firing = np.zeros(4, dtype=bool)
        self.state.outputs = np.zeros(4)

    def update_network_weight(self, row: int, col: int, value: float):
        if not (0 <= row < 4 and 0 <= col < 4):
            raise ValueError("Indices must be between 0 and 3")

        self.state.network_weights[row, col] = np.clip(value, 0, 1)

    def update_threshold(self, index: int, value: float):
        if not 0 <= index < 4:
            raise ValueError("Index must be between 0 and 3")

        self.state.thresholds[index] = np.clip(value, 0, 1)

    def update_output_weight(self, row: int, col: int, value: float):
        if not (0 <= row < 4 and 0 <= col < 4):
            raise ValueError("Indices must be between 0 and 3")

        self.state.output_weights[row, col] = np.clip(value, 0, 1)

    def enable_activation_leak(self, leak_factor: float = 0.95):

        self.state.use_activation_leak = True
        self.state.activation_leak = np.clip(leak_factor, 0, 1)

    def disable_activation_leak(self):
        self.state.use_activation_leak = False

    def enable_refraction_decay(self, refraction_period: int = 3):
        self.state.use_refraction_decay = True
        self.state.refraction_period = max(1, refraction_period)

    def disable_refraction_decay(self):
        self.state.use_refraction_decay = False

    def set_dynamics_parameters(
        self,
        use_activation_leak: bool = None,
        activation_leak: float = None,
        use_refraction_decay: bool = None,
        refraction_period: int = None,
    ):
        if use_activation_leak is not None:
            self.state.use_activation_leak = use_activation_leak
        if activation_leak is not None:
            self.state.activation_leak = np.clip(activation_leak, 0, 1)
        if use_refraction_decay is not None:
            self.state.use_refraction_decay = use_refraction_decay
        if refraction_period is not None:
            self.state.refraction_period = max(1, refraction_period)

    def reset_dynamics_to_defaults(self):
        self.state.use_activation_leak = False
        self.state.activation_leak = 0.95
        self.state.use_refraction_decay = False
        self.state.refraction_period = 3

    def randomize_weights(self):
        # Randomize network weights (0 to 1)
        self.state.network_weights = np.random.random((4, 4))

    def clear(self):
        self.state = NeuralNetworkState()

    def set_output_identity(self):
        self.state.output_weights = np.eye(4)

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
