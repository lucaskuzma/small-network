"""
Neural Network Sequencer Implementation

Core class for the 4-neuron echo state network sequencer.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class NeuralNetworkState:
    """State of the neural network at any given time."""

    network_weights: np.ndarray
    thresholds: np.ndarray
    output_weights: np.ndarray
    activations: np.ndarray
    firing: np.ndarray
    outputs: np.ndarray
    # New parameters for enhanced dynamics
    use_activation_leak: bool = False
    activation_leak: float = 0.95
    use_refraction_decay: bool = False
    refraction_period: int = 3
    # Track refractory state for each neuron
    refractory_counters: np.ndarray = None

    def __init__(self, **kwargs):
        self.network_weights = kwargs.get("network_weights", np.zeros((4, 4)))
        self.thresholds = kwargs.get("thresholds", np.full((4,), 0.5))
        self.output_weights = kwargs.get("output_weights", np.eye(4))
        self.activations = kwargs.get("activations", np.zeros(4))
        self.firing = kwargs.get("firing", np.zeros(4, dtype=bool))
        self.outputs = kwargs.get("outputs", np.zeros(4))
        # Initialize new parameters
        self.use_activation_leak = kwargs.get("use_activation_leak", False)
        self.activation_leak = kwargs.get("activation_leak", 0.95)
        self.use_refraction_decay = kwargs.get("use_refraction_decay", False)
        self.refraction_period = kwargs.get("refraction_period", 3)
        # Initialize refractory counters
        self.refractory_counters = kwargs.get(
            "refractory_counters", np.zeros(4, dtype=int)
        )


class NeuralNetwork:
    """
    4-Neuron Echo State Network Sequencer

    This class implements a discrete-time neural network that can be
    driven by external clock ticks. Each neuron has activation levels,
    firing thresholds, and contributes to output values through
    weighted connections.
    """

    def __init__(self, initial_state: Optional[NeuralNetworkState] = None):
        """
        Initialize the neural network.

        Args:
            initial_state: Optional initial state, uses defaults if None
        """
        if initial_state is None:
            self.state = NeuralNetworkState()
        else:
            self.state = initial_state

    def tick(self):
        """
        Process one clock tick - the main simulation step.

        This method should be called externally for each clock step.
        It processes activations, determines firing, and calculates outputs.
        """
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
        """
        Manually trigger a specific neuron.

        Args:
            neuron_index: Index of the neuron to trigger (0-3)
        """
        if not 0 <= neuron_index < 4:
            raise ValueError("Neuron index must be between 0 and 3")

        # Set the neuron to fire
        # self.state.firing[neuron_index] = True

        # Set activation to 1
        self.state.activations[neuron_index] = 1

    def manual_activate(self, neuron_index: int, value: float):
        """
        Manually set the activation of a specific neuron.

        Args:
            neuron_index: Index of the neuron to activate (0-3)
            value: Value to add to the activation (0-1)
        """
        self.state.activations[neuron_index] += value
        self.state.activations[neuron_index] = np.clip(
            self.state.activations[neuron_index], 0, 1
        )

    def clear_firing(self):
        """Clear all firing states and outputs."""
        self.state.firing = np.zeros(4, dtype=bool)
        self.state.outputs = np.zeros(4)

    def update_network_weight(self, row: int, col: int, value: float):
        """
        Update a specific network weight.

        Args:
            row: Row index (0-3)
            col: Column index (0-3)
            value: New weight value
        """
        if not (0 <= row < 4 and 0 <= col < 4):
            raise ValueError("Indices must be between 0 and 3")

        self.state.network_weights[row, col] = np.clip(value, 0, 1)

    def update_threshold(self, index: int, value: float):
        """
        Update a specific neuron threshold.

        Args:
            index: Neuron index (0-3)
            value: New threshold value
        """
        if not 0 <= index < 4:
            raise ValueError("Index must be between 0 and 3")

        self.state.thresholds[index] = np.clip(value, 0, 1)

    def update_output_weight(self, row: int, col: int, value: float):
        """
        Update a specific output weight.

        Args:
            row: Row index (0-3)
            col: Column index (0-3)
            value: New weight value
        """
        if not (0 <= row < 4 and 0 <= col < 4):
            raise ValueError("Indices must be between 0 and 3")

        self.state.output_weights[row, col] = np.clip(value, 0, 1)

    def enable_activation_leak(self, leak_factor: float = 0.95):
        """
        Enable activation leak with specified decay factor.

        Args:
            leak_factor: Decay factor (0-1). Lower values = stronger decay.
                        Recommended: 0.3 for strong refractory effect, 0.95 for subtle decay
        """
        self.state.use_activation_leak = True
        self.state.activation_leak = np.clip(leak_factor, 0, 1)

    def disable_activation_leak(self):
        """Disable activation leak."""
        self.state.use_activation_leak = False

    def enable_refraction_decay(self, refraction_period: int = 3):
        """
        Enable refractory period using activation decay.

        Args:
            refraction_period: Approximate number of steps for refractory period.
                              Works best with activation_leak < 0.5
        """
        self.state.use_refraction_decay = True
        self.state.refraction_period = max(1, refraction_period)

    def disable_refraction_decay(self):
        """Disable refractory period."""
        self.state.use_refraction_decay = False

    def set_dynamics_parameters(
        self,
        use_activation_leak: bool = None,
        activation_leak: float = None,
        use_refraction_decay: bool = None,
        refraction_period: int = None,
    ):
        """
        Set multiple dynamics parameters at once.

        Args:
            use_activation_leak: Whether to enable activation leak
            activation_leak: Decay factor for activation leak (0-1)
            use_refraction_decay: Whether to enable refractory period
            refraction_period: Approximate refractory period in steps
        """
        if use_activation_leak is not None:
            self.state.use_activation_leak = use_activation_leak
        if activation_leak is not None:
            self.state.activation_leak = np.clip(activation_leak, 0, 1)
        if use_refraction_decay is not None:
            self.state.use_refraction_decay = use_refraction_decay
        if refraction_period is not None:
            self.state.refraction_period = max(1, refraction_period)

    def reset_dynamics_to_defaults(self):
        """Reset all dynamics parameters to their default values."""
        self.state.use_activation_leak = False
        self.state.activation_leak = 0.95
        self.state.use_refraction_decay = False
        self.state.refraction_period = 3

    def randomize_all(self):
        """Randomize all weights and thresholds."""
        # Randomize network weights (0 to 1)
        self.state.network_weights = np.random.random((4, 4))

        # Randomize thresholds (0 to 1)
        self.state.thresholds = np.random.random(4)

        # Randomize output weights (0 to 1)
        self.state.output_weights = np.random.random((4, 4))

    def randomize_thresholds(self):
        """Randomize all thresholds."""
        # Randomize network weights (0 to 1)
        self.state.thresholds = np.random.random(4)

    def randomize_weights(self):
        """Randomize all weights."""
        # Randomize network weights (0 to 1)
        self.state.network_weights = np.random.random((4, 4))

    def clear(self):
        """Clear all weights and reset to initial state."""
        self.state = NeuralNetworkState()

    def set_output_identity(self):
        """Set output weights to identity matrix (1:1 mapping)."""
        self.state.output_weights = np.eye(4)

    def get_state(self) -> NeuralNetworkState:
        """Get a copy of the current state."""
        return NeuralNetworkState(
            network_weights=self.state.network_weights.copy(),
            thresholds=self.state.thresholds.copy(),
            output_weights=self.state.output_weights.copy(),
            activations=self.state.activations.copy(),
            firing=self.state.firing.copy(),
            outputs=self.state.outputs.copy(),
            use_activation_leak=self.state.use_activation_leak,
            activation_leak=self.state.activation_leak,
            use_refraction_decay=self.state.use_refraction_decay,
            refraction_period=self.state.refraction_period,
            refractory_counters=self.state.refractory_counters.copy(),
        )

    def set_state(self, new_state: NeuralNetworkState):
        """Set the network to a specific state."""
        self.state = new_state
        self._ensure_numpy_arrays()

    def get_network_summary(self) -> dict:
        """Get a summary of the current network state."""
        return {
            "activations": self.state.activations.tolist(),
            "firing": self.state.firing.tolist(),
            "outputs": self.state.outputs.tolist(),
        }

    def __str__(self) -> str:
        """String representation of the network state."""
        summary = self.get_network_summary()
        return f"NeuralNetwork(activations={summary['activations']}, firing={summary['firing']}, outputs={summary['outputs']})"

    def __repr__(self) -> str:
        """Detailed representation of the network."""
        return f"NeuralNetwork(state={self.state})"
