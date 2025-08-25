# %%

from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class NeuralNetworkState:
    num_neurons: int = 64
    network_weights: np.ndarray = field(init=False)
    thresholds: np.ndarray = field(init=False)
    output_weights: np.ndarray = field(init=False)
    activations: np.ndarray = field(init=False)
    firing: np.ndarray = field(init=False)
    outputs: np.ndarray = field(init=False)
    refractory_counters: np.ndarray = field(init=False)

    use_activation_leak: bool = False
    activation_leak: float = 0.95
    refraction_leak: float = 0.4
    use_refraction_decay: bool = False
    refraction_period: int = 3
    use_tanh_activation: bool = False

    def __post_init__(self):
        self.network_weights = np.zeros((self.num_neurons, self.num_neurons))
        self.thresholds = np.full((self.num_neurons,), 0.5)
        self.output_weights = np.eye(self.num_neurons)
        self.activations = np.zeros(self.num_neurons)
        self.firing = np.zeros(self.num_neurons, dtype=bool)
        self.outputs = np.zeros(self.num_neurons)
        self.refractory_counters = np.zeros(self.num_neurons, dtype=int)


class NeuralNetwork:
    def __init__(
        self, num_neurons: int = 64, initial_state: Optional[NeuralNetworkState] = None
    ):
        if initial_state is None:
            self.state = NeuralNetworkState(num_neurons=num_neurons)
        else:
            self.state = initial_state

    def tick(self):
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
                incoming_activation /= self.state.num_neurons

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
                if new_activations[i] >= self.state.thresholds[i]:

                    # fire!
                    new_firing[i] = True

                    if self.state.use_refraction_decay:
                        new_refractory_counters[i] = self.state.refraction_period
                    else:
                        new_activations[i] = 0

            if self.state.use_refraction_decay:
                if new_refractory_counters[i] > 0:
                    new_activations[i] *= self.state.refraction_leak

        # Calculate outputs based on firing neurons
        new_outputs = np.zeros(self.state.num_neurons)
        for i in range(self.state.num_neurons):
            if new_firing[i]:
                for j in range(self.state.num_neurons):
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

    def clear_firing(self):
        """Clear all firing states and outputs."""
        self.state.firing = np.zeros(self.state.num_neurons, dtype=bool)
        self.state.outputs = np.zeros(self.state.num_neurons)

    def update_network_weight(self, row: int, col: int, value: float):
        if not (
            0 <= row < self.state.num_neurons and 0 <= col < self.state.num_neurons
        ):
            raise ValueError(
                f"Indices must be between 0 and {self.state.num_neurons - 1}"
            )

        self.state.network_weights[row, col] = np.clip(value, 0, 1)

    def update_threshold(self, index: int, value: float):
        if not 0 <= index < self.state.num_neurons:
            raise ValueError(
                f"Index must be between 0 and {self.state.num_neurons - 1}"
            )

        self.state.thresholds[index] = np.clip(value, 0, 1)

    def update_output_weight(self, row: int, col: int, value: float):
        if not (
            0 <= row < self.state.num_neurons and 0 <= col < self.state.num_neurons
        ):
            raise ValueError(
                f"Indices must be between 0 and {self.state.num_neurons - 1}"
            )

        self.state.output_weights[row, col] = np.clip(value, 0, 1)

    def enable_activation_leak(self, leak_factor: float = 0.95):
        self.state.use_activation_leak = True
        self.state.activation_leak = np.clip(leak_factor, 0, 1)

    def disable_activation_leak(self):
        self.state.use_activation_leak = False

    def enable_refraction_decay(
        self, refraction_period: int = 3, refraction_leak: float = 0.4
    ):
        self.state.refraction_leak = np.clip(refraction_leak, 0, 1)
        self.state.use_refraction_decay = True
        self.state.refraction_period = max(1, refraction_period)

    def disable_refraction_decay(self):
        self.state.use_refraction_decay = False

    def randomize_weights(self):
        self.state.network_weights = np.random.random(
            (self.state.num_neurons, self.state.num_neurons)
        )

    def randomize_thresholds(self):
        self.state.thresholds = np.random.random(self.state.num_neurons)

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
        self.state = NeuralNetworkState(num_neurons=self.state.num_neurons)

    def set_output_identity(self):
        self.state.output_weights = np.eye(self.state.num_neurons)

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

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.heatmap(
        data_matrix.T,
        annot=False,
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": f"Weight Value"},
        yticklabels=[f"N{i}" for i in range(num_neurons)],
    )

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
    ax.set_title(f"Neural Network {data_type.capitalize()} Over Time")

    plt.tight_layout()
    plt.show()


# =======================================================================

network = NeuralNetwork(num_neurons=16)
steps = 256
network.clear()
network.set_output_identity()

# network.update_network_weight(0, 1, 0.5)  # N0 → N1
# network.update_network_weight(1, 2, 0.5)  # N1 → N2
# network.update_network_weight(2, 3, 0.5)  # N2 → N3
# network.update_network_weight(3, 0, 0.5)  # N3 → N0 (feedback loop)

network.randomize_weights()
# network.sinusoidal_weights()
network.randomize_thresholds()
network.set_diagonal_weights(0)  # no self-feedback

network.enable_activation_leak(0.97)
network.enable_refraction_decay(5, 0.5)


history = {"activations": [], "firing": [], "outputs": [], "step": []}

# input patterns
stimulators = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0],
    [0],
    [0],
    [0],
    [1, 0, 0, 0, 0, 0, 0, 0],
]
stimulator_strength = 0.25

# Initial state
# network.manual_trigger(0)
# history["activations"].append(network.state.activations.copy())
# history["step"].append(0)

# Run simulation
for step in range(steps):
    # network.manual_activate(0, 0.1)
    for i, pattern in enumerate(stimulators):
        network.manual_activate(i, pattern[step % len(pattern)] * stimulator_strength)
    network.tick()
    history["activations"].append(network.state.activations.copy())
    history["firing"].append(network.state.firing.copy())
    history["outputs"].append(network.state.outputs.copy())
    history["step"].append(step)


plot_weight_heatmap(network.state.num_neurons)
plot_neural_heatmap(history, "activations", network.state.num_neurons)
plot_neural_heatmap(history, "firing", network.state.num_neurons)
