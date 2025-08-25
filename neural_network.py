# %%

from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class NeuralNetworkState:
    network_weights: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    thresholds: np.ndarray = field(default_factory=lambda: np.full((4,), 0.5))
    output_weights: np.ndarray = field(default_factory=lambda: np.eye(4))
    activations: np.ndarray = field(default_factory=lambda: np.zeros(4))
    firing: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))
    outputs: np.ndarray = field(default_factory=lambda: np.zeros(4))

    use_activation_leak: bool = False
    activation_leak: float = 0.95
    use_refraction_decay: bool = False
    refraction_period: int = 3
    refractory_counters: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=int)
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

    def randomize_weights(self):
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


# =======================================================================
def plot_neural_heatmap(history, data_type="activations"):
    steps_to_show = len(history.get("step", []))
    tick_step = steps_to_show // 16

    data_matrix = np.array(history[data_type][:steps_to_show])

    fig, ax = plt.subplots(figsize=(16, 4))

    sns.heatmap(
        data_matrix.T,
        annot=False,
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": f"{data_type.capitalize()} Value"},
        yticklabels=[f"N{i}" for i in range(4)],
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

network = NeuralNetwork()
steps = 256
network.clear()

network.update_network_weight(0, 1, 0.5)  # N0 → N1
network.update_network_weight(1, 2, 0.5)  # N1 → N2
network.update_network_weight(2, 3, 0.5)  # N2 → N3
network.update_network_weight(3, 0, 0.5)  # N3 → N0 (feedback loop)

network.state.thresholds = np.full((4,), 0.75)

network.set_output_identity()

network.enable_activation_leak(0.9)
network.enable_refraction_decay(3)


history = {"activations": [], "firing": [], "outputs": [], "step": []}

# input patterns
stimulators = [
    [1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
stimulator_strength = 0.2

# Initial state
# network.manual_trigger(0)
# history["activations"].append(network.state.activations.copy())
# history["step"].append(0)

# Run simulation
for step in range(steps):
    # network.manual_activate(0, 0.1)
    for i, pattern in enumerate(stimulators):
        network.manual_activate(i, pattern[step % 4] * stimulator_strength)
    network.tick()
    history["activations"].append(network.state.activations.copy())
    history["firing"].append(network.state.firing.copy())
    history["outputs"].append(network.state.outputs.copy())
    history["step"].append(step)

# Call the function to plot both heatmaps
plot_neural_heatmap(history, "activations")
plot_neural_heatmap(history, "firing")
