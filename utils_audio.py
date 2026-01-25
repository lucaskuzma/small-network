"""
Audio synthesis utilities for neural network outputs.

Maps network outputs to oscillator parameters and synthesizes audio.
"""

import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SAMPLE_RATE = 11025  # Quick iteration rate (44100 for final)
DEFAULT_FREQ_MIN = 100.0  # Hz
DEFAULT_FREQ_MAX = 800.0  # Hz (covers ~3 octaves)


# =============================================================================
# Synthesis
# =============================================================================


def map_output_to_freq(
    output: np.ndarray,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
) -> np.ndarray:
    """
    Map network output (0-1) to frequency (Hz).

    Uses exponential mapping so equal output differences = equal musical intervals.
    """
    # Exponential mapping: output 0 -> freq_min, output 1 -> freq_max
    # freq = freq_min * (freq_max/freq_min)^output
    output = np.clip(output, 0, 1)
    return freq_min * np.power(freq_max / freq_min, output)


def synthesize_oscillators(
    output_history: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize audio from network outputs using sine oscillators.

    Args:
        output_history: (T, num_voices, 3) array where last dim is [freq, phase_mod, amp]
        sample_rate: Audio sample rate in Hz
        freq_min: Minimum frequency (Hz) when output=0
        freq_max: Maximum frequency (Hz) when output=1

    Returns:
        audio: (T,) mixed audio signal, normalized to [-1, 1]
        freq_history: (T, num_voices) frequency values for analysis
    """
    num_samples, num_voices, num_outputs = output_history.shape
    assert num_outputs == 3, f"Expected 3 outputs per voice, got {num_outputs}"

    # Extract parameters
    freq_raw = output_history[:, :, 0]  # (T, V)
    phase_mod = output_history[:, :, 1]  # (T, V) - phase modulation depth
    amp = output_history[:, :, 2]  # (T, V)

    # Map to frequencies
    freqs = map_output_to_freq(freq_raw, freq_min, freq_max)  # (T, V)

    # Synthesize each voice
    voices = np.zeros((num_samples, num_voices))
    phases = np.zeros(num_voices)  # Current phase for each oscillator

    for t in range(num_samples):
        for v in range(num_voices):
            # Phase increment based on frequency
            phase_inc = 2 * np.pi * freqs[t, v] / sample_rate

            # Phase modulation (subtle, scaled by output)
            phase_offset = phase_mod[t, v] * np.pi * 0.5  # Max ±π/2 modulation

            # Generate sample
            voices[t, v] = amp[t, v] * np.sin(phases[v] + phase_offset)

            # Advance phase
            phases[v] = (phases[v] + phase_inc) % (2 * np.pi)

    # Mix voices (simple sum, then normalize)
    mixed = np.sum(voices, axis=1)

    # Normalize to [-1, 1] with headroom
    max_amp = np.max(np.abs(mixed))
    if max_amp > 0:
        mixed = mixed / max_amp * 0.9

    return mixed, freqs


def synthesize_oscillators_vectorized(
    output_history: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
    amp_smoothing: float = 0.999,  # Leaky integrator for amplitude
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized synthesis (faster for long sequences).

    Uses leaky integrator for amplitude so transient spikes become sustained tones.

    Args:
        output_history: (T, num_voices, 3) array
        sample_rate: Audio sample rate
        freq_min: Min frequency mapping
        freq_max: Max frequency mapping  
        amp_smoothing: Decay factor for amplitude integrator (0.999 = slow decay)
    """
    num_samples, num_voices, num_outputs = output_history.shape
    assert num_outputs == 3, f"Expected 3 outputs per voice, got {num_outputs}"

    # Extract parameters
    freq_raw = output_history[:, :, 0]  # (T, V)
    phase_mod = output_history[:, :, 1]  # (T, V)
    amp_raw = output_history[:, :, 2]  # (T, V)

    # Map to frequencies
    freqs = map_output_to_freq(freq_raw, freq_min, freq_max)  # (T, V)

    # Smooth amplitude with leaky integrator: amp[t] = decay * amp[t-1] + (1-decay) * raw[t]
    # This turns brief spikes into sustained envelopes
    amp = np.zeros_like(amp_raw)
    amp[0] = amp_raw[0]
    for t in range(1, num_samples):
        amp[t] = amp_smoothing * amp[t - 1] + (1 - amp_smoothing) * amp_raw[t]

    # Compute phase increments and cumulative phase
    phase_incs = 2 * np.pi * freqs / sample_rate  # (T, V)
    phases = np.cumsum(phase_incs, axis=0)  # (T, V)
    phases = np.mod(phases, 2 * np.pi)

    # Phase modulation
    phase_offsets = phase_mod * np.pi * 0.5  # (T, V)

    # Generate samples
    voices = amp * np.sin(phases + phase_offsets)  # (T, V)

    # Mix
    mixed = np.sum(voices, axis=1)  # (T,)

    # Normalize
    max_amp = np.max(np.abs(mixed))
    if max_amp > 0:
        mixed = mixed / max_amp * 0.9

    return mixed, freqs


# =============================================================================
# File I/O
# =============================================================================


def synthesize_raw_multiply(
    output_history: np.ndarray,
    stretch_factor: int = 1,
) -> np.ndarray:
    """
    Convert raw network outputs directly to audio by multiplying channels.
    
    Each voice has 3 outputs. All 3 must be high simultaneously to produce sound.
    This forces the network to coordinate its outputs.
    
    Args:
        output_history: (T, num_voices, 3) array of outputs in [0, 1]
        stretch_factor: Interpolate to this many audio samples per network step.
                       E.g., stretch_factor=8 means 256 network steps → 2048 audio samples.
        
    Returns:
        audio: (T * stretch_factor,) mixed audio signal in [-1, 1]
    """
    # Multiply the 3 channels per voice: requires coordination
    per_voice = np.prod(output_history, axis=2)  # (T, num_voices)
    
    # Shift from [0, 1] to [-1, 1]
    per_voice = per_voice * 2 - 1
    
    # Mix voices by averaging
    audio = np.mean(per_voice, axis=1)  # (T,)
    
    # Interpolate if stretch_factor > 1
    if stretch_factor > 1:
        T = len(audio)
        new_T = T * stretch_factor
        audio = np.interp(
            np.linspace(0, T - 1, new_T),
            np.arange(T),
            audio,
        )
    
    return audio


def save_wav(
    samples: np.ndarray,
    filename: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """
    Save audio samples to WAV file.

    Args:
        samples: (T,) array of samples in [-1, 1]
        filename: Output path
        sample_rate: Sample rate in Hz

    Returns:
        Path to saved file
    """
    # Convert to 16-bit PCM
    samples = np.clip(samples, -1, 1)
    pcm = (samples * 32767).astype(np.int16)

    with wave.open(filename, "w") as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    return filename


def load_wav(filename: str) -> tuple[np.ndarray, int]:
    """
    Load WAV file and return samples and sample rate.

    Returns:
        samples: (T,) array of samples in [-1, 1]
        sample_rate: Sample rate in Hz
    """
    with wave.open(filename, "r") as wav:
        sample_rate = wav.getframerate()
        num_frames = wav.getnframes()
        raw_data = wav.readframes(num_frames)

        # Assume 16-bit mono
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
        samples = samples / 32767.0

    return samples, sample_rate


# =============================================================================
# Visualization
# =============================================================================


def save_waveform_plot(
    samples: np.ndarray,
    filename: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    title: Optional[str] = None,
) -> str:
    """
    Save waveform (amplitude vs time) plot.

    Args:
        samples: (T,) audio samples
        filename: Output PNG path
        sample_rate: Sample rate for time axis
        title: Optional plot title

    Returns:
        Path to saved file
    """
    duration = len(samples) / sample_rate
    time = np.linspace(0, duration, len(samples))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, samples, linewidth=0.5, color="#3498db")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    return filename


def save_frequency_plot(
    freq_history: np.ndarray,
    filename: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    title: Optional[str] = None,
) -> str:
    """
    Save frequency over time plot for each voice.

    Args:
        freq_history: (T, num_voices) frequency values in Hz
        filename: Output PNG path
        sample_rate: Sample rate for time axis
        title: Optional plot title

    Returns:
        Path to saved file
    """
    num_samples, num_voices = freq_history.shape
    duration = num_samples / sample_rate
    time = np.linspace(0, duration, num_samples)

    # Voice colors
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#e67e22"]

    fig, ax = plt.subplots(figsize=(12, 4))

    for v in range(num_voices):
        ax.plot(
            time,
            freq_history[:, v],
            linewidth=0.8,
            color=colors[v % len(colors)],
            label=f"Voice {v+1}",
            alpha=0.8,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim(0, duration)
    ax.set_yscale("log")  # Log scale for frequency (musical perception)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    return filename


def save_spectrogram(
    samples: np.ndarray,
    filename: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    title: Optional[str] = None,
) -> str:
    """
    Save spectrogram of audio.

    Args:
        samples: (T,) audio samples
        filename: Output PNG path
        sample_rate: Sample rate in Hz
        title: Optional plot title

    Returns:
        Path to saved file
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Use short FFT window for time resolution
    nperseg = min(256, len(samples) // 4)

    ax.specgram(
        samples,
        Fs=sample_rate,
        NFFT=nperseg,
        noverlap=nperseg // 2,
        cmap="magma",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 2000)  # Focus on lower frequencies

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    return filename


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    # Test synthesis with random network-like outputs
    import os

    np.random.seed(42)

    num_samples = DEFAULT_SAMPLE_RATE  # 1 second
    num_voices = 3

    # Simulate network outputs: smooth random walks
    output_history = np.zeros((num_samples, num_voices, 3))

    for v in range(num_voices):
        # Random walk for each parameter, clipped to [0, 1]
        for p in range(3):
            walk = np.cumsum(np.random.randn(num_samples) * 0.01)
            walk = (walk - walk.min()) / (walk.max() - walk.min() + 1e-6)
            output_history[:, v, p] = walk

    # Synthesize
    audio, freqs = synthesize_oscillators_vectorized(output_history)

    # Save outputs
    os.makedirs("test_audio", exist_ok=True)
    save_wav(audio, "test_audio/test_synth.wav")
    save_waveform_plot(audio, "test_audio/test_waveform.png", title="Test Synthesis")
    save_frequency_plot(freqs, "test_audio/test_frequencies.png", title="Voice Frequencies")
    save_spectrogram(audio, "test_audio/test_spectrogram.png", title="Spectrogram")

    print("Test outputs saved to test_audio/")
    print(f"  Duration: {len(audio) / DEFAULT_SAMPLE_RATE:.2f}s")
    print(f"  Freq range: {freqs.min():.1f} - {freqs.max():.1f} Hz")

