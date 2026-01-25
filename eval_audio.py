"""
Audio fitness evaluation based on consonance.

Consonant intervals have frequency ratios that are small integer fractions.
This evaluator scores how close voice frequency ratios are to these "pure" intervals.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from utils_audio import (
    map_output_to_freq,
    DEFAULT_FREQ_MIN,
    DEFAULT_FREQ_MAX,
    DEFAULT_SAMPLE_RATE,
)


# =============================================================================
# Consonance definitions
# =============================================================================

# Consonant frequency ratios (relative to lowest voice)
# Format: (ratio, name, consonance_weight)
# Weight reflects how "pure" the interval is (1.0 = most consonant)
CONSONANT_INTERVALS = [
    (1.0, "unison", 1.0),
    (1.0595, "minor 2nd", 0.2),  # 2^(1/12) - dissonant
    (1.1225, "major 2nd", 0.3),  # 2^(2/12)
    (1.1892, "minor 3rd", 0.7),  # 2^(3/12) = 6:5 ≈ 1.2
    (1.2599, "major 3rd", 0.8),  # 2^(4/12) = 5:4 = 1.25
    (1.3348, "perfect 4th", 0.9),  # 2^(5/12) = 4:3 ≈ 1.333
    (1.4142, "tritone", 0.1),  # 2^(6/12) - very dissonant
    (1.4983, "perfect 5th", 0.95),  # 2^(7/12) = 3:2 = 1.5
    (1.5874, "minor 6th", 0.6),  # 2^(8/12) = 8:5 = 1.6
    (1.6818, "major 6th", 0.7),  # 2^(9/12) = 5:3 ≈ 1.667
    (1.7818, "minor 7th", 0.4),  # 2^(10/12)
    (1.8877, "major 7th", 0.3),  # 2^(11/12)
    (2.0, "octave", 1.0),  # 2:1
    (2.5, "octave + M3", 0.75),  # 5:2
    (3.0, "octave + P5", 0.9),  # 3:1
    (4.0, "2 octaves", 0.95),  # 4:1
]

# Just the ratios for quick lookup, extended to cover more range
CONSONANT_RATIOS = np.array([r for r, _, _ in CONSONANT_INTERVALS])
CONSONANCE_WEIGHTS = np.array([w for _, _, w in CONSONANT_INTERVALS])


def ratio_consonance(ratio: float, tolerance: float = 0.05) -> float:
    """
    Score how consonant a frequency ratio is (0-1).

    Args:
        ratio: Frequency ratio (higher/lower), should be >= 1
        tolerance: How close to a pure interval counts as consonant

    Returns:
        Consonance score 0-1 (1 = perfectly consonant)
    """
    if ratio < 1:
        ratio = 1.0 / ratio  # Normalize to >= 1

    # Reduce to within 2 octaves for matching
    while ratio > 4.0:
        ratio /= 2.0

    # Find distance to nearest consonant ratio
    distances = np.abs(CONSONANT_RATIOS - ratio)
    nearest_idx = np.argmin(distances)
    min_distance = distances[nearest_idx]
    weight = CONSONANCE_WEIGHTS[nearest_idx]

    # Score based on distance, weighted by interval's inherent consonance
    # Gaussian falloff from target ratio
    score = weight * np.exp(-0.5 * (min_distance / tolerance) ** 2)

    return float(score)


def compute_instantaneous_consonance(
    freqs: np.ndarray, amps: np.ndarray, amp_threshold: float = 0.1
) -> tuple[float, float]:
    """
    Compute consonance for a single timestep, weighted by amplitude.

    Only considers voice pairs where BOTH voices are sounding (amp > threshold).

    Args:
        freqs: (num_voices,) array of frequencies in Hz
        amps: (num_voices,) array of amplitudes (0-1)
        amp_threshold: Minimum amplitude to count as "sounding"

    Returns:
        (consonance_score, sounding_weight)
        - consonance_score: 0-1, or 0 if no voices sounding
        - sounding_weight: how much sound is happening (for weighted averaging)
    """
    num_voices = len(freqs)
    if num_voices < 2:
        return 0.0, 0.0  # Need at least 2 voices for consonance

    # Which voices are actually sounding?
    sounding = amps > amp_threshold

    # Total "sound energy" for weighting
    sound_energy = np.sum(amps[sounding]) if np.any(sounding) else 0.0

    if np.sum(sounding) < 2:
        # Less than 2 voices sounding - can't measure consonance
        return 0.0, sound_energy

    scores = []
    weights = []
    for i in range(num_voices):
        for j in range(i + 1, num_voices):
            if sounding[i] and sounding[j]:
                ratio = max(freqs[i], freqs[j]) / min(freqs[i], freqs[j])
                score = ratio_consonance(ratio)
                # Weight by combined amplitude of the pair
                weight = amps[i] * amps[j]
                scores.append(score)
                weights.append(weight)

    if not scores:
        return 0.0, sound_energy

    # Weighted average of consonance scores
    weights = np.array(weights)
    consonance = np.average(scores, weights=weights)

    return float(consonance), float(sound_energy)


def compute_consonance_over_time(
    freq_history: np.ndarray,
    amp_history: np.ndarray,
    amp_threshold: float = 0.1,
) -> tuple[float, float]:
    """
    Compute consonance over the full simulation, weighted by amplitude.

    Only measures consonance when voices are actually sounding.
    Returns 0 if there's not enough sustained sound.

    Args:
        freq_history: (T, num_voices) frequency values
        amp_history: (T, num_voices) amplitude values
        amp_threshold: Minimum amplitude to count as "sounding"

    Returns:
        mean_consonance: Weighted average consonance (0 if mostly silent)
        sounding_fraction: What fraction of time had 2+ voices sounding
    """
    num_samples = freq_history.shape[0]

    consonance_scores = []
    consonance_weights = []

    for t in range(num_samples):
        cons, weight = compute_instantaneous_consonance(
            freq_history[t], amp_history[t], amp_threshold
        )
        if weight > 0:
            consonance_scores.append(cons)
            consonance_weights.append(weight)

    # How much of the time were we actually measuring consonance?
    sounding_fraction = len(consonance_scores) / num_samples if num_samples > 0 else 0.0

    if not consonance_scores:
        # Nothing was sounding - consonance is meaningless
        return 0.0, 0.0

    # Weighted average
    weights = np.array(consonance_weights)
    mean_consonance = float(np.average(consonance_scores, weights=weights))

    return mean_consonance, sounding_fraction


# =============================================================================
# Activity metrics
# =============================================================================


def compute_amplitude_activity(
    amp_history: np.ndarray, amp_threshold: float = 0.1
) -> tuple[float, float, float]:
    """
    Score based on amplitude activity - reward sustained sound, penalize silence.

    Args:
        amp_history: (T, num_voices) amplitude values (0-1)
        amp_threshold: Minimum amplitude to count as "sounding"

    Returns:
        (activity_score, mean_amplitude, sounding_fraction)
        - activity_score: 0-1, rewards sustained sound around target amplitude
        - mean_amplitude: raw mean amplitude
        - sounding_fraction: what fraction of (time, voice) pairs are sounding
    """
    num_samples, num_voices = amp_history.shape

    # What fraction of samples have at least 2 voices sounding?
    voices_sounding_per_step = np.sum(amp_history > amp_threshold, axis=1)
    steps_with_sound = np.sum(voices_sounding_per_step >= 2)
    sounding_fraction = steps_with_sound / num_samples if num_samples > 0 else 0.0

    # Mean amplitude when sounding
    sounding_mask = amp_history > amp_threshold
    if np.any(sounding_mask):
        mean_amp_when_sounding = np.mean(amp_history[sounding_mask])
    else:
        mean_amp_when_sounding = 0.0

    mean_amp = np.mean(amp_history)

    # Score components:
    # 1. Must have sustained sound (sounding_fraction > 0.5 to score well)
    # 2. When sounding, amplitude should be decent (~0.4-0.6)

    # Penalize silence harshly
    if sounding_fraction < 0.3:
        # Less than 30% of time with 2+ voices = very low score
        score = sounding_fraction * 0.5
    else:
        # Reward based on sounding fraction and amplitude quality
        # sounding_fraction component (want > 0.5)
        frac_score = min(1.0, sounding_fraction / 0.7)  # Saturates at 70%

        # amplitude quality (want ~0.5 when sounding)
        target_amp = 0.5
        amp_score = np.exp(-4 * (mean_amp_when_sounding - target_amp) ** 2)

        score = frac_score * amp_score

    return float(score), float(mean_amp), float(sounding_fraction)


def compute_frequency_stability(freq_history: np.ndarray) -> float:
    """
    Score frequency stability - reward relatively stable pitches over time.

    Too much frequency wobble sounds like noise/chaos.
    Too little means static/boring.

    Args:
        freq_history: (T, num_voices) frequency values

    Returns:
        Stability score 0-1
    """
    # Compute relative frequency changes (cents per sample)
    # cents = 1200 * log2(f2/f1)
    freq_history = np.clip(freq_history, 1, None)  # Avoid log(0)

    # Relative changes between adjacent samples
    ratios = freq_history[1:] / freq_history[:-1]
    cents_changes = 1200 * np.log2(ratios + 1e-10)

    # RMS of changes (how "wobbly" is the frequency?)
    rms_wobble = np.sqrt(np.mean(cents_changes**2))

    # Target: small wobble (maybe 10-50 cents per sample is interesting)
    # Penalize both too stable (0 wobble) and too chaotic (>100 cents)
    # Peak at ~20 cents wobble
    target_wobble = 20
    max_wobble = 200

    if rms_wobble > max_wobble:
        score = 0.0
    else:
        # Gaussian peaked at target
        score = np.exp(-0.5 * ((rms_wobble - target_wobble) / 30) ** 2)

    return float(score)


def compute_voice_independence(freq_history: np.ndarray) -> float:
    """
    Score voice independence - penalize if all voices move together.

    We want polyphonic interest, not unison movement.

    Args:
        freq_history: (T, num_voices) frequency values

    Returns:
        Independence score 0-1
    """
    num_voices = freq_history.shape[1]
    if num_voices < 2:
        return 1.0

    # Compute pairwise correlation of frequency trajectories
    correlations = []
    for i in range(num_voices):
        for j in range(i + 1, num_voices):
            # Check for zero variance (constant signal)
            std_i = np.std(freq_history[:, i])
            std_j = np.std(freq_history[:, j])
            if std_i < 1e-6 or std_j < 1e-6:
                # Constant signals: if same value, perfectly correlated; else uncorrelated
                if std_i < 1e-6 and std_j < 1e-6:
                    # Both constant - check if same value
                    if abs(np.mean(freq_history[:, i]) - np.mean(freq_history[:, j])) < 1e-6:
                        correlations.append(1.0)  # Same constant = correlated
                    else:
                        correlations.append(0.0)  # Different constants = independent
                continue

            corr = np.corrcoef(freq_history[:, i], freq_history[:, j])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

    if not correlations:
        return 1.0

    mean_corr = np.mean(correlations)

    # Score: 1 when uncorrelated (mean_corr=0), 0 when perfectly correlated (mean_corr=1)
    # But allow some correlation (unison/octave movement is fine)
    # Penalize only very high correlation (>0.9)
    if mean_corr < 0.5:
        score = 1.0
    else:
        score = 1.0 - (mean_corr - 0.5) * 2  # Linear decay from 0.5 to 1.0

    return float(max(0, score))


# =============================================================================
# Main evaluation
# =============================================================================


@dataclass
class AudioMetrics:
    """Container for audio evaluation metrics."""

    consonance: float  # 0-1, how consonant the frequency ratios are (when sounding)
    activity: float  # 0-1, sustained sound level
    stability: float  # 0-1, frequency stability (not too wobbly)
    independence: float  # 0-1, voice independence (not all moving together)
    composite_score: float  # Combined fitness

    # Diagnostic metrics
    sounding_fraction: float = 0.0  # What fraction of time 2+ voices were sounding
    mean_amplitude: float = 0.0  # Raw mean amplitude

    def __str__(self) -> str:
        return (
            f"AudioMetrics(\n"
            f"  composite: {self.composite_score:.3f}\n"
            f"  consonance: {self.consonance:.3f} (sounding {self.sounding_fraction:.1%} of time)\n"
            f"  activity: {self.activity:.3f} (mean_amp={self.mean_amplitude:.3f})\n"
            f"  stability: {self.stability:.3f}\n"
            f"  independence: {self.independence:.3f}\n"
            f")"
        )


class AudioAnalyzer:
    """
    Analyzer for audio outputs from neural network.

    Evaluates consonance, activity, and other musical qualities.
    """

    def __init__(
        self,
        freq_min: float = DEFAULT_FREQ_MIN,
        freq_max: float = DEFAULT_FREQ_MAX,
        consonance_weight: float = 0.5,
        activity_weight: float = 0.2,
        stability_weight: float = 0.15,
        independence_weight: float = 0.15,
    ):
        """
        Args:
            freq_min: Minimum frequency mapping
            freq_max: Maximum frequency mapping
            consonance_weight: Weight for consonance in composite score
            activity_weight: Weight for activity in composite score
            stability_weight: Weight for stability in composite score
            independence_weight: Weight for independence in composite score
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.consonance_weight = consonance_weight
        self.activity_weight = activity_weight
        self.stability_weight = stability_weight
        self.independence_weight = independence_weight

    def analyze(
        self,
        output_history: np.ndarray,
        amp_smoothing: float = 0.999,
    ) -> AudioMetrics:
        """
        Analyze network outputs and return metrics.

        Args:
            output_history: (T, num_voices, 3) where 3 = [freq, phase_mod, amp]
            amp_smoothing: Leaky integrator decay for amplitude (must match synthesis)

        Returns:
            AudioMetrics with scores
        """
        num_samples, num_voices, num_outputs = output_history.shape

        if num_samples == 0 or num_outputs != 3:
            return AudioMetrics(
                consonance=0.0,
                activity=0.0,
                stability=0.0,
                independence=0.0,
                composite_score=0.0,
                sounding_fraction=0.0,
                mean_amplitude=0.0,
            )

        # Extract parameters
        freq_raw = output_history[:, :, 0]  # (T, V)
        amp_raw = output_history[:, :, 2]  # (T, V)

        # Apply same smoothing as synthesis
        amp = np.zeros_like(amp_raw)
        amp[0] = amp_raw[0]
        for t in range(1, num_samples):
            amp[t] = amp_smoothing * amp[t - 1] + (1 - amp_smoothing) * amp_raw[t]

        # Map to actual frequencies
        freqs = map_output_to_freq(freq_raw, self.freq_min, self.freq_max)

        # Compute metrics (amplitude-aware)
        consonance, sounding_fraction = compute_consonance_over_time(freqs, amp)
        activity, mean_amplitude, _ = compute_amplitude_activity(amp)
        stability = compute_frequency_stability(freqs)
        independence = compute_voice_independence(freqs)

        # Composite score: consonance * activity
        composite = consonance * activity

        return AudioMetrics(
            consonance=consonance,
            activity=activity,
            stability=stability,
            independence=independence,
            composite_score=composite,
            sounding_fraction=sounding_fraction,
            mean_amplitude=mean_amplitude,
        )


def evaluate_audio(
    output_history: np.ndarray,
    **kwargs,
) -> AudioMetrics:
    """Quick evaluation of network outputs (oscillator mode)."""
    analyzer = AudioAnalyzer(**kwargs)
    return analyzer.analyze(output_history)


def compute_pitchiness(audio: np.ndarray, min_lag: int = 20, max_lag: int = 500) -> float:
    """
    Measure how tonal/pitched the audio is using autocorrelation.
    
    Periodic signals have strong autocorrelation peaks at the period.
    Noise has weak autocorrelation.
    
    Args:
        audio: (T,) audio samples
        min_lag: Minimum lag to search (excludes very high frequencies)
        max_lag: Maximum lag to search (excludes very low frequencies)
        
    Returns:
        Pitchiness score 0-1 (1 = strongly periodic/tonal)
    """
    if len(audio) < max_lag * 2:
        max_lag = len(audio) // 2
    
    if max_lag <= min_lag:
        return 0.0
    
    # Normalize audio
    audio = audio - np.mean(audio)
    if np.std(audio) < 1e-6:
        return 0.0
    audio = audio / np.std(audio)
    
    # Compute autocorrelation for lags in range
    n = len(audio)
    autocorr = np.correlate(audio, audio, mode='full')[n-1:]  # Positive lags only
    
    # Normalize by zero-lag (which equals variance = 1 after normalization)
    autocorr = autocorr / autocorr[0]
    
    # Find peak in the valid lag range
    valid_autocorr = autocorr[min_lag:max_lag]
    if len(valid_autocorr) == 0:
        return 0.0
    
    peak = np.max(valid_autocorr)
    
    # Peak > 0.5 is quite tonal, > 0.8 is very tonal
    # Clamp to [0, 1]
    return float(np.clip(peak, 0, 1))


def evaluate_raw_audio(
    output_history: np.ndarray,
    stretch_factor: int = 1,
) -> tuple[float, float, float]:
    """
    Evaluate raw multiply synthesis for tonality and activity.
    
    Args:
        output_history: (T, num_voices, 3) network outputs
        stretch_factor: Interpolation factor for audio
        
    Returns:
        (fitness, pitchiness, activity)
    """
    from utils_audio import synthesize_raw_multiply
    
    audio = synthesize_raw_multiply(output_history, stretch_factor)
    
    # Activity: RMS amplitude
    rms = np.sqrt(np.mean(audio ** 2))
    # Target RMS around 0.2, saturate at 0.4
    activity = min(1.0, rms / 0.2)
    
    # Pitchiness: autocorrelation
    pitchiness = compute_pitchiness(audio)
    
    # Fitness: product
    fitness = pitchiness * activity
    
    return fitness, pitchiness, activity


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test with synthetic examples

    print("=" * 60)
    print("Testing consonance scoring")
    print("=" * 60)

    # Test individual ratios
    test_ratios = [1.0, 1.25, 1.333, 1.5, 1.414, 2.0, 3.0]
    for ratio in test_ratios:
        score = ratio_consonance(ratio)
        print(f"  Ratio {ratio:.3f}: consonance = {score:.3f}")

    print("\n" + "=" * 60)
    print("Testing full evaluation")
    print("=" * 60)

    # Create test outputs
    np.random.seed(42)
    num_samples = DEFAULT_SAMPLE_RATE
    num_voices = 3

    # Test 1: Perfectly consonant (voices at octaves)
    print("\nTest 1: Perfect octaves (should be highly consonant)")
    output_consonant = np.zeros((num_samples, num_voices, 3))
    # Voice 1: fixed low frequency (output=0.3 -> ~200 Hz)
    output_consonant[:, 0, 0] = 0.3
    output_consonant[:, 0, 2] = 0.5  # amplitude
    # Voice 2: octave above (output=0.65 -> ~400 Hz with exp mapping)
    output_consonant[:, 1, 0] = 0.65
    output_consonant[:, 1, 2] = 0.5
    # Voice 3: another octave (output=1.0 -> ~800 Hz)
    output_consonant[:, 2, 0] = 1.0
    output_consonant[:, 2, 2] = 0.5

    metrics = evaluate_audio(output_consonant)
    print(metrics)

    # Test 2: Random noise (should be less consonant)
    print("\nTest 2: Random noise (should be dissonant)")
    output_random = np.random.rand(num_samples, num_voices, 3)
    metrics = evaluate_audio(output_random)
    print(metrics)

    # Test 3: Silent (should score poorly on activity)
    print("\nTest 3: Silent (should have low activity)")
    output_silent = np.zeros((num_samples, num_voices, 3))
    output_silent[:, :, 0] = 0.5  # Some frequency
    output_silent[:, :, 2] = 0.0  # Zero amplitude
    metrics = evaluate_audio(output_silent)
    print(metrics)

