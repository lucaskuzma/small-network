# %%
# Sonification and MIDI utilities

import numpy as np
import time


# =======================================================================
# Threshold helpers
# =======================================================================


def find_threshold(history, percentile=0.8, key="outputs", per_channel=True):
    """
    Find threshold value(s) that keep the top percentile of the output history.

    Args:
        history: dict containing time series data
        percentile: percentile threshold (0-1)
        key: which data to use from history
        per_channel: if True, compute one threshold per channel; if False, compute global threshold

    Returns:
        array of thresholds (one per channel) if per_channel=True, else single float
    """
    data = np.array(history[key])  # (T, D) shape

    if per_channel:
        # Compute threshold for each channel independently
        thresholds = np.percentile(data, percentile * 100, axis=0)
        return thresholds
    else:
        # Global threshold across all data
        threshold = np.percentile(data, percentile * 100)
    return threshold


# =======================================================================
# Live audio playback
# =======================================================================


def play_neural_outputs_live(history, tempo=120, threshold=0.5, key="outputs"):
    """
    Play neural network data as audio in real-time.

    Args:
        history: dict containing time series data
        tempo: beats per minute
        threshold: single float or array of thresholds (one per channel)
        key: which data to use from history
    """
    import pygame

    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    pygame.init()

    # Map each neuron to a different frequency
    base_freq = 220  # A3
    freq_ratio = 2 ** (1 / 12)  # Semitone ratio

    # try with phrygian dominant scale
    phrygian_dominant_scale = [0, 1, 4, 5, 7, 8, 10]
    phrygian_dominant_scale_freqs = [
        base_freq * (freq_ratio**i) for i in phrygian_dominant_scale
    ]

    # or major scale
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    major_scale_freqs = [base_freq * (freq_ratio**i) for i in major_scale]

    # Calculate time per step
    step_duration = 60.0 / tempo / 4  # Assuming 16th notes

    # Convert threshold to array if needed
    threshold_array = np.atleast_1d(threshold)

    for step, outputs in enumerate(history[key]):
        # Mix all active neurons for this time step
        mixed_audio = np.zeros(int(44100 * step_duration))

        for index, output_value in enumerate(outputs):
            # Get threshold for this channel (or use single threshold for all)
            thresh = (
                threshold_array[index]
                if index < len(threshold_array)
                else threshold_array[0]
            )

            if output_value > thresh:  # Threshold for activation
                # Map neuron index to frequency (each neuron gets a different note)
                # frequency = base_freq * (freq_ratio ** ((index * 5) % 36))
                frequency = major_scale_freqs[index % len(major_scale)]

                # Generate sine wave for this neuron
                t = np.linspace(0, step_duration, len(mixed_audio))
                wave = np.sin(2 * np.pi * frequency * t) * output_value

                # fade out over duration
                wave *= np.linspace(1, 0, len(t))

                # Add to mixed audio
                mixed_audio += wave

        # Normalize and convert to 16-bit audio
        if np.max(np.abs(mixed_audio)) > 0:
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.8
        audio = (mixed_audio * 32767).astype(np.int16)

        # Play the mixed audio for this time step
        sound = pygame.sndarray.make_sound(audio)
        sound.play()

        time.sleep(step_duration)


# =======================================================================
# MIDI export
# =======================================================================


def save_neural_outputs_as_midi(
    history,
    filename="neural_output.mid",
    tempo=120,
    threshold=0.5,
    key="clusters",
):
    """
    Save neural network outputs as MIDI file.
    All clusters on one channel, differentiated by pitch.
    Uses 16th notes for timing, following the mapping from play_neural_outputs_live.

    Args:
        history: dict containing time series data
        filename: output MIDI filename
        tempo: beats per minute
        threshold: single float or array of thresholds (one per cluster)
        key: which data to use from history
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage

    # Create MIDI file with timing resolution
    mid = MidiFile()
    ticks_per_beat = 480  # Standard MIDI resolution
    mid.ticks_per_beat = ticks_per_beat

    # Create single track
    track = MidiTrack()
    mid.tracks.append(track)

    # Add metadata
    track.append(MetaMessage("track_name", name="Neural Output", time=0))
    microseconds_per_beat = int(60_000_000 / tempo)
    track.append(MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))

    # Map to major scale (same as play_neural_outputs_live)
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    base_note = 57  # A3 in MIDI

    # Calculate ticks per 16th note
    ticks_per_16th = ticks_per_beat // 4

    # Convert threshold to array if needed
    threshold_array = np.atleast_1d(threshold)

    # Get data
    data = np.array(history[key])  # (T, num_clusters)
    num_clusters = data.shape[1]

    # Track active notes for each cluster
    active_notes = {}  # {cluster_idx: (note, start_time)}

    # Track absolute time for delta calculation
    current_absolute_time = 0

    # Process each timestep
    for step_idx, outputs in enumerate(data):
        step_time = step_idx * ticks_per_16th

        for cluster_idx, output_value in enumerate(outputs):
            # Get threshold for this cluster
            thresh = (
                threshold_array[cluster_idx]
                if cluster_idx < len(threshold_array)
                else threshold_array[0]
            )

            # Map cluster index to MIDI note (same mapping as play_neural_outputs_live)
            note = base_note + major_scale[cluster_idx % len(major_scale)]

            # MIDI velocity based on output value (scaled to 1-127)
            velocity = int(np.clip(output_value * 127, 1, 127))

            # Check if note should be active
            is_active = output_value > thresh

            # Handle note on/off
            if is_active and cluster_idx not in active_notes:
                # Start new note
                delta = step_time - current_absolute_time
                track.append(
                    Message(
                        "note_on", note=note, velocity=velocity, time=delta, channel=0
                    )
                )
                current_absolute_time = step_time
                active_notes[cluster_idx] = note

            elif not is_active and cluster_idx in active_notes:
                # End active note
                note_to_end = active_notes[cluster_idx]
                delta = step_time - current_absolute_time
                track.append(
                    Message(
                        "note_off", note=note_to_end, velocity=0, time=delta, channel=0
                    )
                )
                current_absolute_time = step_time
                del active_notes[cluster_idx]

    # End any remaining active notes
    final_time = len(data) * ticks_per_16th
    for cluster_idx, note in active_notes.items():
        delta = final_time - current_absolute_time
        track.append(Message("note_off", note=note, velocity=0, time=delta, channel=0))
        current_absolute_time = final_time

    # Add end of track message
    track.append(MetaMessage("end_of_track", time=0))

    # Save MIDI file
    mid.save(filename)
    print(f"MIDI file saved to: {filename}")
    print(f"  Tempo: {tempo} BPM")
    print(f"  Duration: {len(data)} steps ({len(data) * ticks_per_16th} ticks)")
    print(f"  Clusters: {num_clusters} (all on channel 0)")
    print(f"  Resolution: 16th notes ({ticks_per_16th} ticks per 16th)")

    return filename


# =======================================================================
# MIDI playback
# =======================================================================


def play_midi(filename):
    """
    Play MIDI file using pygame.

    Args:
        filename: path to MIDI file
    """
    import pygame

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    print(f"Playing: {filename}")
    print("Press Ctrl+C to stop...")

    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

