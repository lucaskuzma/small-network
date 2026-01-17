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


def save_readout_outputs_as_midi(
    output_history,
    filename="neural_output.mid",
    tempo=120,
    threshold=0.5,
    base_notes=None,
):
    """
    Save neural network readout outputs as MIDI file.
    Each readout (voice) on a separate track, chromatic pitch classes (0-11).
    
    Args:
        output_history: numpy array of shape (T, num_readouts, 12) or (T, num_readouts, n_pitches)
        filename: output MIDI filename
        tempo: beats per minute
        threshold: single float or array of thresholds (one per readout/voice)
        base_notes: list of base MIDI notes for each voice (default: [48, 60, 72, 84] = C3, C4, C5, C6)
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage

    # Parse output_history shape
    if len(output_history.shape) == 3:
        num_steps, num_readouts, n_pitches = output_history.shape
    else:
        raise ValueError(
            f"output_history must be 3D (steps, readouts, pitches), got shape {output_history.shape}"
        )

    # Default base notes - each voice gets its own octave
    if base_notes is None:
        base_notes = [48, 60, 72, 84]  # C3, C4, C5, C6
    
    # Ensure we have enough base notes
    while len(base_notes) < num_readouts:
        base_notes.append(base_notes[-1] + 12)

    # Create MIDI file
    mid = MidiFile()
    ticks_per_beat = 480
    mid.ticks_per_beat = ticks_per_beat

    # Calculate ticks per 16th note
    ticks_per_16th = ticks_per_beat // 4

    # Convert threshold to array if needed
    threshold_array = np.atleast_1d(threshold)

    # Create a track for each readout (voice)
    for readout_idx in range(num_readouts):
        track = MidiTrack()
        mid.tracks.append(track)

        # Add metadata
        track.append(
            MetaMessage("track_name", name=f"Voice {readout_idx + 1}", time=0)
        )
        if readout_idx == 0:
            # Only first track gets tempo
            microseconds_per_beat = int(60_000_000 / tempo)
            track.append(MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))

        # Get threshold for this readout
        thresh = (
            threshold_array[readout_idx]
            if readout_idx < len(threshold_array)
            else threshold_array[0]
        )

        # Get base note for this voice
        base_note = base_notes[readout_idx]

        # Track active notes for each pitch class in this voice
        active_notes = {}  # {pitch_class: note}
        current_absolute_time = 0

        # Process each timestep
        for step_idx in range(num_steps):
            step_time = step_idx * ticks_per_16th
            outputs = output_history[step_idx, readout_idx, :]

            for pitch_class, output_value in enumerate(outputs):
                # MIDI note = base note + chromatic offset
                note = base_note + pitch_class

                # MIDI velocity based on output value
                velocity = int(np.clip(output_value * 127, 1, 127))

                # Check if note should be active
                is_active = output_value > thresh

                # Handle note on/off
                if is_active and pitch_class not in active_notes:
                    # Start new note
                    delta = step_time - current_absolute_time
                    track.append(
                        Message(
                            "note_on",
                            note=note,
                            velocity=velocity,
                            time=delta,
                            channel=readout_idx,
                        )
                    )
                    current_absolute_time = step_time
                    active_notes[pitch_class] = note

                elif not is_active and pitch_class in active_notes:
                    # End active note
                    note_to_end = active_notes[pitch_class]
                    delta = step_time - current_absolute_time
                    track.append(
                        Message(
                            "note_off",
                            note=note_to_end,
                            velocity=0,
                            time=delta,
                            channel=readout_idx,
                        )
                    )
                    current_absolute_time = step_time
                    del active_notes[pitch_class]

        # End any remaining active notes
        final_time = num_steps * ticks_per_16th
        for pitch_class, note in active_notes.items():
            delta = final_time - current_absolute_time
            track.append(
                Message(
                    "note_off",
                    note=note,
                    velocity=0,
                    time=delta,
                    channel=readout_idx,
                )
            )
            current_absolute_time = final_time

        # Add end of track message
        track.append(MetaMessage("end_of_track", time=0))

    # Save MIDI file
    mid.save(filename)
    print(f"MIDI file saved to: {filename}")
    print(f"  Tempo: {tempo} BPM")
    print(f"  Duration: {num_steps} steps ({num_steps * ticks_per_16th} ticks)")
    print(f"  Voices: {num_readouts}")
    print(f"  Pitch classes per voice: {n_pitches} (chromatic)")
    print(f"  Base notes: {base_notes[:num_readouts]}")
    print(f"  Resolution: 16th notes ({ticks_per_16th} ticks per 16th)")

    return filename


# =======================================================================
# Motion-based MIDI export
# =======================================================================


def save_motion_outputs_as_midi(
    output_history: np.ndarray,
    filename: str = "motion_output.mid",
    tempo: int = 120,
    velocity_threshold: float = 0.3,
    start_pitches: list = None,
    velocity_range: tuple = (70, 100),
):
    """
    Save network outputs as MIDI using motion encoding.

    Output mapping per voice (8 outputs):
      [u1, u4, u7, d1, d3, d8, v1, v2]

      Motion = (u1×1 + u4×4 + u7×7) - (d1×1 + d3×3 + d8×8)
      Range: -12 to +12 semitones

      Velocity = |v1 - v2| (soft XOR, high when they differ)
      Note triggered when velocity > threshold.

    Behavior:
      - All voices start at unison (C), each in own octave
      - No note triggered until first non-zero motion
      - Pitch wraps modulo 12 within each voice's octave
      - Sustain: note continues when velocity high but motion = 0

    Args:
        output_history: numpy array of shape (T, num_voices, 8)
        filename: output MIDI filename
        tempo: beats per minute
        velocity_threshold: minimum |v1-v2| to trigger/sustain a note
        start_pitches: starting MIDI note per voice (default: [48, 60, 72, 84] = C3-C6)
        velocity_range: (min, max) MIDI velocity range (default: 70-100 for narrow dynamics)
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage

    # Parse output_history shape
    if len(output_history.shape) != 3 or output_history.shape[2] != 8:
        raise ValueError(
            f"output_history must be 3D with 8 outputs per voice (T, voices, 8), "
            f"got shape {output_history.shape}"
        )

    num_steps, num_voices, _ = output_history.shape

    # Default start pitches - each voice gets its own octave, all start at C
    if start_pitches is None:
        start_pitches = [48, 60, 72, 84]  # C3, C4, C5, C6

    # Ensure we have enough start pitches
    while len(start_pitches) < num_voices:
        start_pitches.append(start_pitches[-1] + 12)

    # Motion weights: up = [1, 4, 7], down = [1, 3, 8]
    up_weights = np.array([1, 4, 7])
    down_weights = np.array([1, 3, 8])

    # Convert velocity_threshold to per-voice array if scalar
    vel_thresholds = np.atleast_1d(velocity_threshold)
    if len(vel_thresholds) == 1:
        vel_thresholds = np.full(num_voices, vel_thresholds[0])

    # Create MIDI file
    mid = MidiFile()
    ticks_per_beat = 480
    mid.ticks_per_beat = ticks_per_beat

    # Calculate ticks per 16th note
    ticks_per_16th = ticks_per_beat // 4

    # Process each voice on separate track
    for voice_idx in range(num_voices):
        track = MidiTrack()
        mid.tracks.append(track)

        # Add metadata
        track.append(MetaMessage("track_name", name=f"Voice {voice_idx + 1}", time=0))
        if voice_idx == 0:
            microseconds_per_beat = int(60_000_000 / tempo)
            track.append(MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))

        # Voice state
        base_pitch = start_pitches[voice_idx]  # Base of this voice's octave
        pitch_class = 0  # Start at C (0) within the octave
        current_note = None  # Currently playing MIDI note (or None)
        has_moved = False  # Track if voice has ever moved (no note until first motion)
        current_absolute_time = 0

        for step_idx in range(num_steps):
            step_time = step_idx * ticks_per_16th
            outputs = output_history[step_idx, voice_idx, :]

            # Compute motion from outputs 0-5
            up_bits = (outputs[0:3] > 0.5).astype(int)
            down_bits = (outputs[3:6] > 0.5).astype(int)
            up_sum = np.dot(up_bits, up_weights)
            down_sum = np.dot(down_bits, down_weights)
            motion = up_sum - down_sum

            # Compute velocity from outputs 6-7 (soft XOR: high when they differ)
            vel_raw = np.abs(outputs[6] - outputs[7])
            voice_threshold = vel_thresholds[voice_idx]
            vel_active = vel_raw > voice_threshold

            # Apply motion (wrap modulo 12 within octave)
            if motion != 0:
                has_moved = True
                pitch_class = (pitch_class + motion) % 12
                # Handle negative modulo correctly
                if pitch_class < 0:
                    pitch_class += 12

            # Determine target note
            target_note = base_pitch + pitch_class

            # Compute MIDI velocity within range from amount above threshold
            vel_min, vel_max = velocity_range
            if vel_active:
                # Map (threshold, 1.0) -> (vel_min, vel_max)
                vel_normalized = (vel_raw - voice_threshold) / (1 - voice_threshold + 1e-6)
                midi_velocity = int(vel_min + vel_normalized * (vel_max - vel_min))
                midi_velocity = np.clip(midi_velocity, vel_min, vel_max)
            else:
                midi_velocity = 0

            # Note logic
            if vel_active and has_moved:
                if current_note is None:
                    # Start new note
                    delta = step_time - current_absolute_time
                    track.append(
                        Message("note_on", note=target_note, velocity=midi_velocity,
                                time=delta, channel=voice_idx)
                    )
                    current_absolute_time = step_time
                    current_note = target_note

                elif motion != 0 and target_note != current_note:
                    # Pitch changed - end old note, start new
                    delta = step_time - current_absolute_time
                    track.append(
                        Message("note_off", note=current_note, velocity=0,
                                time=delta, channel=voice_idx)
                    )
                    track.append(
                        Message("note_on", note=target_note, velocity=midi_velocity,
                                time=0, channel=voice_idx)
                    )
                    current_absolute_time = step_time
                    current_note = target_note
                # else: sustain (motion=0 or same pitch) - do nothing

            elif not vel_active and current_note is not None:
                # Velocity dropped below threshold - end note
                delta = step_time - current_absolute_time
                track.append(
                    Message("note_off", note=current_note, velocity=0,
                            time=delta, channel=voice_idx)
                )
                current_absolute_time = step_time
                current_note = None

        # End any remaining note
        if current_note is not None:
            final_time = num_steps * ticks_per_16th
            delta = final_time - current_absolute_time
            track.append(
                Message("note_off", note=current_note, velocity=0,
                        time=delta, channel=voice_idx)
            )
            current_absolute_time = final_time

        track.append(MetaMessage("end_of_track", time=0))

    # Save MIDI file
    mid.save(filename)
    print(f"MIDI file saved to: {filename}")
    print(f"  Tempo: {tempo} BPM")
    print(f"  Duration: {num_steps} steps ({num_steps * ticks_per_16th} ticks)")
    print(f"  Voices: {num_voices}")
    print(f"  Encoding: motion [u1,u4,u7,d1,d3,d8,v1,v2]")
    print(f"  Start pitches: {start_pitches[:num_voices]}")
    print(f"  Velocity thresholds: {[f'{t:.3f}' for t in vel_thresholds]}")
    print(f"  Velocity range: {velocity_range[0]}-{velocity_range[1]}")

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

