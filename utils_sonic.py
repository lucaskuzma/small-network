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


def save_argmax_outputs_as_midi(
    output_history: np.ndarray,
    filename: str = "argmax_output.mid",
    tempo: int = 120,
    min_threshold: float = 0.3,
    base_notes: list = None,
):
    """
    Save network outputs as MIDI using argmax selection per voice.
    
    Only one pitch per voice per timestep: the one with highest output value.
    Note only triggers if that highest value exceeds min_threshold.
    
    Args:
        output_history: numpy array of shape (T, num_voices, n_pitches)
        filename: output MIDI filename
        tempo: beats per minute
        min_threshold: minimum output value for argmax winner to trigger note
        base_notes: base MIDI note per voice (default: [48, 60, 72, 84])
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage

    # Parse output_history shape
    if len(output_history.shape) != 3:
        raise ValueError(
            f"output_history must be 3D (steps, voices, pitches), got shape {output_history.shape}"
        )

    num_steps, num_voices, n_pitches = output_history.shape

    # Default base notes
    if base_notes is None:
        base_notes = [48, 60, 72, 84]  # C3, C4, C5, C6

    # Ensure enough base notes
    while len(base_notes) < num_voices:
        base_notes.append(base_notes[-1] + 12)

    # Create MIDI file
    mid = MidiFile()
    ticks_per_beat = 480
    mid.ticks_per_beat = ticks_per_beat
    ticks_per_16th = ticks_per_beat // 4

    for voice_idx in range(num_voices):
        track = MidiTrack()
        mid.tracks.append(track)

        # Metadata
        track.append(MetaMessage("track_name", name=f"Voice {voice_idx + 1}", time=0))
        if voice_idx == 0:
            microseconds_per_beat = int(60_000_000 / tempo)
            track.append(MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))

        base_note = base_notes[voice_idx]
        current_note = None  # Currently playing note
        current_absolute_time = 0

        for step_idx in range(num_steps):
            step_time = step_idx * ticks_per_16th
            outputs = output_history[step_idx, voice_idx, :]

            # Argmax: find highest output
            best_idx = np.argmax(outputs)
            best_val = outputs[best_idx]

            # Determine target note (or None if below threshold)
            target_note = None
            if best_val > min_threshold:
                target_note = base_note + best_idx

            # Handle note transitions
            if target_note != current_note:
                # End current note if any
                if current_note is not None:
                    delta = step_time - current_absolute_time
                    track.append(
                        Message(
                            "note_off",
                            note=current_note,
                            velocity=0,
                            time=delta,
                            channel=voice_idx % 16,
                        )
                    )
                    current_absolute_time = step_time

                # Start new note if any
                if target_note is not None:
                    delta = step_time - current_absolute_time
                    velocity = int(np.clip(best_val * 127, 1, 127))
                    track.append(
                        Message(
                            "note_on",
                            note=target_note,
                            velocity=velocity,
                            time=delta,
                            channel=voice_idx % 16,
                        )
                    )
                    current_absolute_time = step_time

                current_note = target_note

        # End any remaining note
        if current_note is not None:
            final_time = num_steps * ticks_per_16th
            delta = final_time - current_absolute_time
            track.append(
                Message(
                    "note_off",
                    note=current_note,
                    velocity=0,
                    time=delta,
                    channel=voice_idx % 16,
                )
            )
            current_absolute_time = final_time

        track.append(MetaMessage("end_of_track", time=0))

    mid.save(filename)
    print(f"MIDI file saved (argmax): {filename}")
    print(f"  Tempo: {tempo} BPM, Voices: {num_voices}, Pitches: {n_pitches}")
    print(f"  Min threshold: {min_threshold}")

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

    Output mapping per voice (7 outputs):
      [u1, u4, u7, d1, d3, d8, vel]

      Motion = (u1×1 + u4×4 + u7×7) - (d1×1 + d3×3 + d8×8)
      Range: -12 to +12 semitones

      Velocity = output[6] directly (single gate, simpler than XOR)
      Note triggered when velocity > threshold.

    Behavior:
      - All voices start at unison (C), each in own octave
      - No note triggered until first non-zero motion
      - Pitch wraps modulo 12 within each voice's octave
      - Sustain: note continues when velocity high but motion = 0

    Args:
        output_history: numpy array of shape (T, num_voices, 7)
        filename: output MIDI filename
        tempo: beats per minute
        velocity_threshold: minimum velocity to trigger/sustain a note
        start_pitches: starting MIDI note per voice (default: [48, 60, 72, 84] = C3-C6)
        velocity_range: (min, max) MIDI velocity range (default: 70-100 for narrow dynamics)
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage

    # Parse output_history shape
    if len(output_history.shape) != 3 or output_history.shape[2] != 7:
        raise ValueError(
            f"output_history must be 3D with 7 outputs per voice (T, voices, 7), "
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

            # Compute velocity from output 6 (single gate)
            vel_raw = outputs[6]
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
    print(f"  Encoding: motion [u1,u4,u7,d1,d3,d8,vel]")
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


# =======================================================================
# Piano Roll Visualization
# =======================================================================


def save_piano_roll_png(
    midi_path: str,
    png_path: str = None,
    figsize: tuple = (14, 4),
    pitch_range: tuple = (36, 96),
    duration_beats: float = 32.0,
    tempo: int = 120,
):
    """
    Save a piano roll visualization of a MIDI file.
    
    Fixed extents for animation compatibility:
    - X-axis: 0 to duration_beats (in beats)
    - Y-axis: pitch_range[0] to pitch_range[1] (MIDI note numbers)
    
    Args:
        midi_path: Path to input MIDI file
        png_path: Path for output PNG (default: same as midi_path with .png extension)
        figsize: Figure size in inches (width, height)
        pitch_range: (min_pitch, max_pitch) for Y-axis (default: 36-96, C2-C7)
        duration_beats: X-axis extent in beats (default: 32 beats)
        tempo: Tempo in BPM for time conversion (default: 120)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from mido import MidiFile
    
    if png_path is None:
        png_path = midi_path.rsplit(".", 1)[0] + ".png"
    
    # Load MIDI and extract notes
    mid = MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    
    notes = []
    for track_idx, track in enumerate(mid.tracks):
        current_tick = 0
        active_notes = {}  # (pitch, channel) -> start_tick
        
        for msg in track:
            current_tick += msg.time
            
            if msg.type == "note_on" and msg.velocity > 0:
                key = (msg.note, msg.channel)
                active_notes[key] = (current_tick, msg.velocity)
            
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in active_notes:
                    start_tick, velocity = active_notes[key]
                    notes.append({
                        "pitch": msg.note,
                        "start_beat": start_tick / ticks_per_beat,
                        "end_beat": current_tick / ticks_per_beat,
                        "velocity": velocity,
                        "track": track_idx,
                    })
                    del active_notes[key]
    
    # Create figure with fixed size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by track (voice)
    track_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    
    # Draw notes as rectangles
    for note in notes:
        color = track_colors[note["track"] % len(track_colors)]
        # Velocity -> alpha (0.4 to 1.0)
        alpha = 0.4 + 0.6 * (note["velocity"] / 127)
        
        rect = patches.Rectangle(
            (note["start_beat"], note["pitch"] - 0.4),
            note["end_beat"] - note["start_beat"],
            0.8,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=alpha,
        )
        ax.add_patch(rect)
    
    # Fixed axes for animation
    ax.set_xlim(0, duration_beats)
    ax.set_ylim(pitch_range[0], pitch_range[1])
    
    # Grid lines on octaves (C notes)
    octave_notes = [n for n in range(pitch_range[0], pitch_range[1] + 1) if n % 12 == 0]
    ax.set_yticks(octave_notes)
    ax.set_yticklabels([f"C{n // 12 - 1}" for n in octave_notes])
    ax.yaxis.grid(True, alpha=0.3)
    
    # Beat grid
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_xticks(range(0, int(duration_beats) + 1, 4))
    
    ax.set_xlabel("Beats")
    ax.set_ylabel("Pitch")
    ax.set_facecolor("#f8f9fa")
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(png_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return png_path
