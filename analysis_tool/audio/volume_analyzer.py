import numpy as np
import wave
from analysis_tool.params import PROJECT_ROOT

# Function to convert amplitude to decibels
def amplitude_to_db(amplitude):
    # Using a reference value for dB calculation (1 for linear scale)
    ref = 1.0
    return 20 * np.log10(np.abs(amplitude) / ref)


def get_max_min_volumes(file_name: str):
    # Open the audio file
    with wave.open(f"{PROJECT_ROOT}/audio_files/{file_name}_speech.wav", "rb") as wav_file:
        # Extract basic information about the audio file
        n_channels = wav_file.getnchannels()  # Number of channels (1 for mono, 2 for stereo)
        sample_width = wav_file.getsampwidth()  # Number of bytes per sample
        frame_rate = wav_file.getframerate()  # Sampling frequency
        n_frames = wav_file.getnframes()  # Total number of audio frames
        
        # Read frames and convert to integer data
        frames = wav_file.readframes(n_frames)
        
        # Convert audio frames to numpy array based on sample width
        if sample_width == 1:
            audio_data = np.frombuffer(frames, dtype=np.uint8) - 128  # 8-bit PCM (unsigned)
        elif sample_width == 2:
            audio_data = np.frombuffer(frames, dtype=np.int16)  # 16-bit PCM (signed)
        else:
            raise ValueError("Unsupported sample width")
        
        # If stereo (2 channels), reshape to split left and right channels
        if n_channels == 2:
            print(f"{audio_data = }")
            audio_data = audio_data.reshape(-1, 2)
            print(f"{audio_data = }")
        
        # Calculate decibel values from amplitude
        db_values = amplitude_to_db(audio_data)
        
        # Time values for each sample (in seconds)
        time_values = np.linspace(0, len(audio_data) / frame_rate, num=len(audio_data))

        # Threshold for high volume (80 dB)
        threshold_db = 80
        high_volume_indices = np.where(db_values > threshold_db)[0]  # Indices where dB > 80

        # Calculate total duration of high-volume sections
        high_volume_times = time_values[high_volume_indices]
        total_high_volume_duration = len(high_volume_indices) / frame_rate  # Duration in seconds

        # Print the seconds where volume exceeds the threshold
        if len(high_volume_times) > 0:
            print(f"Seconds with volume higher than {threshold_db} dB:")
            for t in high_volume_times:
                print(f"{t:.2f} seconds")
        else:
            print(f"No parts of the audio file exceed {threshold_db} dB.")
        
        print(f"Total duration of high-volume sections (> {threshold_db} dB): {total_high_volume_duration:.2f} seconds")