import os

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.fftpack import fft

from analysis_tool.params import AUDIO_FILES_PATH


def to_decibels(rms):
    return 20 * np.log10(rms)


class AudioVolumeAnalyzer:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name

    def get_too_loud_fragments(self) -> list[tuple[float, float]]:
        return self._get_volume_problems(self._is_too_loud)
    
    def get_too_quiet_fragments(self) -> list[tuple[float, float]]:
        return self._get_volume_problems(self._is_too_low)
    
    def get_high_noise_fragments(self) -> list[tuple[float, float]]:
        return self._get_volume_problems(self._is_ambient_noise_too_loud_for_audio_chunk)

    def _get_volume_problems(self, callback: callable) -> list[tuple[float, float]]:
        # Load the audio file
        audio = AudioSegment.from_file(os.path.join(AUDIO_FILES_PATH, self.file_name))

        # Convert to mono (if stereo) and get raw data
        audio = audio.set_channels(1)
        # Split the audio into chunks (e.g., 100ms windows)
        chunk_length_ms = 300
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        # Detect loud fragments by computing the RMS
        searched_fragments = []
        least_chunks_to_count = 4
        exceptions_count_to_stop_counting = 3
        ambient_noise = min([to_decibels(ch.rms) for ch in chunks if to_decibels(ch.rms) != -np.inf])

        start_time = None
        record_counter = 0
        exceptions_count = 0
        for i, chunk in enumerate(chunks):
            # db = to_decibels(chunk.rms)
            # ts = i * chunk_length_ms / 1e3

            # print(f"{db, ts, ambient_noise = }")
            ts = i * chunk_length_ms / 1e3
            matches = callback(chunk, ambient_noise, ts=ts)
            # print(f"{start_time, matches, record_counter, ts = }")

            if matches and start_time:
                record_counter += 1

            if matches and not start_time:
                start_time = i * chunk_length_ms / 1e3
                record_counter = 1
                exceptions_count = 0
            
            if not matches and start_time:
                exceptions_count += 1

                if exceptions_count >= exceptions_count_to_stop_counting:
                    if record_counter >= least_chunks_to_count:
                        searched_fragments.append((start_time, (i+1) * chunk_length_ms / 1e3))

                    record_counter = 0  
                    start_time = None
                    exceptions_count = 0
                    continue
        
        if start_time:
            searched_fragments.append((start_time, (i+1) * chunk_length_ms / 1e3))

        # Join overlapping timestamps
        new_fragments = []
        _start, _end = None, None
        for start_ts, end_ts in searched_fragments:
            if _start is None:
                _start, _end = start_ts, end_ts
                continue
            
            if _end == start_ts:
                _end = end_ts
            else:
                new_fragments.append((_start, _end))
                _start, _end = None, None
        
        if _start:
            new_fragments.append((_start, _end))


        return new_fragments

    @staticmethod
    def _is_ambient_noise_too_loud_for_audio_chunk(audio: np.array, *args, **kwargs) -> bool:
        raw_data = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate

        # Perform Fast Fourier Transform (FFT)
        n = len(raw_data)
        fft_data = fft(raw_data)

        # Get the frequency spectrum (positive frequencies only)
        frequencies = np.fft.fftfreq(n, 1/sample_rate)
        positive_frequencies = frequencies[:n//2]
        magnitude = np.abs(fft_data[:n//2])

        # Speech frequency band (300 Hz to 3000 Hz)
        speech_band_low = 300
        speech_band_high = 3000

        # Calculate total energy (sum of magnitudes)
        total_energy = np.sum(magnitude)

        # Energy in the speech band
        speech_band_energy = np.sum(magnitude[(positive_frequencies >= speech_band_low) & (positive_frequencies <= speech_band_high)])

        # Energy outside the speech band (ambient noise energy)
        ambient_noise_energy = total_energy - speech_band_energy

        # This threshold was set empirically xD
        return ambient_noise_energy >= 60_000_000

        # print(f"{total_energy:_}, {ambient_noise_energy:_}, {kwargs = }")

        # # Percentage of energy in the ambient noise
        # ambient_noise_ratio = (ambient_noise_energy / total_energy) * 100
        # return ambient_noise_ratio > 50
    
    @staticmethod
    def _is_too_loud(audio: np.array, *args, **kwargs) -> bool:
        db = to_decibels(audio.rms)
        too_loud_speech_thresh = 55  # db
        return db > too_loud_speech_thresh

    @staticmethod
    def _is_too_low(audio: np.array, ambient_noise: np.float32, *args, **kwargs) -> bool:
        db = to_decibels(audio.rms)
        too_quiet_speech_thresh = 40  # db
        return db > too_quiet_speech_thresh or db - ambient_noise < 10