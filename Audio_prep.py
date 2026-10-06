"""Teach waveform, FFT, STFT, and MFCC views using one complete recording.

These are illustrative plots, not the exact feature() training pipeline: no edge
trimming, four-second crop/padding, or CNN z-normalization is applied here.
"""
import argparse
from pathlib import Path
import librosa
import numpy as np
import plotly.graph_objects as go
from reporting import write_report


def main():
    """Load mono 16 kHz audio and export four complementary views."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', default='sample_audio.wav')
    parser.add_argument('--output', default='runs/analysis/audio_features.html')
    args = parser.parse_args()
    # Start here if audio is new to you. A waveform is amplitude vs time.
    # Dividing sample indices by sr converts the horizontal axis to seconds.
    signal, sr = librosa.load(args.audio, sr=16000)
    if not len(signal) or not np.isfinite(signal).all():
        raise ValueError('Audio must contain finite samples.')
    waveform = go.Figure(go.Scatter(x=(np.arange(len(signal))/sr).tolist(), y=signal.tolist(), mode='lines'))
    waveform.update_layout(xaxis_title='Time (seconds)', yaxis_title='Amplitude')
    # FFT describes frequency content across the ENTIRE clip, losing when a
    # sound occurred. rfft omits redundant negative frequencies of real audio.
    # abs removes phase and keeps magnitude; this plot is not a power spectrum.
    magnitude = np.abs(np.fft.rfft(signal)) / len(signal)
    spectrum = go.Figure(go.Scatter(x=np.fft.rfftfreq(len(signal), 1/sr).tolist(), y=magnitude.tolist(), mode='lines'))
    spectrum.update_layout(xaxis_title='Frequency (Hz)', yaxis_title='FFT magnitude / sample count')
    # STFT = short-time Fourier transform: repeat the FFT in overlapping windows.
    # Window length trades time localization against frequency detail. Its matrix
    # keeps frequency vertically and time horizontally, unlike one global FFT.
    stft = librosa.stft(signal, n_fft=512, hop_length=160)
    # Amplitude-to-dB uses a factor of 20; power-to-dB uses 10 because power is
    # proportional to amplitude squared. Do not interchange them without checking.
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sr, hop_length=160).tolist()
    spectrogram = go.Figure(go.Heatmap(z=db.tolist(), x=times, y=librosa.fft_frequencies(sr=sr, n_fft=512).tolist(), colorscale='Viridis'))
    spectrogram.update_layout(xaxis_title='Time (seconds)', yaxis_title='Frequency (Hz)')
    # MFCCs summarize spectral shape; coefficient index is NOT Hertz or pitch.
    # This educational plot shows 20 coefficients. The MLP uses 40; the selected
    # CNN uses 64 log-mel bands instead. This call also uses librosa's default mel
    # filter count; training explicitly sets 64. This is visualization, not training.
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=512, hop_length=160, n_mfcc=20)
    coefficients = go.Figure(go.Heatmap(z=mfcc.tolist(), x=times, y=list(range(1, 21)), colorscale='RdBu'))
    coefficients.update_layout(xaxis_title='Time (seconds)', yaxis_title='MFCC coefficient')
    write_report(args.output, 'Audio feature exploration', f'{Path(args.audio).name} | {len(signal)/sr:.2f} seconds | resampled to {sr:,} Hz', [
        ('Waveform', 'Amplitude over time, with a time axis measured in seconds.', waveform),
        ('Frequency spectrum', 'The nonnegative-frequency half of the real-valued signal’s Fourier transform.', spectrum),
        ('Spectrogram', 'Short-time Fourier magnitudes in decibels relative to the largest magnitude.', spectrogram),
        ('MFCCs', 'Twenty mel-frequency cepstral coefficients describe the spectral envelope.', coefficients)])
    print(Path(args.output).resolve())


if __name__ == '__main__':
    main()
