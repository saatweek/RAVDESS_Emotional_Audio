"""Teach waveform addition and Fourier frequency peaks using synthetic tones.

Start here for the simplest signal example; no dataset, training, or emotion
labels are involved. Next read Audio_prep.py. See CODE_WALKTHROUGH.md.
"""
import numpy as np
import plotly.graph_objects as go
from reporting import write_report


def main():
    """Generate one second of two tones and save an offline interactive report."""
    sr = 1000
    # sr samples per second for one second. These two pure tones are a teaching
    # example, not an emotion dataset. Faster oscillations mean higher frequency.
    time = np.arange(sr) / sr
    first = .5 * np.sin(2 * np.pi * 30 * time)
    second = .2 * np.sin(2 * np.pi * 70 * time)
    wave = go.Figure()
    for name, signal in [('30 Hz', first), ('70 Hz', second), ('Combined', first + second)]:
        wave.add_scatter(x=time.tolist(), y=signal.tolist(), name=name, mode='lines')
    # Zoom the display to 0.2 seconds; the FFT still receives the full second.
    wave.update_layout(xaxis_title='Time (seconds)', yaxis_title='Amplitude', xaxis_range=[0, .2])
    # Recover the constituent frequencies from their sum. Multiplying magnitudes
    # by 2 accounts for the omitted negative-frequency half for these interior
    # bins; a general amplitude estimator treats DC/Nyquist bins separately.
    # Integer cycles fit the one-second window, so this example has clean peaks.
    spectrum = go.Figure(go.Scatter(x=np.fft.rfftfreq(sr, 1/sr).tolist(),
                                    y=(2*np.abs(np.fft.rfft(first+second))/sr).tolist(), mode='lines'))
    spectrum.update_layout(xaxis_title='Frequency (Hz)', yaxis_title='Single-sided amplitude', xaxis_range=[0, 100])
    write_report('runs/analysis/fourier.html', 'How two tones combine', 'Two sinusoidal signals sampled at 1,000 Hz.', [
        ('Time domain', 'The combined waveform is the sum of the two tones.', wave),
        ('Frequency domain', 'Peaks recover the 30 Hz and 70 Hz tones and their amplitudes.', spectrum)])


if __name__ == '__main__':
    main()
