import librosa, librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

file = 'sample_audio.wav'

# Waveform_____

# We'll first load the file using librosa.load(), which will take in the path of the file and the sample rate
# It shall return signal and the sample rate
# The signal is going to be a 1-D numpy array and it's  length = (sr * duration of the song)
signal, sr = librosa.load(file, sr=22050)
x = np.linspace(0, len(signal), len(signal))
y = np.array(signal)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
fig.update_layout(xaxis_title='time', yaxis_title='amplitude', title='waveform')
fig.show()

# FFT -> Spectrum
fft = np.fft.fft(signal) # returns a 1-D numpy array with length = (sr * duration of the song). Each of which is a complex value
magnitude = np.abs(fft)  # now we get the magnitude of the complex value
                         # This magnitude indicates the contribution of each frequency to the overall sound
frequency = np.linspace(0, sr, len(magnitude))
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=frequency, y=magnitude, mode='lines'))
fig2.update_layout(xaxis_title='Frequency', yaxis_title='Magnitude', title='Power Spectrum')
fig2.show()
# You'll see that the graph is symmetrical, and it will always be symmetrical, so we therefore, take only half of it

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=left_frequency, y=left_magnitude, mode='lines'))
fig3.update_layout(xaxis_title='Frequency', yaxis_title='Magnitude', title='Power Spectrum (Half)')
fig3.show()

# STFT -> Spectrogram
n_fft = 2048  # no. of samples per fft. This is basically the window that we're considering when performing a single
              # short fourier transform

hop_length = 512 # This is the amount that we're shifting each fourier transform to the right
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

fig4 = go.Figure(data=go.Heatmap(x=x, y=left_frequency, z=log_spectrogram, type='heatmap', colorscale='Viridis'))
fig4.update_layout(title='Spectrogram',
                   yaxis=dict(title='Frequency'),
                   xaxis=dict(title='Time'))
fig4.show()

# If you want to use matplotlib to view the spectrogram, un-comment the following 5 lines
# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel('Time')
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

# MFCCs
MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

fig5 = go.Figure(data=go.Heatmap(x=x, y=np.arange(1, 13), z=MFCCs, type='heatmap', colorscale='Viridis'))
fig5.update_layout(title='MFCC',
                   yaxis=dict(title='MFCC Coeficients'),
                   xaxis=dict(title='Time'))
fig5.show()

# If you want to use matplotlib to view the MFCC, un-comment the following 5 lines
# librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("MFCC")
# plt.colorbar()
# plt.show()