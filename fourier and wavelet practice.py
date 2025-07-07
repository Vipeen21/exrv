import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import morlet, cwt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt


# here we simulate a simple signal: sum of two sine waves
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)

# calcualte the fourier transform
fft_values = fft(signal)

# plot the original signal
plt.figure(figsize=(20,11))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# plot the fourier transform of the simulated sequence
frequencies = np.fft.fftfreq(len(t), d=t[1] - t[0])
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fft_values))
plt.title('Fourier Transform (Frequency Domain)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()



# simulate another signal: sum of sine waves with varying frequencies
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 20 * t)

# set the wavelet scales
scales = np.arange(1, 100)

# performe continuous wavelet transform using the Morlet wavelet
coefficients = cwt(signal, morlet, scales)

# display the original signal and the wavelet transform result
plt.figure(figsize=(20, 11))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 100], cmap='PRGn', aspect='auto',
           vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
plt.title('Wavelet Transform (Time-Frequency Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Scale')
plt.tight_layout()
plt.show()

#see this link for more details
# #https://medium.com/pythoneers/wavelet-transform-a-practical-approach-to-time-frequency-analysis-662bdadeb08b

#real world seismic data analysis


# create a client for IRIS data
client = Client("IRIS")

# specify the event time and parameters
starttime = UTCDateTime("2020-01-01T00:00:00")
endtime = starttime + 60 * 60  # 1 hour of data

# retrieve the waveform data for a specific station
# Example: Station ANMO, Network IU, Channel BHZ (Broadband High Gain Vertical)
waveform = client.get_waveforms(network="IU", station="ANMO", location="00", channel="BHZ",
                                starttime=starttime, endtime=endtime)

# extract the trace
tr = waveform[0]

# plot the waveform (time domain)
tr.plot(type="relative")


# extract the seismic data from the trace
seismic_data = tr.data
times = np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts)

# increase the range and resolution of scales
scales = np.arange(1, 256)

# perform wavelet transform using the Morlet wavelet
coefficients = cwt(seismic_data, morlet, scales)

# plot the original signal and the wavelet transform result
plt.figure(figsize=(12, 6))

# plot the seismic signal (time domain)
plt.subplot(2, 1, 1)
plt.plot(times, seismic_data)
plt.title('Seismic Signal (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# plot the wavelet transform (time-frequency domain) with adjusted color scaling
plt.subplot(2, 1, 2)
plt.imshow(np.abs(coefficients), extent=[times[0], times[-1], scales[-1], scales[0]],
           cmap='inferno', aspect='auto', vmax=np.percentile(np.abs(coefficients), 99),
           vmin=np.percentile(np.abs(coefficients), 1))
plt.title('Wavelet Transform (Time-Frequency Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Scale')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()