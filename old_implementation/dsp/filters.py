import numpy as np
from scipy.signal import butter, sosfilt, periodogram


def butter_band_pass(lo, hi, fs, order):
    nyq = 0.5 * fs
    low = lo / nyq
    high = hi / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_band_pass_filter(data, freq_lo, freq_hi, freq_sample, order=5):
    sos = butter_band_pass(freq_lo, freq_hi, freq_sample, order=order)
    y = sosfilt(sos, data)
    return y


def butter_high_pass(lo, fs, order):
    nyq = 0.5 * fs
    low = lo / nyq
    sos = butter(order, low, analog=False, btype='high', output='sos')
    return sos


def butter_high_pass_filter(data, lo, fs, order=5):
    sos = butter_high_pass(lo, fs, order=order)
    y = sosfilt(sos, data)
    return y


def get_power(series, sampling_freq, t):
    agg_size = sampling_freq * t
    peak_powers = np.array([
        periodogram(series[i:i + agg_size], sampling_freq)[1].max() for i in range(0, len(series), agg_size)
    ])
    return peak_powers


def get_amplitude(series, sampling_freq, t):
    agg_size = sampling_freq * t
    peak_amplitude = np.array([
        np.max(series[i:i + agg_size]) for i in range(0, len(series), agg_size)
    ])
    return peak_amplitude
