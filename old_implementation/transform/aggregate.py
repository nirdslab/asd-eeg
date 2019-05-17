import numpy as np

from old_implementation.dsp import amplitude, power


def aggregate_samples(data, sample_freq, t):
    # Aggregate results of t seconds
    return np.array([sub_series for series in data for sub_series in
                     [amplitude(series, sample_freq, t), power(series, sample_freq, t)]])
