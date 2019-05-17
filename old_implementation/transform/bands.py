import numpy as np

from old_implementation.dsp import filters


def get_band_data(channels_data, bands, sample_freq):
    w, h = channels_data.shape
    o = np.empty((0, h))
    for channel_data in channels_data:
        d = np.array(
            # Band pass for given bands
            [filters.butter_band_pass_filter(channel_data, lo, hi, sample_freq) for (lo, hi) in bands]
            # High pass band
            # [highpass(channel_data, bands[-1][-1], sample_freq)]
        )
        o = np.append(o, d, axis=0)
    return o
