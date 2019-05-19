import numpy as np

# Dataset Information
PARTICIPANTS = ['%03d' % p for p in [2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
ADOS_SCORES = [19, 12, 5, 0, 5, 11, 16, 16, 0, 7, 4, 0, 20, 2, 9, 4, 0]
CUTOFF = 7
LABELS = [1 if x >= CUTOFF else 0 for x in ADOS_SCORES]
BASELINE = 'BASELINE'
EPOCHS = ['START', 'MIDDLE', 'END']
FREQ = 250

# Hyper Parameters
SCALE_COUNT = 150  # 150 scales of frequency
W = 180 // 30  # 30 second chunks
WAVELET = 'cmor1.5-1.0'  # Complex Morlet Wavelet with center frequency = 1 Hz and bandwidth = 1.5 Hz
WT_SCALES = np.arange(1, SCALE_COUNT + 1)
