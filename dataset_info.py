import numpy as np

# Dataset Information
PARTICIPANTS = ['%03d' % p for p in [2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
ADOS_SCORES = [19, 12, 5, 0, 5, 11, 16, 16, 0, 7, 4, 0, 20, 2, 9, 4, 0]
LABELS = [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
BASELINE = 'BASELINE'
EPOCHS = ['START', 'MIDDLE', 'END']
FREQ = 250

# Hyper Parameters
SCALE_COUNT = 150  # 150 scales of frequency
W = 180 // 30  # 30 second chunks
WAVELET = 'cmor1.5-1.0'  # Complex Morlet Wavelet with center frequency = 1 Hz and bandwidth = 1.5 Hz
WT_SCALES = np.arange(1, SCALE_COUNT + 1)

THERMAL_DATA_COLS = [
    'right_cheek_before', 'rc_max_before', 'left_cheek_before', 'lc_max_before', 'right_eye_before',
    're_max_before', 'left_eye_before', 'le_max_before', 'right_crook_before', 'rcr_max_before', 'left_crook_before',
    'lcr_max_before', 'nose_before', 'n_min_before', 'open_mouth_(avg)_before', 'open_mouth_max_before',
    'avg_left_max_before', 'avg_right_max_before', 'avg_max_before', 'left_right_before', 'right_cheek_during',
    'rc_max_during', 'left_cheek_during', 'lc_max_during', 'right_eye_during', 're_max_during', 'left_eye_during',
    'le_max_during', 'right_crook_during', 'rcr_max_during', 'left_crook_during', 'lcr_max_during', 'nose_during',
    'n_min_during', 'open_mouth_(avg)_during', 'open_mouth_max_during', 'avg_left_max_during', 'avg_right_max_during',
    'avg_max_during', 'left_right_during', 'right_cheek_after', 'rc_max_after', 'left_cheek_after', 'lc_max_after',
    'right_eye_after', 're_max_after', 'left_eye_after', 'le_max_after', 'right_crook_after', 'rcr_max_after',
    'left_crook_after', 'lcr_max_after', 'nose_after', 'n_min_after', 'open_mouth_(avg)_after', 'open_mouth_max_after',
    'avg_left_max_after', 'avg_right_max_after', 'avg_max_after', 'left_right_after', 'right_cheek_delta',
    'rc_max_delta', 'left_cheek_delta', 'lc_max_delta', 'right_eye_delta', 're_max_delta', 'left_eye_delta',
    'le_max_delta', 'right_crook_delta', 'rcr_max_delta', 'left_crook_delta', 'lcr_max_delta', 'nose_delta',
    'n_min_delta', 'open_mouth_(avg)_delta', 'open_mouth_max_delta', 'avg_left_max_delta', 'avg_right_max_delta',
    'avg_t_max_delta', 'left_right_delta'
]


def get_category(ados_score: int, module: int) -> int:
    if module in [3, 4]:
        if ados_score < 7:
            return 0
        elif (ados_score < 9 and module == 3) or (ados_score < 10 and module == 4):
            return 1
        else:
            return 2
    elif module in [1, 2]:
        raise Exception('Cutoffs Unknown for ADOS-2 Module - %d' % module)
    else:
        raise Exception('Unknown ADOS-2 Module - %d' % module)
