import os
import shutil
import tarfile
from multiprocessing import Pool, cpu_count
from threading import Lock

import pandas as pd
from pywt import cwt

from dataset_info import *

baselines = {}
baselines_lock = Lock()


def get_wt_power(_time, _signal, _scales, _wavelet):
    dt = _time[1] - _time[0]
    [coefficients, frequencies] = cwt(_signal, _scales, _wavelet, dt)
    power = (abs(coefficients)) ** 2
    return power, frequencies


def read_file(data, epoch):
    return pd.read_csv('data/eeg/' + data + '_' + epoch + '.csv')


def save_results(results):
    _data, (_participant, _epoch, _electrode, _score) = results
    for _ref_power, chunk in _data:
        file_path = 'out/csv/%s_%s_%s_%d_%d.csv' % (_participant, _epoch, _electrode, _score, chunk)
        np.savetxt(file_path, _ref_power, delimiter=',', fmt='%i')
    print('\t(P: %s) [%s] (E: %s): OK' % (_participant, _epoch, _electrode))


def run():
    with Pool(cpu_count()) as pool:
        for i, p in enumerate(PARTICIPANTS):
            # Baseline DataFrame
            baseline_df = read_file(p, BASELINE)
            # Iterate through START, MIDDLE and END epochs
            for e in EPOCHS:
                # Epoch DataFrame
                epoch_df = read_file(p, e)
                # Run for each EEG electrode
                for el in baseline_df:
                    # Baseline Signal for Electrode
                    b_signal: pd.Series = baseline_df[el].values.squeeze()
                    # Epoch Signal for Electrode
                    e_signal: pd.Series = epoch_df[el].values.squeeze()
                    # perform_calculation(e_signal[:-1], b_signal[:-1], (p, e, el, LABELS[i]))
                    pool.apply_async(perform_calculation, [e_signal[:-1], b_signal[:-1], (p, e, el, LABELS[i])],
                                     callback=save_results)
        pool.close()
        pool.join()
    # Compress all CSVs in csv.tar.gz
    print('Generating csv.tar.gz...', end=' ')
    with tarfile.open("out/csv.tar.gz", "w:gz") as tar:
        tar.add("out/csv", arcname="csv")
        tar.close()
        print('OK')
    # Remove generated CSVs
    print('Removing temporary files...', end=' ')
    shutil.rmtree("out/csv/")
    print('OK')


def resize(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).max(-1).max(1)


def perform_calculation(_e_signal_full: pd.Series, _b_signal: pd.Series, metadata):
    # print('Obtaining Baseline Mean Power')
    key = '%s_%s' % (metadata[0], metadata[2])
    baselines_lock.acquire()
    if key in baselines.keys():
        b_power, b_freqs = baselines[key]
    else:
        b_power, b_freqs = get_wt_power(np.arange(0, _b_signal.size, 1. / FREQ), _b_signal, WT_SCALES, WAVELET)
        baselines[key] = (b_power, b_freqs)
    baselines_lock.release()
    b_mean_power = b_power.mean(1).reshape(SCALE_COUNT, 1)
    # print('Calculating Electrode Power')
    _e_power_full, e_freqs = get_wt_power(np.arange(0, _e_signal_full.size, 1. / FREQ), _e_signal_full, WT_SCALES,
                                          WAVELET)
    # Verify that both baseline and epoch signals have same frequencies after WT
    assert np.array_equal(b_freqs, e_freqs)
    # reference to baseline
    ref_p_full = np.subtract(_e_power_full, b_mean_power)
    # Split transform into W chunks
    data = []
    i = 0
    for _ref_p_chunk_trans in np.reshape(np.transpose(ref_p_full), (W, ref_p_full.shape[1] // W, ref_p_full.shape[0])):
        _ref_chunk = _ref_p_chunk_trans.transpose()
        # down sampling in time axis
        _ref_chunk = resize(_ref_chunk, (SCALE_COUNT, SCALE_COUNT))
        # Normalize range
        min_, max_, mean_ = np.min(_ref_chunk), np.max(_ref_chunk), np.mean(_ref_chunk)
        _ref_chunk = ((_ref_chunk - min_) / (max_ - min_) * 255).astype(np.uint8)
        data.append([_ref_chunk, i])
        i += 1
    return data, metadata


if __name__ == '__main__':
    os.makedirs('out/csv', exist_ok=True)
    run()
