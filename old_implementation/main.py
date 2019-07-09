import pandas as pd

from old_implementation import reader
from old_implementation.transform import get_band_data, aggregate_samples

participants = [2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
diagnosis = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
ados = {1: 2, 2: 19, 4: 12, 5: 5, 6: 9, 7: 0, 8: 5, 9: 17, 10: 26, 11: 11, 12: 16, 13: 16, 14: 0, 15: 7, 16: 4, 17: 0,
        18: 20, 19: 2, 20: 9, 21: 4, 22: 0}
label = ['NON-ASD', 'ASD']

# Brainwave Frequencies
# DELTA (0.1 to 3.5 Hz)
# THETA (4-8 Hz)
# ALPHA (8-12 Hz)
# BETA (12-30 Hz)
# GAMMA (above 30 Hz) [50-60 Hz filtered out)
bands = [(0.1, 4), (4, 7.5), [7.5, 12], (12, 30), (30, 100)]
band_names = ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
epochs = ['BASELINE', 'START', 'MIDDLE', 'END']


def run():
    participant_all_agg_df = None
    participant_band_df = None
    for p in participants:
        print('Participant {id:03d}:'.format(id=p), end=' ... ')
        epoch_agg_df = None
        epoch_band_df = None
        for epoch in epochs:
            # frequency is 250 Hz
            sampling_frequency = 250
            # ==================
            # Raw Data
            channels, data = reader.get_channels_and_readings(p, epoch)
            # raw = reader.get_mne_array(channels, ['eeg' for _ in range(len(channels))], sampling_frequency, readings)
            # raw.plot(duration=30, butterfly=False)
            # ==================
            # Band-Filtered Data
            band_data = get_band_data(data, bands, sampling_frequency)
            band_channels = [x + '_' + band_names[y] for x in channels for y in range(len(bands))]
            # DataFrame from Band Data
            df = pd.DataFrame(data=band_data.transpose(), columns=band_channels)
            df['Epoch'] = epoch
            if epoch_band_df is None:
                epoch_band_df = df
            else:
                epoch_band_df = epoch_band_df.append(df)
            # raw = reader.get_mne_array(band_channels, ['eeg'] * len(band_channels), sampling_frequency, band_data)
            # raw.plot(duration=30, butterfly=False)
            # ==================
            # Aggregate Data
            agg_data = aggregate_samples(band_data, sampling_frequency, 1)
            agg_channels = [y for x in band_channels for y in [x + '_A', x + '_P']]
            # raw = reader.get_mne_array(agg_channels, ['eeg'] * len(agg_channels), 1, agg_data)
            # raw.plot(duration=30, butterfly=False)
            df = pd.DataFrame(data=agg_data.transpose()[:-1], columns=agg_channels)
            df['Epoch'] = epoch
            if epoch_agg_df is None:
                epoch_agg_df = df
            else:
                epoch_agg_df = epoch_agg_df.append(df)
        # Participant Band Details
        epoch_band_df['Participant'] = p
        if participant_band_df is None:
            participant_band_df = epoch_band_df
        else:
            participant_band_df = participant_band_df.append(epoch_band_df)
        # Participant Aggregated Details
        epoch_agg_df['Participant'] = p
        if participant_all_agg_df is None:
            participant_all_agg_df = epoch_agg_df
        else:
            participant_all_agg_df = participant_all_agg_df.append(epoch_agg_df)
        print('OK %s %s' % (str(participant_band_df.shape), str(participant_all_agg_df.shape)))
    # Iterations Completed. Export CSVs
    participant_band_df['Z'] = participant_band_df['Participant'].apply(
        lambda x: label[diagnosis[participants.index(x)]])
    participant_all_agg_df['Z'] = participant_all_agg_df['Participant'].apply(
        lambda x: label[diagnosis[participants.index(x)]])
    participant_all_agg_df['ZR'] = participant_all_agg_df['Participant'].apply(
        lambda x: ados[x])
    # participant_band_df.to_csv('data/BAND.csv')
    participant_all_agg_df.to_csv('data/AGG.csv')


def prepare():
    reader.convert_to_csv(participants)


# Uncomment this to create CSV files from XLSX files (Much Faster)
# prepare()
# Uncomment this line to execute filtering, feature extraction and aggregation
run()
