import mne
import numpy as np
import pandas as pd


def convert_to_csv(participant_ids):
    for p in participant_ids:
        file_name = 'data/{id:03d}_TS'.format(id=p)
        print('Reading %s.xlsx' % file_name)
        eeg_csv_data = pd.read_excel(file_name + '.xlsx', sheet_name=None)
        # remove the timestamp column
        for key in eeg_csv_data.keys():
            sheet = eeg_csv_data[key]
            del sheet[' ']
            tag = key[4:].strip().upper()
            print('Processing Sheet: %s' % tag, end='...')
            sheet.to_csv('%s_%s.csv' % (file_name[:-3], tag), index=False)
            print('OK')


def get_data_frame(participant_id: int, epoch) -> pd.DataFrame:
    eeg_csv_data = pd.read_csv('data/{id:03d}_{epoch}.csv'.format(id=participant_id, epoch=epoch))
    return eeg_csv_data.applymap(lambda x: float(x) / (10 ** 6))


def get_channels_and_readings(participant_id: int, epoch: str) -> [object, np.ndarray]:
    eeg_csv_data = get_data_frame(participant_id, epoch)
    channels = list(eeg_csv_data)
    readings = eeg_csv_data.values.transpose()
    return channels, readings


def get_mne_array(channels, channel_types, frequency, readings) -> mne.io.RawArray:
    # Create the info structure needed by MNE
    info = mne.create_info(ch_names=channels, sfreq=frequency, ch_types=channel_types)

    # Finally, create the Raw object and return it
    raw_array = mne.io.RawArray(readings, info)
    return raw_array
