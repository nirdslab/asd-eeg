import os

import numpy as np
from matplotlib import pyplot as plt

data_dir = 'out/csv/'


def run():
    file_names = [data_dir + x for x in os.listdir(data_dir)]

    electrodes = sorted(["F7", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "C3", "C4"])
    file_names = sorted([x for x in sorted(file_names) if x.split('_')[2] in electrodes and "BASELINE" not in x])

    max_data = [[[0, np.zeros((150, 150))] for _ in range(len(electrodes))] for _ in range(2)]
    sum_data = [[[0, np.zeros((150, 150))] for _ in range(len(electrodes))] for _ in range(2)]

    for file_name in file_names:
        participant, epoch, electrode, label, chunk = file_name.split('/')[-1].split('_')
        chunk = int(chunk[:-4])
        label = int(label)
        diagnosis = 'ASD' if label else 'TD'

        data = np.loadtxt(file_name, delimiter=',').transpose()

        # Create individual PNG image
        name = '(%s) %s - %s [%s, %s/6] ' % (diagnosis, participant, epoch, electrode, chunk + 1)
        plt.figure(name)
        plt.title(name)
        plt.imshow(data, cmap='seismic')
        f_name = 'out/png/%s_%s_%s_%s_%s.png' % (diagnosis, electrode, epoch, chunk, participant)
        plt.savefig(f_name)
        plt.close(name)
        print(f_name + '\tDONE!')

        i = electrodes.index(electrode)

        # Max Aggregate for each Electrode
        max_data[label][i][0] += 1
        max_data[label][i][1] = np.maximum(max_data[label][i][1], data)

        # Avg Aggregate for each Electrode
        sum_data[label][i][0] += 1
        sum_data[label][i][1] = np.add(sum_data[label][i][1], data)

    for label, electrodes_data in enumerate(max_data):
        for i, electrode_data in enumerate(electrodes_data):
            electrode = electrodes[i]
            (cnt, mat) = electrode_data
            diagnosis = 'ASD' if label else 'TD'

            name = '(%s) Max Aggregate [%s] ' % (diagnosis, electrode)
            plt.figure(name)
            plt.title(name)
            plt.imshow(mat, cmap='seismic')

            f_name = 'out/png-max/%s_%s' % (diagnosis, electrode)
            plt.savefig(f_name)
            plt.close(name)

    for label, electrodes_data in enumerate(sum_data):
        for i, electrode_data in enumerate(electrodes_data):
            electrode = electrodes[i]
            (cnt, mat) = electrode_data
            mat /= cnt
            diagnosis = 'ASD' if label else 'TD'

            name = '(%s) Mean Aggregate [%s] ' % (diagnosis, electrode)
            plt.figure(name)
            plt.title(name)
            plt.imshow(mat, cmap='seismic')

            f_name = 'out/png-avg/%s_%s' % (diagnosis, electrode)
            plt.savefig(f_name)
            plt.close(name)


if __name__ == '__main__':
    os.makedirs('out/png', exist_ok=True)
    os.makedirs('out/png-avg', exist_ok=True)
    os.makedirs('out/png-max', exist_ok=True)
    run()
