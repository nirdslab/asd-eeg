import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from dataset_info import THERMAL_DATA_COLS, get_category
from ml import *

np.set_printoptions(threshold=sys.maxsize)


def save_training_progress(typ: str, history, name: str):
    p = 'mean_squared_error' if typ == 'linear' else 'acc'
    metric = history.history[p]
    val_metric = history.history['val_%s' % p]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    np.savez_compressed('out/metrics/%s-%s.npz' % (typ, name), metric=metric, val_metric=val_metric, loss=loss,
                        val_loss=val_loss)


def visualize_training_progress(typ: str, name: str):
    npz = np.load('out/metrics/%s-%s.npz' % (typ, name))
    metric = npz['metric']
    val_metric = npz['val_metric']
    loss = npz['loss']
    val_loss = npz['val_loss']
    epochs = range(1, len(metric) + 1)

    # Subplot 1
    plt.subplot(1, 2, 1)
    axes = plt.gca()
    axes.set_xlim([0, 100])
    if typ != "linear":
        axes.set_ylim([0, 1])
    else:
        axes.set_ylim([0, 50])
    plt.plot(epochs, metric, 'b', label='Training Set')
    plt.plot(epochs, val_metric, 'g', label='Validation Set')
    plt.title('Mean Squared Error' if typ == 'linear' else 'Accuracy')
    plt.legend()
    plt.draw()
    # Subplot 2
    plt.subplot(1, 2, 2)
    axes = plt.gca()
    axes.set_xlim([0, 100])
    if typ != "linear":
        axes.set_ylim([0, 1])
    else:
        axes.set_ylim([0, 50])
    plt.plot(epochs, loss, 'b', label='Training Set')
    plt.plot(epochs, val_loss, 'g', label='Validation Set')
    plt.title('Loss')
    plt.legend()
    plt.draw()
    # Show Plot
    plt.savefig('out/metrics/%s-%s.png' % (typ, name))
    plt.close()


def visualize_training_progress_of_validation_set_comparison(typ: str, best_so_far=False):
    # Subplot 1 (Init)
    plt.subplot(1, 2, 1)
    axes = plt.gca()
    axes.set_xlim([0, 100])
    if typ != "linear":
        axes.set_ylim([0, 1])
    else:
        axes.set_ylim([0, 50])

    # Subplot 2 (Init)
    plt.subplot(1, 2, 2)
    axes = plt.gca()
    axes.set_xlim([0, 100])
    if typ != "linear":
        axes.set_ylim([0, 1])
    else:
        axes.set_ylim([0, 50])

    for i, name in enumerate(['eeg', 'combined']):
        npz = np.load('out/metrics/%s-%s.npz' % (typ, name))
        val_metric = npz['val_metric']
        val_loss = npz['val_loss']
        epochs = range(1, len(val_metric) + 1)

        if best_so_far:
            val_metric = pd.Series(val_metric).cummax() if typ != 'linear' else pd.Series(val_metric).cummin()
            val_loss = pd.Series(val_loss).cummin()

        # Subplot 1 (Plot)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, val_metric, 'gb'[i], label=name)
        plt.title('Mean Squared Error Comparison' if typ == 'linear' else 'Accuracy Comparison')
        # Subplot 2 (Plot)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_loss, 'gb'[i], label=name)
        plt.title('Loss Comparison')

    # Subplot 1 (Draw)
    plt.subplot(1, 2, 1)
    plt.legend()
    plt.draw()

    # Subplot 2 (Draw)
    plt.subplot(1, 2, 2)
    plt.legend()
    plt.draw()

    # Show Plot
    plt.savefig('out/metrics/%s-comparison%s.png' % (typ, '-best' if best_so_far else ''))
    plt.close()


def get_data(typ: str, eeg: bool = False, thermal: bool = False, one_hot_encode=True, data='train'):
    """

    :param data: `test` data or `train` data depending on need
    :param one_hot_encode: Whether to one hot encode y. Use false for classification reports
    :param typ: `categorical` `binary` or `linear`
    :param eeg: True to get EEG data
    :param thermal: True to get thermal data
    :return:
    """
    if 'data.npz' not in os.listdir('out/'):
        print('data cache not found. building...')

        # Thermal Data
        df = pd.read_csv('data/thermal/thermal_data.csv')
        df['ID'] = df['ID'].str[-3:]
        df.set_index('ID', inplace=True)

        # EEG Data
        inputs = ['out/csv/' + x for x in os.listdir('out/csv')]
        eeg_x_l, thermal_x_l, y_category_l, y_score_l = [], [], [], []
        for i in inputs:
            print("Loading: %s" % i)
            p = i.split('/')[-1].split('_')[0]
            p_row = df.loc[p]
            score: int = int(p_row['ados'])
            module: int = int(p_row['module'])
            # Append values
            eeg_x_l.append(np.loadtxt(i, delimiter=','))
            thermal_x_l.append(p_row[THERMAL_DATA_COLS].values)
            # Category (0: TD - typically developing, 1: ASD - autism spectrum disorder, 2: AD - autism disorder)
            y_category_l.append(get_category(score, module))
            y_score_l.append(score)

        # generate n-d arrays
        eeg_x = (np.asarray(eeg_x_l) / 255).astype(np.float16)
        thermal_x = np.asarray(thermal_x_l).astype(np.float16)
        y_c = np.asarray(y_category_l).astype(np.int8)
        y_s = np.asarray(y_score_l).astype(np.int8)

        # test train split
        train_i, test_i = train_test_split(np.arange(len(eeg_x)), test_size=VALIDATION_SPLIT, shuffle=True)

        test_eeg_x, test_thermal_x = eeg_x[test_i], thermal_x[test_i]
        train_eeg_x, train_thermal_x = eeg_x[train_i], thermal_x[train_i]

        test_y_c, test_y_s = y_c[test_i], y_s[test_i]
        train_y_c, train_y_s = y_c[train_i], y_s[train_i]

        # save caches
        print('saving data cache...', end=' ')
        np.savez_compressed('out/data.npz',
                            eeg_x_train=train_eeg_x, thermal_x_train=train_thermal_x,
                            y_c_train=train_y_c, y_s_train=train_y_s,
                            eeg_x_test=test_eeg_x, thermal_x_test=test_thermal_x,
                            y_c_test=test_y_c, y_s_test=test_y_s)
        print('OK')

    print('loading data from cache...', end=' ')
    npz = np.load('out/data.npz')
    print('OK')

    if typ == 'categorical':
        y = npz['y_c_%s' % data]
        if one_hot_encode:
            y = to_categorical(y, NUM_CLASSES, 'int8')
    elif typ == 'binary':
        y = npz['y_c_%s' % data]
        y = np.where(y == 2, 1, y)
    elif typ == 'linear':
        y = npz['y_s_%s' % data]
    else:
        raise Exception('Only `categorical`, `binary` and `linear` types are allowed')

    if eeg and thermal:
        return npz['eeg_x_%s' % data], npz['thermal_x_%s' % data], y
    elif eeg:
        return npz['eeg_x_%s' % data], y
    elif thermal:
        return npz['thermal_x_%s' % data], y
    else:
        raise Exception('Should request at least one data')


def generate_model(typ: str, eeg: bool = False, thermal: bool = False, verbose=True):
    """
    :param verbose: Print model summary, etc if true
    :param typ: `categorical` `binary` or `linear`
    :param eeg: True if model accepts EEG data
    :param thermal: True if model accepts thermal data
    :return: Compiled Model ready to train
    """
    if eeg and thermal:
        MODEL_IN = [EEG_IN, THERMAL_IN]
        FUNC = layers.concatenate([EEG_F_OUT, THERMAL_F_OUT])
    elif eeg:
        MODEL_IN = [EEG_IN]
        FUNC = EEG_F_OUT
    elif thermal:
        MODEL_IN = [THERMAL_IN]
        FUNC = THERMAL_F_OUT
    else:
        raise Exception('No data requested')

    # Terminal Layer
    if typ == 'categorical':
        MODEL_OUT = layers.Dense(NUM_CLASSES, activation=act.softmax, kernel_regularizer=reg.l2(REG_SIZE))(FUNC)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    elif typ == 'binary':
        MODEL_OUT = layers.Dense(1, activation=act.sigmoid, kernel_regularizer=reg.l2(REG_SIZE))(FUNC)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif typ == 'linear':
        MODEL_OUT = layers.Dense(1, activation=act.linear, kernel_regularizer=reg.l2(REG_SIZE))(FUNC)
        loss = 'mse'
        metrics = ['mse']
    else:
        raise Exception('Only `categorical`, `binary` and `linear` types are allowed')

    # Create model
    model = Model(inputs=MODEL_IN, outputs=MODEL_OUT)
    if verbose:
        print(model.summary())
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model


def train_using_eeg_only(typ: str):
    eeg_x, y = get_data(typ, eeg=True)
    model = generate_model(typ, eeg=True)
    monitor = 'val_mean_squared_error' if typ == 'linear' else 'val_acc'
    cp = ModelCheckpoint(filepath="out/model/%s-eeg-weights.hdf5" % typ, monitor=monitor, verbose=0,
                         save_best_only=True)
    fit_history = model.fit(eeg_x, y, validation_split=VALIDATION_SPLIT, shuffle=True, epochs=EPOCHS, verbose=2,
                            batch_size=BATCH_SIZE, callbacks=[cp])
    model.load_weights("out/model/%s-eeg-weights.hdf5" % typ)
    model.save('out/model/%s-eeg-model.h5' % typ)
    save_training_progress(typ, fit_history, "eeg")


def train_using_thermal_only(typ: str):
    thermal_x, y = get_data(typ, thermal=True)
    model = generate_model(typ, thermal=True)
    monitor = 'val_mean_squared_error' if typ == 'linear' else 'val_acc'
    cp = ModelCheckpoint(filepath="out/model/%s-thermal-weights.hdf5" % typ, monitor=monitor, verbose=0,
                         save_best_only=True)
    fit_history = model.fit(thermal_x, y, validation_split=VALIDATION_SPLIT, shuffle=True, epochs=EPOCHS, verbose=2,
                            batch_size=BATCH_SIZE, callbacks=[cp])
    model.load_weights("out/model/%s-thermal-weights.hdf5" % typ)
    model.save('out/model/%s-thermal-model.h5' % typ)
    save_training_progress(typ, fit_history, "thermal")


def train_using_eeg_and_thermal(typ: str):
    eeg_x, thermal_x, y = get_data(typ, eeg=True, thermal=True)
    model = generate_model(typ, eeg=True, thermal=True)
    monitor = 'val_mean_squared_error' if typ == 'linear' else 'val_acc'
    cp = ModelCheckpoint(filepath="out/model/%s-combined-weights.hdf5" % typ, monitor=monitor, verbose=0,
                         save_best_only=True)
    fit_history = model.fit([eeg_x, thermal_x], y, validation_split=VALIDATION_SPLIT, shuffle=True, epochs=EPOCHS,
                            verbose=2, batch_size=BATCH_SIZE, callbacks=[cp])
    model.load_weights("out/model/%s-combined-weights.hdf5" % typ)
    model.save('out/model/%s-combined-model.h5' % typ)
    save_training_progress(typ, fit_history, "combined")


def get_y_predicted(typ: str, model, x):
    pred = model.predict(x)
    if typ == 'binary':
        return np.round(pred).flatten().astype(np.uint8)
    elif typ == 'categorical':
        return np.argmax(pred, axis=1)


def measure_classifier_performance(typ: str):
    if typ == 'linear':
        print('Linear Model not supported for F1, Precision, Recall and Support')

    eeg_model = generate_model(typ, eeg=True, verbose=False)
    eeg_model.load_weights("out/model/%s-eeg-weights.hdf5" % typ)
    eeg_x, eeg_y_true = get_data(typ, eeg=True, one_hot_encode=False, data='test')
    print('Calculating metrics: %s - %s' % (typ, 'EEG'))
    eeg_y_predicted = get_y_predicted(typ, eeg_model, eeg_x)
    print('Done')
    print(classification_report(eeg_y_true, eeg_y_predicted))
    if typ == 'binary':
        print("AUC: %f" % metrics.roc_auc_score(eeg_y_true, eeg_y_predicted, "weighted"))

    thermal_model = generate_model(typ, thermal=True, verbose=False)
    thermal_model.load_weights("out/model/%s-thermal-weights.hdf5" % typ)
    thermal_x, thermal_y_true = get_data(typ, thermal=True, one_hot_encode=False, data='test')
    thermal_x, i = np.unique(thermal_x, axis=0, return_index=True)
    thermal_y_true = thermal_y_true[i]
    print('Calculating metrics: %s - %s' % (typ, 'Thermal'))
    thermal_y_predicted = get_y_predicted(typ, thermal_model, thermal_x)
    print('Done')
    print(classification_report(thermal_y_true, thermal_y_predicted))
    if typ == 'binary':
        print("AUC: %f" % metrics.roc_auc_score(thermal_y_true, thermal_y_predicted, "weighted"))

    combined_model = generate_model(typ, eeg=True, thermal=True, verbose=False)
    combined_model.load_weights("out/model/%s-combined-weights.hdf5" % typ)
    eeg_x, thermal_x, combined_y_true = get_data(typ, eeg=True, thermal=True, one_hot_encode=False, data='test')
    print('Calculating metrics: %s - %s' % (typ, 'EEG + Thermal'))
    combined_y_predicted = get_y_predicted(typ, combined_model, [eeg_x, thermal_x])
    print('Done')
    print(classification_report(combined_y_true, combined_y_predicted))
    if typ == 'binary':
        print("AUC: %f" % metrics.roc_auc_score(combined_y_true, combined_y_predicted, "weighted"))


if __name__ == '__main__':
    os.makedirs('out/model', exist_ok=True)
    os.makedirs('out/metrics', exist_ok=True)
    for run_type in ['binary', 'categorical', 'linear']:
        # train_using_eeg_only(run_type)
        # train_using_thermal_only(run_type)
        # train_using_eeg_and_thermal(run_type)
        visualize_training_progress(run_type, "eeg")
        visualize_training_progress(run_type, "thermal")
        visualize_training_progress(run_type, "combined")
        visualize_training_progress_of_validation_set_comparison(run_type)
        visualize_training_progress_of_validation_set_comparison(run_type, best_so_far=True)
    for run_type in ['binary', 'categorical']:
        measure_classifier_performance(run_type)
# ======================================================================================================================
