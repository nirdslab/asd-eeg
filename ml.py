import keras.activations as act
import keras.layers as layers
import keras.regularizers as reg
from keras import Input

VALIDATION_SPLIT = 0.1
EPOCHS = 100
BATCH_SIZE = 256
REG_SIZE = 0.001
DROPOUT_RATE = 0.2
EEG_IN = Input(shape=(150, 150))
NUM_CLASSES = 3
THERMAL_IN = Input(shape=(80,))

# EEG Pipeline - 1 - High Resolution Layer
EEG_F_CONV_1 = layers.Conv1D(16, 4, activation=act.relu, kernel_regularizer=reg.l2(REG_SIZE))(EEG_IN)
EEG_F_CONV_1 = layers.MaxPooling1D(pool_size=2, strides=2)(EEG_F_CONV_1)
EEG_F_CONV_1 = layers.Dropout(rate=DROPOUT_RATE)(EEG_F_CONV_1)

# EEG Pipeline - 2- Low Resolution Layer
EEG_F_CONV_2 = layers.Conv1D(32, 4, activation=act.relu, kernel_regularizer=reg.l2(REG_SIZE))(EEG_F_CONV_1)
EEG_F_CONV_2 = layers.MaxPooling1D(pool_size=2, strides=2)(EEG_F_CONV_2)
EEG_F_CONV_2 = layers.Dropout(rate=DROPOUT_RATE)(EEG_F_CONV_2)

# EEG Pipeline - 3 - Dense Layer (Output)
EEG_F_DENSE = layers.Flatten()(EEG_F_CONV_2)
EEG_F_DENSE = layers.Dense(128, activation=act.relu, kernel_regularizer=reg.l2(REG_SIZE))(EEG_F_DENSE)
EEG_F_OUT = layers.Dropout(rate=DROPOUT_RATE)(EEG_F_DENSE)

# Thermal Pipeline - 1 - Dense Layer
THERMAL_F_DENSE_1 = layers.Dense(16, activation=act.relu, kernel_regularizer=reg.l2(REG_SIZE))(THERMAL_IN)
THERMAL_F_DENSE_1 = layers.Dropout(rate=DROPOUT_RATE)(THERMAL_F_DENSE_1)

# Thermal Pipeline - 2 - Dense Layer (Output)
THERMAL_F_DENSE_2 = layers.Dense(128, activation=act.relu, kernel_regularizer=reg.l2(REG_SIZE))(THERMAL_F_DENSE_1)
THERMAL_F_OUT = layers.Dropout(rate=DROPOUT_RATE)(THERMAL_F_DENSE_2)
