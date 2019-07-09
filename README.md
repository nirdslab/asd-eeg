# Predicting ASD using EEG Data

## Dataset #1

* **Study Name** - EEG and Thermal Activity during Social Interaction in ASD
* **Sample Age** - 5-17 years

### Data Acquisition Description 
* EEG and infrared thermographs were collected during a live administration of the ADOS-2.
* Session durations averaged 40-45min. 

### EEG Equipment 
* Brain Products 32-channel Live Amp wireless system (sampling rate = 256Hz). ActiCap Slim active electrodes (gel-based). 

### Acquisition Software
* Brain Vision Recorder

### Data Pre-processing Software
* EEGLab version 14.1.2b

### Data Pre-processing Pipeline
1. Remove low frequency baseline drift with a 1 Hz high-pass filter.
2. Remove 50-60 Hz AC line noise by applying the CleanLine plugin.
3. Clean continuous raw data using the clean raw data plugin [Mullen et al. 2015]. The
clean raw-data plugin first performs bad channel rejection based on two criteria: (1) channels
that have flat signals longer than 5 seconds and (2) channels poorly correlated with adjacent channels.
It then applies artifact subspace reconstruction (ASR)—an algorithm that removes non-stationary, high
variance signals from the EEG then uses calibration data (1 min sections of clean EEG) to reconstruct
the missing data using a spatial mixing matrix.
4. Interpolate removed channels.
5. Re-reference channels to average reference.
6. ICA

### Epoch EEG Files Procedure
For each EEG time series (TS), the following 4 epoch files were created:
1.	TS-baseline   = 60 sec baseline
2.	TS-start      = 180 sec epoch at the start of the social interaction
3.	TS-middle     = 180 sec epoch at the middle of the social interaction
4.	TS-end        = 180 sec epoch at the end of the social interaction

i/2 = middle of TS, i = total duration of TS

### File Naming Scheme

XXX_TS_epoch

* XXX = subject
* TS = time series
* Example: 
    * 001_TS_baseline = subject #1 baseline epoch
    * 004_TS_end = subject #4 end epoch

### Training Model
CNN with window size of 25 and stride of 5 was used

### Sample Run
```
Epoch 1/32
2355/2355 [==============================] - 1s 323us/sample - loss: 0.6861 - acc: 0.5715
Epoch 2/32
2355/2355 [==============================] - 1s 284us/sample - loss: 0.6485 - acc: 0.6416
Epoch 3/32
2355/2355 [==============================] - 1s 292us/sample - loss: 0.5340 - acc: 0.7461
Epoch 4/32
2355/2355 [==============================] - 1s 292us/sample - loss: 0.4212 - acc: 0.8136
Epoch 5/32
2355/2355 [==============================] - 1s 290us/sample - loss: 0.3424 - acc: 0.8454
Epoch 6/32
2355/2355 [==============================] - 1s 272us/sample - loss: 0.3009 - acc: 0.8764
Epoch 7/32
2355/2355 [==============================] - 1s 285us/sample - loss: 0.2444 - acc: 0.9045
Epoch 8/32
2355/2355 [==============================] - 1s 283us/sample - loss: 0.2192 - acc: 0.9197
Epoch 9/32
2355/2355 [==============================] - 1s 283us/sample - loss: 0.1978 - acc: 0.9304
Epoch 10/32
2355/2355 [==============================] - 1s 286us/sample - loss: 0.1727 - acc: 0.9384
Epoch 11/32
2355/2355 [==============================] - 1s 271us/sample - loss: 0.1521 - acc: 0.9473
Epoch 12/32
2355/2355 [==============================] - 1s 271us/sample - loss: 0.1410 - acc: 0.9507
Epoch 13/32
2355/2355 [==============================] - 1s 268us/sample - loss: 0.1501 - acc: 0.9469
Epoch 14/32
2355/2355 [==============================] - 1s 272us/sample - loss: 0.1283 - acc: 0.9605
Epoch 15/32
2355/2355 [==============================] - 1s 271us/sample - loss: 0.1104 - acc: 0.9622
Epoch 16/32
2355/2355 [==============================] - 1s 269us/sample - loss: 0.1081 - acc: 0.9648
Epoch 17/32
2355/2355 [==============================] - 1s 274us/sample - loss: 0.1046 - acc: 0.9614
Epoch 18/32
2355/2355 [==============================] - 1s 272us/sample - loss: 0.0957 - acc: 0.9669
Epoch 19/32
2355/2355 [==============================] - 1s 273us/sample - loss: 0.0991 - acc: 0.9660
Epoch 20/32
2355/2355 [==============================] - 1s 268us/sample - loss: 0.0918 - acc: 0.9694
Epoch 21/32
2355/2355 [==============================] - 1s 267us/sample - loss: 0.0837 - acc: 0.9754
Epoch 22/32
2355/2355 [==============================] - 1s 265us/sample - loss: 0.0778 - acc: 0.9741
Epoch 23/32
2355/2355 [==============================] - 1s 276us/sample - loss: 0.0764 - acc: 0.9749
Epoch 24/32
2355/2355 [==============================] - 1s 270us/sample - loss: 0.0689 - acc: 0.9762
Epoch 25/32
2355/2355 [==============================] - 1s 271us/sample - loss: 0.0594 - acc: 0.9788
Epoch 26/32
2355/2355 [==============================] - 1s 268us/sample - loss: 0.0641 - acc: 0.9813
Epoch 27/32
2355/2355 [==============================] - 1s 268us/sample - loss: 0.0588 - acc: 0.9800
Epoch 28/32
2355/2355 [==============================] - 1s 273us/sample - loss: 0.0549 - acc: 0.9822
Epoch 29/32
2355/2355 [==============================] - 1s 272us/sample - loss: 0.0537 - acc: 0.9826
Epoch 30/32
2355/2355 [==============================] - 1s 270us/sample - loss: 0.0529 - acc: 0.9839
Epoch 31/32
2355/2355 [==============================] - 1s 267us/sample - loss: 0.0480 - acc: 0.9864
Epoch 32/32
2355/2355 [==============================] - 1s 274us/sample - loss: 0.0461 - acc: 0.9856
1161/1161 [==============================] - 0s 139us/sample - loss: 0.0345 - acc: 0.9940
```
