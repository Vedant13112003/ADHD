import warnings
warnings.filterwarnings("ignore")
#Since your current working directory is already inside the Code folder, simply change your import line in adhd_fmri_cv.py from:
#from Code.data_generator import FMRIDataGenerator
from data_generator import FMRIDataGenerator

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

import tensorflow as tf

from tensorflow.keras.layers import Conv3D, MaxPool3D, TimeDistributed, Flatten, LSTM, Dense
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger

import tensorflow.keras as keras

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# ============================ DATA WORK ============================

file_num = sys.argv[1]

# Dataframes
dataset_dir = "/Users/vedantshirapure/Desktop/University/Capestone Project /Diagnosing-ADHD-With-ConvLSTM/model_data/"
model_train_data = pd.read_csv("/Users/vedantshirapure/Desktop/University/Capestone Project /Diagnosing-ADHD-With-ConvLSTM/Data/training_data_{}".format(file_num) )
model_val_data = pd.read_csv("/Users/vedantshirapure/Desktop/University/Capestone Project /Diagnosing-ADHD-With-ConvLSTM/Data/validatation_data_{}".format(file_num) )

# Dictionary of data values
partition = {'train': model_train_data['Image'].values, 
             'validation': model_val_data['Image'].values}

# Training Data
train_labels = {}
for index, row in model_train_data.iterrows():
    train_labels[row['Image']] = row['DX']
    
# Validation Data
val_labels = {}
for index, row in model_val_data.iterrows():
    val_labels[row['Image']] = row['DX']

# ============================ MODEL META ============================

epochs = 100
batch_size = 4
input_shape=(177,28,28,28,1)

train_steps_per_epoch = model_train_data.shape[0] // batch_size
validate_steps_per_epoch = model_val_data.shape[0] // batch_size

# Generators
training_generator = FMRIDataGenerator(partition['train'], train_labels, dataset_dir, batch_size)
validation_generator = FMRIDataGenerator(partition['validation'], val_labels, dataset_dir, batch_size)

curr_time = f'{datetime.now():%H-%M-%S%z_%m%d%Y}'

# Ensure the Saved_Models directory exists clearly
os.makedirs("/Users/vedantshirapure/Desktop/University/Capestone Project /Diagnosing-ADHD-With-ConvLSTM/Saved_Models", exist_ok=True)
logger_path = "/Users/vedantshirapure/Desktop/University/Capestone Project /Diagnosing-ADHD-With-ConvLSTM/Saved_Models/adhd-fmri-history_cv{num}_{time}.csv".format(num=file_num,time=curr_time)

csv_logger = CSVLogger(logger_path, append=True)

callbacks = [csv_logger]

# ============================ MODEL ARCHITECTURE ============================

with tf.device('/gpu:0'):
    cnn_lstm_model = Sequential()

    cnn_lstm_model.add(TimeDistributed(Conv3D(filters=64,kernel_size=(3,3,3),activation='relu'),
                                  input_shape=input_shape, name="Input_Conv_Layer"))

    cnn_lstm_model.add(TimeDistributed(MaxPool3D(
                                    pool_size=(2, 2, 2),
                                    strides=(2, 2, 2),
                                    padding='valid'
                                    ), name="Pool_Layer_1"))

    cnn_lstm_model.add(TimeDistributed(Flatten(), name="Flatten_Layer"))
    
with tf.device('/cpu:0'):

    cnn_lstm_model.add(LSTM(10, dropout = 0.3, recurrent_dropout = 0.3, name="LSTM_Layer"))

with tf.device('/gpu:0'):

    cnn_lstm_model.add(Dense(1, activation = 'sigmoid', name="Output_Dense_Layer"))

    cnn_lstm_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])


cnn_lstm_model.fit(training_generator,
                   steps_per_epoch=train_steps_per_epoch,
                   verbose=1,
                   callbacks=callbacks,
                   validation_data=validation_generator,
                   validation_steps=validate_steps_per_epoch,
                   epochs=epochs)
