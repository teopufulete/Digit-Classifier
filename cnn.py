import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import tensorflow as tf
from sklearn import preprocessing
import random

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

NUM_OF_TRAINING_SAMPLES = 60000
NUM_TESTING_SAMPLES = 10000
BATCH_SIZE = 512
NUM_OF_BATCHES = 1000
SEED = 420
EPOCHS = 2
random.seed(SEED)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

#Reads Data
data_test = pd.read_csv(mnist_test.csv')
data_train = pd.read_csv(mnist_train.csv')

y_train = np.array(data_train.iloc[:, 0])
x_train = np.array(data_train.iloc[:, 1:])
y_test = np.array(data_test.iloc[:, 0])
x_test = np.array(data_test.iloc[:, 1:])

x_train = x_train[:NUM_OF_TRAINING_SAMPLES]
x_test = x_test[:NUM_OF_TRAINING_SAMPLES]


x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)

x_train = np.array(x_train.reshape((NUM_OF_TRAINING_SAMPLES, 28,28,1)))
x_test = np.array(x_test.reshape((NUM_TESTING_SAMPLES, 28,28,1)))
                         
#NN BEGINS HERE---------------------
c1,n1,n2,desc = (1024,512, 64,"FF NEURAL NET WITH 1024 NEURONS ")
model = models.Sequential()
model.add(layers.Conv2D(c1, (3, 3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(c1, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(n1, activation='relu'))
model.add(layers.Dense(n2, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['acc'])
                         
for i in range(EPOCHS):
    c=0
    for batch in iterate_minibatches(x_train, y_train, 1024, shuffle=True):
        c+=1
        x_batch, y_batch = batch
        model.fit(x_batch,y_batch,epochs=1)

        if c%10 == 0:
            print("EPOCH: " + str(i))
            test_loss, test_acc = model.evaluate(x_test, y_test)
            print(f"MODEL TYPE: {desc}")
            print(f"TEST LOSS:{test_loss} TEST ACCURACY:{test_acc}")

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"MODEL TYPE: {desc}")
print(f"TEST LOSS:{test_loss} TEST ACCURACY:{test_acc}")
