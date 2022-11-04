from tensorflow.keras import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import LSTM, Dense, Masking
import numpy as np


class MyBatchGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=1, shuffle=True):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *X[index].shape))
        yb = np.empty((self.batch_size, *y[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = X[index]
            yb[s] = y[index]
        return Xb, yb


# Parameters
N = 1000
halfN = int(N/2)
dimension = 2
lstm_units = 3

# Data
np.random.seed(123)  # to generate the same numbers
# create sequence lengths between 1 to 10
seq_lens = np.random.randint(1, 20, halfN)
X_zero = np.array([np.random.normal(0, 1, size=(seq_len, dimension)) for seq_len in seq_lens])
y_zero = np.zeros((halfN, 1))
X_one = np.array([np.random.normal(1, 1, size=(seq_len, dimension)) for seq_len in seq_lens])
y_one = np.ones((halfN, 1))
p = np.random.permutation(N)  # to shuffle zero and one classes
X = np.concatenate((X_zero, X_one))[p]
y = np.concatenate((y_zero, y_one))[p]

# Batch = 1
model = Sequential()
model.add(LSTM(lstm_units, input_shape=(None, dimension)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
model.fit_generator(MyBatchGenerator(X, y, batch_size=1), epochs=2)




# Padding and Masking
special_value = -10.0
max_seq_len = max(seq_lens)
Xpad = np.full((N, max_seq_len, dimension), fill_value=special_value)
for s, x in enumerate(X):
    seq_len = x.shape[0]
    Xpad[s, 0:seq_len, :] = x
model2 = Sequential()
model2.add(Masking(mask_value=special_value, input_shape=(max_seq_len, dimension)))
model2.add(LSTM(lstm_units))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model2.summary())
model2.fit(Xpad, y, epochs=50, batch_size=32)