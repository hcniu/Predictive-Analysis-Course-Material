import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import LSTM, Dense, Masking,TimeDistributed

x = tf.random.normal((100, 10, 2))
layer = SimpleRNN(5, input_shape=(10, 2))
output = layer(x)
print(output.shape)


model1 = Sequential()
model1.add(SimpleRNN(5, input_shape=(10, 2)))
model1.add(Dense(1))
output1 = model1(x)
print(output1.shape)


model2 = Sequential()
model2.add(SimpleRNN(5, input_shape=(10, 2),
                    return_sequences=True))
output2 = model2(x)
print(output2.shape)



model3 = Sequential()
model3.add(SimpleRNN(5, input_shape=(10, 2),
                    return_sequences=True))
model3.add(TimeDistributed(Dense(10, activation='softmax')))
output3 = model3(x)
print(output3.shape)


model4 = Sequential()
model4.add(SimpleRNN(5, input_shape=(10, 2), return_sequences=True))
model4.add(SimpleRNN(5))
output4 = model4(x)
print(output4.shape)