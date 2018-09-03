import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow Version", tf.__version__)

# download imdb dataset
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# print(train_data[0], "show how data is saved as integers instead of words")

# print(len(train_data[0]), len(train_data[1]), "data entries don't have same length")

# a dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# the first indices are reversed
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# decode integers back into readable text
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_review(train_data[0]))  # print out first review

# make the reviews the same length, add padding at the end -> word_index["<PAD>"] = 0
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],
                                                        padding='post', maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],
                                                        padding='post', maxlen=256)

# check changing length worked
# print(len(train_data[0]), "length train data [0]", len(test_data[0]), "length test data [0]")
# print(train_data[0])

# build the model
# input shape is the vocabulary count used for the movie reviews (10'000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# configure the model
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train the model
history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=40,
                    verbose=1, validation_data=(x_val, y_val))

# evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)

# history object returned by model.fit() contains dict
history_dict = history.history
# print(history_dict.keys(), "Keys in history dict")

# start plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# plot losses
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot accuracy
plt.clf()  # clear figure
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
