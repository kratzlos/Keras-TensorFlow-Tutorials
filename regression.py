from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("TensorFlow Version", tf.__version__)

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))
print("Test set: {}".format(test_data.shape))

# print(train_data[0], "Notice the different scales")

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
# print(df.head())

# print(train_labels[0:10], "first 10 entries")  # display the first 10 entries

# normalise features -> subtract mean and divide by standard deviation

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# print(train_data[0], "first entry normalised")


# build the model
def build_model():
    model = keras.Sequential([keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
                              keras.layers.Dense(64, activation=tf.nn.relu),
                              keras.layers.Dense(1)])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    return model


model = build_model()
print(model.summary())


# display trainig progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 500

# stop training when the score doesn't improve after a set batch of epochs
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# store training data sets
history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2,
                    verbose=0, callbacks=[early_stop, PrintDot()])


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean abs error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='Validation loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


plot_history(history)

# evaluate model on test set
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Test set Mean Absolute Error: ${:7.2f}".format(mae*1000))

# predicting housing prices
test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [$1000]')
plt.ylabel('Predictions [$1000]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# plot the prediction errors
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction error [$1000]")
_ = plt.ylabel("Count")
plt.show()
