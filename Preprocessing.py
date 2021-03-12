import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# Get the dataset and the info for the dataset
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

# split dataset into out training and test set. mnist does not have a validation set
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

# Creating our Validation sets
# first decide how many validations we want as percentage of train sample
# in this case we compute 10 percent of the training set
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
# cast the value to an integer
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

# do it with the test set as well
# we could have computed this by hand but we are assuming that this is a generic code
# regardless of the dataset we can run this automatically with any dataset
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


# input layer contains grey scale value 0-255 we want it to be 0-1, this helps computations and speeds up process
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label


# apply our scale to all of our test validation and test data
scaled_train_and_validation__data = mnist_train.map(scale)
test_data = mnist_test.map(scale)


# We need to shuffle the data because we are implement mini batch gradient descent
# If it was just gradient descent we don't have to shuffle but since we using batches we want
# the data to be as shuffled as possible to have a batch that contains a close number of any target
# While we shuffle we keep in memory up to 10000 samples or less.
# We huge data sets we can't actually fit all the data into memory
BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation__data.shuffle(BUFFER_SIZE)

# Create validation set
# Take the number of validation samples calculated from our train and validation data 6000
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
# Skip the number of validation samples and take the remaining data let remaining 54000 samples left
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

# Create our batches
BATCH_SIZE = 120
train_data = train_data.batch(BATCH_SIZE)
# We don't need to batch but tensorflow is expecting a batch which is why we pass it to the validation and test data
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)
validation_inputs, validation_targets = next(iter(validation_data))


# Build Model
input_size = 784
output_size = 10
hidden_layer_size = 225

model = tf.keras.Sequential([
    # input layer, flatten will create tensor and order 28x28x1 data into 784x1
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # dense is applying output = activation(dot(input,weight) + bias)
    # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    # Second Hidden Layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),


    # output layer softmax for classification problem
    tf.keras.layers.Dense(output_size, activation='softmax')
])
early_stop = tf.keras.callbacks.EarlyStopping(patience=7)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
NUM_EPOCHS = 100
model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets),
          validation_steps=10, verbose=2, callbacks=early_stop)

test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

# homework = try different activation function, adding hidden layers Not a good idea use at least 2
#
