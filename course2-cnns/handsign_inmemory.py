import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd


def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        line = 0
        labels = []
        images = []
        for l, row in enumerate(csv_reader):
            if l == 0:
                continue
            labels.append(row[0])
            images.append(np.array_split(row[1:], 28))
        labels = np.array(labels).astype(np.float)
        images = np.array(images).astype(np.float)
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# In this section you will have to add another dimension to the data
# So, for example, if your array is (10000, 28, 28)
# You will need to make it (10000, 28, 28, 1)
# Hint: np.expand_dims

training_images = np.expand_dims(training_images, -1)
testing_images = np.expand_dims(testing_images, -1)

# Create an ImageDataGenerator and do Image Augmentation
from tensorflow.keras.utils import to_categorical

training_labels = to_categorical(training_labels)
testing_labels = to_categorical(testing_labels)

train_datagen = ImageDataGenerator(
    # Your Code Here
    rescale=1. / 255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow(training_images,
                                     training_labels,
                                     batch_size=126)

validation_datagen = ImageDataGenerator(rescale=1 / 255)
validation_generator = validation_datagen.flow(testing_images,
                                               testing_labels,
                                               batch_size=126)
# Your Code Here)

# Keep These
print(training_images.shape)
print(testing_images.shape)

# Their output should be:
# (27455, 28, 28, 1)
# (7172, 28, 28, 1)

# Define the model
# Use no more than 2 Conv2D and 2 MaxPooling2D
model = tf.keras.models.Sequential([
    # Your Code Here
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(25, activation='softmax')]
    )

# Compile Model.
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the Model
history = model.fit_generator(train_generator,
                              validation_data= validation_generator,
                              epochs=2,
                              verbose=1)

model.evaluate(testing_images/255, testing_labels, verbose=0)