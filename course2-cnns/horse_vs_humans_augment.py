from pathlib import Path
import tensorflow as tf

print(tf.__version__)


base_dir = Path.home() / 'Downloads/lesson3/'
train_dir = base_dir / 'horse-or-human'
valid_dir = base_dir / 'validation-horse-or-human'

train_horse_dir = train_dir / 'horses'
train_human_dir = train_dir / 'humans'
valid_horse_dir = valid_dir / 'horses'
valid_human_dir = valid_dir / 'humans'

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=40,  # rotate up to 40 degree
                                   width_shift_range=0.2,  # width-orient shift pct
                                   height_shift_range=0.2,  #height-orient shift pct
                                   shear_range=0.2,  # 비트는 정도 pct
                                   zoom_range=0.2,  # 확대 정도 pct
                                   horizontal_flip=True,  # 좌우반전
                                   fill_mode='nearest'  # 변형과전에서 누락되는 픽셀은 가장 가까운것 interpolate
                                   )
validation_datagen = ImageDataGenerator(rescale=1/255)  # Note that the validation data should not be augmented!

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir.resolve(),  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=64,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        valid_dir.resolve(),  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

if __name__ == '__main__':
    history = model.fit(
          train_generator,
          steps_per_epoch=16,
          epochs=100,
          verbose=1,
          validation_data = validation_generator,
          validation_steps=8,
          use_multiprocessing=True,
          workers=12)

    import matplotlib.pyplot as plt

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()