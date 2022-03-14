from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top = False, input_shape = (224,224,3))

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range =0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip =True, )

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        '/Users/christiangrinling/Desktop/Keras/Chess/train_dir',
        target_size = (224,224),
        batch_size = 20,
        class_mode = 'categorical',
        shuffle=True)

validation_generator = train_datagen.flow_from_directory(
            '/Users/christiangrinling/Desktop/Keras/Chess/validation_dir',
            target_size = (224,224),
            batch_size = 10,
            class_mode = 'categorical',
            shuffle=True)

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(4, activation='softmax'))

model.summary()
print(train_generator.class_indices)

model.compile(loss = 'categorical_crossentropy',
                    optimizer= optimizers.Adam(lr = 2e-5),
                    metrics = ['acc'])

for data_batch, labels_batch in train_generator:
     print(data_batch.shape)
     print(labels_batch.shape)
     break



history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps =50)

model.save('chess2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss= history.history['loss']
val_loss= history.history['val_loss']

epochs = range(1, len(acc) +1 )

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation acc')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
