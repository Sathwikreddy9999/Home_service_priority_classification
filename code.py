


import os
import math
import random
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#The code imports essential libraries 
# for data manipulation, analysis, visualization, 
# and machine learning using TensorFlow. 
# It prepares the environment to work with data, create plots,
# and build neural network models.

import os
import shutil

BASE_DIR = 'C:/Users/sathw/Desktop/lawns'
names = ["BAD_LAWN", "GOOD_lAWN", "OVERGROWN_LAWN"]

tf.random.set_seed(1)

if not os.path.isdir(os.path.join(BASE_DIR, 'train')):
    for name in names:
        os.makedirs(os.path.join(BASE_DIR, 'train', name))
        os.makedirs(os.path.join(BASE_DIR, 'val', name))
        os.makedirs(os.path.join(BASE_DIR, 'test', name))

orig_folders = ["0001/", "0002/", "0003/"]
for folder_idx, folder in enumerate(orig_folders):
    folder_path = os.path.join(BASE_DIR, folder)
    files = os.listdir(folder_path)
    number_of_images = len(files)
    n_train = int((number_of_images * 0.6) + 0.5)
    n_valid = int((number_of_images * 0.25) + 0.5)
    n_test = number_of_images - n_train - n_valid
    print(number_of_images, n_train, n_valid, n_test)
    for idx, file in enumerate(files):
        file_name = os.path.join(folder_path, file)
        if idx < n_train:
            shutil.move(file_name, os.path.join(BASE_DIR, "train", names[folder_idx]))
        elif idx < n_train + n_valid:
            shutil.move(file_name, os.path.join(BASE_DIR, "val", names[folder_idx]))
        else:
            shutil.move(file_name, os.path.join(BASE_DIR, "test", names[folder_idx]))


#In short, this code snippet organizes a collection of images into training, validation, 
# and testing sets within category-specific subdirectories. This is a common preprocessing 
# step before training a machine learning model on image data.

train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_batches = train_gen.flow_from_directory(
    'C:/Users/sathw/Desktop/lawns/train',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=True,
    color_mode="rgb",
    classes=names   
)

val_batches = valid_gen.flow_from_directory(
    'C:/Users/sathw/Desktop/lawns/val',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode="rgb",
    classes=names
)

test_batches = test_gen.flow_from_directory(
    'C:/Users/sathw/Desktop/lawns/test',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode="rgb",
    classes=names
)

#train_gen, valid_gen, test_gen: ImageDataGenerators for training, validation, and test data, with pixel normalization (0-1 range).

#train_batches, val_batches, test_batches: Batches of data from respective directories, resized to 256x256, in RGB format, with batch size of 4.

#Classification mode is 'sparse', class labels are derived from subdirectory names, and shuffling is applied to training data. Validation and test data are not shuffled.

train_batch = train_batches[0]
print(train_batch[0].shape)

print(train_batch[1])
test_batch = test_batches[0]

print(test_batch[0].shape)
print(test_batch[1])




def show(batch, pred_labels=None):
    plt.figure(figsize=(10,10))
    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch[0][i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        lbl = names[int(batch[1][i])]
        if pred_labels is not None:
            lbl += "/ Pred:" + names[int(pred_labels[i])]
        plt.xlabel(lbl)
    plt.show()

show(test_batch)



show(train_batch)


#train_batch = train_batches[0]: Retrieves first training batch.
#print(train_batch[0].shape): Displays training image shape.
#print(train_batch[1]): Prints training labels.
#test_batch = test_batches[0]: Retrieves first test batch.
#print(test_batch[0].shape): Displays test image shape.
#print(test_batch[1]): Prints test labels.
#def show(batch, pred_labels=None): Defines image display function.
#Displays images with labels using show() function.


model = keras.models.Sequential()



model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation='relu', input_shape=(256, 256,3)))




model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, 3, activation='relu'))




model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5))
print(model.summary())



# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)



epochs = 30


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=2
)

history = model.fit(train_batches, validation_data=val_batches,
                    callbacks=[early_stopping],
                      epochs=epochs, verbose=2)


model.save("LAWN_CLASSIFICATION")


#Defines loss, optimizer, and metrics.

#Compiles the model.

#Trains the model using training and validation data.

#Implements early stopping.

#Saves the trained model.
# plot loss and acc
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.grid()
plt.legend(fontsize=15)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='valid acc')
plt.grid()
plt.legend(fontsize=15);



# evaluate on test data
model.evaluate(test_batches, verbose=2)




# make some predictions
predictions = model.predict(test_batches)
predictions = tf.nn.softmax(predictions)
labels = np.argmax(predictions, axis=1)

print(test_batches[0][1])
print(labels[0:4])



show(test_batches[0], labels[0:4])




vgg_model = tf.keras.applications.vgg16.VGG16()
print(type(vgg_model))
vgg_model.summary()


model = keras.models.Sequential()
for layer in vgg_model.layers[0:-1]:
    model.add(layer)




model.summary()




# set trainable=False for all layers
# we don't want to train them again
for layer in model.layers:
    layer.trainable = False
model.summary()


model.add(layers.Dense(5))



# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)




# get the preprocessing function of this model
preprocess_input = tf.keras.applications.vgg16.preprocess_input



# Generate batches of tensor image data with real-time data augmentation.

train_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
valid_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_batches = train_gen.flow_from_directory(
    'C:/Users/sathw/Desktop/lawns/train',
    target_size=(224, 224),
    class_mode='sparse',
    batch_size=4,
    shuffle=True,
    color_mode="rgb",
    classes=names   
)

val_batches = valid_gen.flow_from_directory(
    'C:/Users/sathw/Desktop/lawns/val',
    target_size=(224, 224),
    class_mode='sparse',
    batch_size=4,
    shuffle=True,
    color_mode="rgb",
    classes=names
)

test_batches = test_gen.flow_from_directory(
    'C:/Users/sathw/Desktop/lawns/test',
    target_size=(224, 224),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode="rgb",
    classes=names
)



epochs = 30

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=2
)

model.fit(train_batches, validation_data=val_batches,
          callbacks=[early_stopping],
          epochs=epochs, verbose=2)



model.evaluate(test_batches, verbose=2)






# %%
