import pandas as pd 
import os
from os.path import join
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def make_df(data_path):
	paths = []
	labels = []
	label_names = os.listdir(join(data_path))
	for dirname in label_names:
		for filename in os.listdir(join(data_path,dirname)):
			paths.append(join(data_path,dirname,filename))
			labels.append(dirname)

	train_data = {"path": paths, "label": labels}
	return pd.DataFrame(train_data), label_names

training_df, class_labels = make_df("dataset-resized/training")
val_df, _ = make_df("dataset-resized/validation")


total_train = training_df.shape[0]
total_val = val_df.shape[0]

print(total_train)
print(total_val)

#print(class_labels)

IMG_HEIGHT = 150
IMG_WIDTH = 150
batch_size = 16
epochs = 20

train_image_gen=ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_gen.flow_from_dataframe(dataframe=training_df,
											 x_col="path",
											 y_col="label",
											 batch_size=batch_size,
                                             shuffle=True,
                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                             class_mode='categorical',
                                             classes=class_labels)

print("train data gen type: ", type(train_data_gen))

val_image_gen=ImageDataGenerator(rescale=1./255)
val_data_gen = val_image_gen.flow_from_dataframe(dataframe=val_df,
										   x_col="path",
										   y_col="label",
                                           shuffle=False,
										   batch_size=batch_size,
                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                           class_mode='categorical',
                                           classes=class_labels)

sample_training_images, _ = next(train_data_gen)

#plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()