import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import os

resnet_model = resnet50.ResNet50(weights='imagenet')

desired_classes = open("desired_classes.txt").read().splitlines()
new_weights = np.zeros((2048,126), np.float32)
new_activations = np.zeros((126),np.float32)
class_number = 0

for i in range(0,1000):
    test = np.zeros((1,1000))
    test[0,i]=1
    if decode_predictions(test)[0][0][0] in desired_classes:
        new_weights[:,class_number] = resnet_model.layers[175].get_weights()[0][:,i]
        new_activations[class_number] = resnet_model.layers[175].get_weights()[1][i]
        class_number += 1

print("number of classes: ", class_number)
print("new_weights:", new_weights)
print("new_activations:", new_activations)

resnet_model.layers.pop()
x = resnet_model.layers[-1].output
x = Dense(126, activation='softmax', name='fc126')(x)
newmodel = Model(inputs=resnet_model.input, outputs=x)
newmodel.layers[175].set_weights([new_weights, new_activations])

original = load_img(os.path.join('dataset-resized-120/training/n02089973-English_foxhound',"n02089973_255.jpg"), target_size=(224, 224))

numpy_image = img_to_array(original)
#plt.imshow(np.uint8(numpy_image))
#plt.show()
#print('numpy array size',numpy_image.shape)
 
# Convert the image / images into batch format
# expand_dims will add an extra dimension to the data at a particular axis
# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# Thus we add the extra dimension to the axis 0.
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
#plt.imshow(np.uint8(image_batch[0]))

# prepare the image for the VGG model
processed_image = resnet50.preprocess_input(image_batch.copy())
 
# get the predicted probabilities for each class
predictions = newmodel.predict(processed_image)


print(predictions)


exit()







 
#Load the ResNet50 model
resnet_model.summary()

# this is the 1000-channel output
x = resnet_model.output

print(x)
os.exit()

num_channels = 1000
#for i in range(num_channels):


x = Dense(126, activation="softmax")(x)
modified_resnet = Model(input=resnet_model.input, output=x)
#print(x)
#resnet_model.add(Dense(10,activation='softmax'))

#print("success")
#resnet_model.summary()
os.exit(1)


filename = 'dataset-resized-120/n00000000-Cat/00000001_008.jpg'
for filename in os.listdir('dataset-resized-120/n00000000-Cat/'):
	print(filename)
	# load an image in PIL format
	original = load_img(os.path.join('dataset-resized-120/n00000000-Cat',filename), target_size=(224, 224))
	#print('PIL image size',original.size)
	#plt.imshow(original)
	#plt.show()

	# convert the PIL image to a numpy array
	# IN PIL - image is in (width, height, channel)
	# In Numpy - image is in (height, width, channel)
	numpy_image = img_to_array(original)
	#plt.imshow(np.uint8(numpy_image))
	#plt.show()
	#print('numpy array size',numpy_image.shape)
	 
	# Convert the image / images into batch format
	# expand_dims will add an extra dimension to the data at a particular axis
	# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
	# Thus we add the extra dimension to the axis 0.
	image_batch = np.expand_dims(numpy_image, axis=0)
	print('image batch size', image_batch.shape)
	#plt.imshow(np.uint8(image_batch[0]))

	# prepare the image for the VGG model
	processed_image = resnet50.preprocess_input(image_batch.copy())
	 
	# get the predicted probabilities for each class
	predictions = resnet_model.predict(processed_image)
	# print predictions
	 
	# convert the probabilities to class labels
	# We will get top 5 predictions which is the default
	label = decode_predictions(predictions)
	print(label)
