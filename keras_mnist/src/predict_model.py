from PIL import Image
import numpy as np
from skimage import transform

from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# use Keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# rescale to have values within 0 - 1 range [0,255] --> [0,1]
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
num_classes = 10

# one-hot encode the labels
# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# input image dimensions 28x28 pixel images.
img_rows, img_cols = 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


#Model Architeture
# build the model object
model = Sequential()
# CONV_1: add CONV layer with RELU activation and depth = 32 kernels
model.add(Conv2D(32, kernel_size=(3, 3), padding='same',activation='relu',input_shape=(28,28,1)))
# POOL_1: downsample the image to choose the best features
model.add(MaxPooling2D(pool_size=(2, 2)))
# CONV_2: here we increase the depth to 64
model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
# POOL_2: more downsampling
model.add(MaxPooling2D(pool_size=(2, 2)))
# flatten since too many dimensions, we only want a classification output
model.add(Flatten())
# FC_1: fully connected to get all relevant data
model.add(Dense(64, activation='relu'))
# FC_2: output a softmax to squash the matrix into output probabilities for the 10 classes
model.add(Dense(10, activation='softmax'))
#model.summary()

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.load_weights('../assets/model.weights.best.hdf5')
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]
# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (28, 28, 1))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image0 = load('../assets/zero.png')
image1 = load('../assets/um.png')
image2 = load('../assets/dois.png')
image4 = load('../assets/quatro.png')
image5 = load('../assets/cinco.png')
image9 = load('../assets/nove.png')

p0 = model.predict_classes(image0)
p1 = model.predict_classes(image1)
p2 = model.predict_classes(image2)
p3 = model.predict_classes(image4)
p4 = model.predict_classes(image5)
p5 = model.predict_classes(image9)

print('0 => ', p0)
print('1 => ', p1)
print('2 => ', p2)
print('4 => ', p3)
print('5 => ', p4)
print('9 => ', p5)
