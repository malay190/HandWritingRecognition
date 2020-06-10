# svm tries to maximise the minimum distance between the two different classes

import warnings
warnings.filterwarnings(action='ignore')
# stander scintific python import
import matplotlib.pyplot as plt
import numpy as np
# import datasets
from  sklearn import datasets, svm

# the digit data set
digits = datasets.load_digits()
print(type(digits))
print("digits : ", digits.keys())
print("digits target ---------", digits.target)
print("digits target names---------", digits.target_names)
# print("digits images ---------", digits.images)
# print("digits DESCR ---------", digits.DESCR)


images_and_labels  = list(zip(digits.images, digits.target))

print("len(images_and_labels) :", len(images_and_labels))

for index, [image, label] in enumerate(images_and_labels[: 5]):

    print("index : ", index, "image : \n", image, "label :", label)
    plt.subplot(2, 5, index+1)
    plt.axis('on')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('training : %i '% label)

# plt.show()

# to apply a classifier on this data, we need to flatten the image to
# turn the data in a (sample, feature) matrix:
n_sample = len(digits.images)  # n_sample = 1797
# print("n_sample = ", n_sample)
print(digits.images.shape)
imageData = digits.images.reshape((n_sample, -1))
print(imageData.shape)
print("after reshape : len(imageData[0]) :  ", len(imageData[0]))  # after reshape : len(imageData[0]) :   64

# create a classifier : a support vector classifier
classifier = svm.SVC(gamma=0.001)

# we learn the digits on the first half of the digits
classifier.fit(imageData[:n_sample//2], digits.target[:n_sample//2])

# now predict the value of the digit on the second half :
expectedY = digits.target[n_sample//2:]

predictedY = classifier.predict(imageData[n_sample//2: ])

images_and_prediction = list(zip(digits.images[n_sample//2:], predictedY))
for index, [image, prediction] in enumerate(images_and_prediction[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('on')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("predict. : %i" % prediction)

print("original values : ", digits.target[n_sample//2 : (n_sample//2)+6])
# plt.show()


# Install pillow library

from scipy.misc import imread, imresize, bytescale
img = imread("seven2.jpeg")
print('image shape = ',img.shape)
img = imresize(img, (8, 8))
classifier = svm.SVC(gamma=0.001)
classifier.fit(imageData[:], digits.target[:])
img = img.astype(digits.images.dtype)
img = bytescale(img, high=16.0, low=0)
x_testData = []
for c in img:
    for r in c:
        x_testData.append(sum(r)/3.0)
print("x_testData : \n", x_testData)
print("x_testData len : \n", len(x_testData))
x_testData = [x_testData]
print("Machine Output = ", classifier.predict(x_testData))
plt.show()

