import csv

import numpy as np
import cv2
import pandas as pd
from sklearn.svm import SVC
from os import listdir
from os.path import isfile, join

basepath_train = 'D:/ImageNet/dataset/training_set/second_attempt/'
basepath_test = 'D:/ImageNet/dataset/test_set/second_attempt/'

X_train = []
y_train = []
X_test = []
y_test = []
sift = cv2.xfeatures2d.SIFT_create()

products = ['apple', 'peach', 'strawberry', 'banana', 'tomato', 'potato', 'onion', 'cucumber', 'cocoa', 'coffee']

for i in range(len(products)):
    traindir = basepath_train + products[i]
    for f in listdir(traindir):
        if isfile(join(traindir, f)):
            image = cv2.imread(traindir + '/' + f)
            kp, descriptor = sift.detectAndCompute(image, None)
            if descriptor is not None:
                X_train.append(descriptor[0])
                y_train.append(i)

    testdir = basepath_test + products[i]
    for f in listdir(testdir):
        if isfile(join(testdir, f)):
            image = cv2.imread(testdir + '/' + f)
            kp, descriptor = sift.detectAndCompute(image, None)
            if descriptor is not None:
                X_test.append(descriptor[0])
                y_test.append(i)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

print(len(X_test), len(y_test))
print(len(X_train), len(y_train))

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))