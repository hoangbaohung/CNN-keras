import keras
import cv2
import numpy as np
from keras import models

#from keras.models import load_model
#model = load_model('hung.h5')
model = keras.models.load_model('hung.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
img = cv2.imread('101500.jpg')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])
classes = model.predict_proba(img)
print(classes)