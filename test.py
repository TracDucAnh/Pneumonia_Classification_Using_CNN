import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from math import sqrt

if len(sys.argv) != 2:
    sys.exit("Usage: python test.py model")

model = tf.keras.models.load_model(sys.argv[1])
print("Successfully loaded the model")

def detector(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (500,500), interpolation=cv2.INTER_AREA)
    img =np.array(img)/255.0
    results = model.predict(img.reshape(1,500,500,1))
    print(img_path)
    if results.argmax() == 0:
        print("Normal lung detected")
        return("Normal lung detected")
    else:
        print("Pneumonia lung detected")
        return("Pneumonia lung detected")

path = os.getcwd()
path = os.path.join(path, "sample")

fig, axs = plt.subplots(int(sqrt(len(os.listdir(path)))), round(sqrt(len(os.listdir(path))))+1)


rows = int(sqrt(len(os.listdir(path))))
cols = round(sqrt(len(os.listdir(path))))+1

path = os.listdir(path)
index = 0

for i in range(rows):
    for j in range(cols):
        try:
            image = plt.imread(os.path.join("sample", path[index]))
        except:
            break
        axs[i,j].imshow(image)
        axs[i, j].set_title(detector(os.path.join("sample", path[index])))
        index += 1
plt.show()

