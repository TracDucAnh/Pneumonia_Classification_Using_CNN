import cv2
import os
import sys
import numpy as np
import tensorflow as tf

IMG_HEIGHT = 500
IMG_WIDTH = 500
EPOCHS = 10

def main():
    #check command-line arguments
    if len(sys.argv) not in [2,3]:
        sys.exit("Usage: python Pneumonia_Detector.py dataset_directory [model.h5]")

    x_rays_train, labels_train, x_rays_test, labels_test = load_data(sys.argv[1])

    x_rays_train = np.array(x_rays_train)/255.0
    labels_train = tf.keras.utils.to_categorical(np.array(labels_train))
    x_rays_test = np.array(x_rays_test)/255.0
    labels_test = tf.keras.utils.to_categorical(np.array(labels_test))

    model = get_model()

    model.fit(x_rays_train, labels_train, epochs = EPOCHS)

    model.evaluate(x_rays_test, labels_test, verbose = 2)

    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(dataset_dir):
    train_data_path = os.path.join(dataset_dir, "xray_dataset", "train")
    test_data_path = os.path.join(dataset_dir, "xray_dataset", "test")

    x_rays_train = []

    labels_train = []

    for folder in os.listdir(train_data_path):
        new_path = os.path.join(train_data_path, folder)
        for file in os.listdir(new_path):
            file_path = os.path.join(new_path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            x_rays_train.append(img)
            if folder == "NORMAL":
                labels_train.append(0)
            else:
                labels_train.append(1)

    x_rays_test = []

    labels_test = []

    for folder in os.listdir(test_data_path):
        new_path = os.path.join(test_data_path, folder)
        for file in os.listdir(new_path):
            file_path = os.path.join(new_path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            x_rays_test.append(img)
            if folder == "NORMAL":
                labels_test.append(0)
            else:
                labels_test.append(1)

    return x_rays_train, labels_train, x_rays_test, labels_test

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            25, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Conv2D(
            25, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(130, activation="relu"),

        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(50, activation="relu"),

        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

if __name__ == "__main__":
    main()
