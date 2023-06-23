import cv2
import pickle
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from vit_keras.vit import vit_b32  # keep this line for VIT model!


MODEL = "VitB32"
AUGMENTATION = "Cut-Mix"
MODEL_PATH = MODEL + "-{0}".format(AUGMENTATION)
BATCH_SIZE = 32
IMAGE_SIZE = 288  # 300
PATHS = {
    "cifar-100-labels": "cifar-100-labels.pkl",
    "checkpoint": "checkpoints/" + MODEL_PATH + ".h5",
    "figures": "figures/" + MODEL_PATH + ".png",
    "log": "logs/" + MODEL_PATH + ".log",
    "report": "reports/" + MODEL_PATH + ".csv",
    "augmented_images": "augmented_images/"
}


def resize_img(img, shape):
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def load_data():
    labels = pickle.load(open(PATHS["cifar-100-labels"], "rb"))
    (_X_train_valid, _y_train_valid), (X_test, y_test) = cifar100.load_data()
    resized_X_test = np.array([resize_img(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in X_test])
    y_test = to_categorical(y_test,  len(labels))
    return resized_X_test, y_test, labels


def run_data_augmentation(X_test, y_test):
    val_te_datagen = ImageDataGenerator(rescale=1/255)
    test_data = val_te_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=True)
    return test_data


def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def compile_model():
    model = load_model(PATHS["checkpoint"], custom_objects={"acc_top5": acc_top5})
    print("Checkpoint Model Loaded")
    print(model.summary())
    return model


def test(X_test, y_test, labels):
    test_data = run_data_augmentation(X_test, y_test)
    model = compile_model()

    test_score = model.evaluate(test_data)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])

    y_pred = model.predict(X_test/255., verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    test_score = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    print("Test Accuracy: ", test_score)

    target = [label for label in labels]
    report = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(PATHS["report"], index=True)


if __name__ == "__main__":
    X_te, y_te, lbs = load_data()
    test(X_te, y_te, lbs)
