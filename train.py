import os
import cv2
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from augmentation import cut_out
from image_data_generator import MixUpImageDataGenerator
from image_data_generator import CutMixImageDataGenerator
# https://github.com/keras-team/keras/issues/17199
# from keras.applications import EfficientNetB3 have bugs
from networks.efficientnet import EfficientNetB3
from vit_keras.vit import vit_b32


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
    "logs": "logs/",
    "report": "reports/" + MODEL_PATH + ".csv",
    "augmented_images": "augmented_images/"
}


def resize_img(img, shape):
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def load_data():
    labels = pickle.load(open(PATHS["cifar-100-labels"], "rb"))
    (X_train_valid, y_train_valid), (X_test, y_test) = cifar100.load_data()
    resized_X_train_valid = np.array([resize_img(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in X_train_valid])
    resized_X_test = np.array([resize_img(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in X_test])
    resized_X_train, resized_X_valid, y_train, y_valid = train_test_split(resized_X_train_valid, y_train_valid,
                                                                          test_size=0.3, random_state=42)
    y_train = to_categorical(y_train, len(labels))
    y_valid = to_categorical(y_valid, len(labels))
    y_test = to_categorical(y_test,  len(labels))
    return resized_X_train, y_train, resized_X_valid, y_valid, resized_X_test, y_test, labels


def show_images(X, y, labels, indices, n_images):
    plt.suptitle("Unaugmented")
    for i, idx in enumerate(indices):
        plt.subplot(1, n_images, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[idx], cmap="gray")
        plt.xlabel(labels[np.argmax(y[idx])])
    plt.tight_layout()
    plt.savefig(PATHS["augmented_images"] + "unaugmented.png")


def show_augmented_images(X, y, labels, indices, n_images, augmentation):
    origin_images = X[indices]
    origin_labels = y[indices]
    if augmentation == "Cut-Out":
        img_datagen = ImageDataGenerator(rescale=1/255,
                                         preprocessing_function=lambda input_img: cut_out(input_img, p=1))
    elif augmentation == "Cut-Mix":
        img_datagen = CutMixImageDataGenerator(rescale=1/255, p=1)
    elif augmentation == "Mix-Up":
        img_datagen = MixUpImageDataGenerator(rescale=1/255, p=1)
    else:
        img_datagen = ImageDataGenerator(rescale=1/255)
    augmented_images = []
    augmented_labels = []
    for batch in img_datagen.flow(origin_images, origin_labels, batch_size=n_images, shuffle=False):
        augmented_images, augmented_labels = batch[0], batch[1]
        break
    plt.suptitle(augmentation)
    for i, image in enumerate(augmented_images):
        plt.subplot(1, n_images, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap="gray")
        if augmentation == "Cut-Mix":
            if labels[np.argmax(augmented_labels[i])] != labels[np.argmax(origin_labels[i])]:
                plt.xlabel(labels[np.argmax(augmented_labels[i])] + " (" + labels[np.argmax(origin_labels[i])] + ")")
            else:
                plt.xlabel(labels[np.argmax(augmented_labels[i])])
        elif augmentation == "Mix-Up":
            if labels[np.argmax(augmented_labels[i])] != labels[np.argmax(origin_labels[i])]:
                plt.xlabel(labels[np.argmax(augmented_labels[i])] + " (" + labels[np.argmax(origin_labels[i])] + ")")
            else:
                plt.xlabel(labels[np.argmax(augmented_labels[i])])
        else:
            plt.xlabel(labels[np.argmax(augmented_labels[i])])
    plt.tight_layout()
    plt.savefig(PATHS["augmented_images"] + augmentation + ".png")


def run_data_augmentation(X_train, y_train, X_valid, y_valid, X_test, y_test, augmentation=None):
    if augmentation == "Cut-Out":
        tr_datagen = ImageDataGenerator(rescale=1/255,
                                        preprocessing_function=lambda input_img: cut_out(input_img))
    elif augmentation == "Cut-Mix":
        tr_datagen = CutMixImageDataGenerator(rescale=1/255)
    elif augmentation == "Mix-Up":
        tr_datagen = MixUpImageDataGenerator(rescale=1/255)
    else:
        tr_datagen = ImageDataGenerator(rescale=1/255)
    val_te_datagen = ImageDataGenerator(rescale=1/255)
    train_data = tr_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_data = val_te_datagen.flow(X_valid, y_valid, batch_size=BATCH_SIZE, shuffle=True)
    test_data = val_te_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=True)
    return train_data, val_data, test_data


def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def compile_model(n_classes):
    if not os.path.exists(PATHS["checkpoint"]):
        net = vit_b32(
            image_size=IMAGE_SIZE,
            activation="softmax",
            pretrained=True,
            pretrained_top=False,
            include_top=False,
            classes=n_classes)
        # net = EfficientNetB3(
        #     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        #     weights="imagenet",
        #     include_top=False,
        #     classes=n_classes
        # )
        x = net.output
        # x = GlobalAveragePooling2D()(x)
        # x = Dropout(0.5)(x)
        predictions = Dense(n_classes, activation="softmax")(x)
        model = Model(inputs=net.input, outputs=predictions)
        optimizer = Adam(learning_rate=1e-5)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy", acc_top5])
    else:
        model = load_model(PATHS["checkpoint"], custom_objects={"acc_top5": acc_top5})
        print("Checkpoint Model Loaded")
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(PATHS["checkpoint"], monitor="val_accuracy", save_best_only=True,
                                 verbose=1, save_weights_only=False)
    lr = ReduceLROnPlateau(monitor="val_loss", mode="min", min_lr=1e-7, patience=10)
    csv_logger = CSVLogger(PATHS["log"])
    tensor_board = TensorBoard(log_dir=PATHS["logs"])
    print(model.summary())
    return model, early_stopping, checkpoint, lr, csv_logger, tensor_board


def train(X_train, y_train, X_valid, y_valid, X_test, y_test, labels, augmentation=None):
    train_data, val_data, test_data = run_data_augmentation(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                            augmentation)
    model, early_stopping, checkpoint, lr, csv_logger, tensor_board = compile_model(len(labels))
    history = model.fit(train_data, epochs=300,
                        validation_data=val_data,
                        callbacks=[early_stopping, checkpoint, lr, csv_logger, tensor_board],
                        steps_per_epoch=math.ceil(len(X_train) / BATCH_SIZE),
                        batch_size=BATCH_SIZE,
                        validation_steps=math.ceil(len(X_valid) / BATCH_SIZE),
                        validation_batch_size=BATCH_SIZE)

    train_score = model.evaluate(train_data, batch_size=BATCH_SIZE, steps=math.ceil(len(X_train) / BATCH_SIZE))
    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])

    val_score = model.evaluate(val_data)
    print("Validation Loss: ", val_score[0])
    print("Validation Accuracy: ", val_score[1])

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

    plt.figure(figsize=(12, 8))
    plt.title("EVALUATION")
    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Val_Loss")
    plt.legend()
    plt.title("Loss Evaluation")
    plt.subplot(2, 2, 2)
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val_Accuracy")
    plt.legend()
    plt.title("Accuracy Evaluation")
    plt.savefig(PATHS["figures"])


if __name__ == "__main__":
    X_tr, y_tr, X_val, y_val, X_te, y_te, lbs = load_data()
    # n_ims = 5
    # ids = random.sample(range(len(X_tr)), n_ims)
    # show_images(X=X_tr, y=y_tr, labels=lbs, indices=ids, n_images=n_ims)
    # augmentations = ["cut_out", "cut_mix", "mix_up"]
    # for aug in augmentations:
    #     show_augmented_images(X=X_tr, y=y_tr, labels=lbs, indices=ids, n_images=n_ims, augmentation=aug)
    train(X_tr, y_tr, X_val, y_val, X_te, y_te, lbs, AUGMENTATION)
