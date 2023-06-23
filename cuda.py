import keras
import tensorflow as tf

from tensorflow.python.client import device_lib

if __name__ == "__main__":
    print(keras.__version__)
    print(tf.__version__)
    print(device_lib.list_local_devices())
