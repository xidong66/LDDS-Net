import numpy as np
from tensorflow.keras.utils import to_categorical
from utils import encode_labels
def load_data(path):
    data = np.load(path, allow_pickle=True)
    return (
        data['ecgstrain'], data['labelstrain'],
        data['ecgsval'], data['labelsval'],
        data['ecgstest'], data['labelstest']
    )

def preprocess_data(y_train, y_val, y_test):
    y_train = encode_labels(y_train)
    y_test = encode_labels(y_test)
    y_val= encode_labels(y_val)
    y_train = to_categorical(y_train, 4)
    y_val = to_categorical(y_val, 4)
    y_test = to_categorical(y_test, 4)
    return y_train, y_val, y_test