# train.py

import numpy as np
from tensorflow.keras.utils import to_categorical
from preprocess import load_data
from model import build_model

# Load and preprocess the data
X_train, y_train = load_data('fer2013/train')
X_test, y_test = load_data('fer2013/test')

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the model
model = build_model(input_shape=(48, 48, 1), num_classes=7)

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

# Save the model
model.save('facial_expression_model.h5')
