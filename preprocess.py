# preprocess.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_data(directory, image_size=(48, 48)):
    """
    Load and preprocess data from the given directory.
    """
    images = []
    labels = []
    label_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}
    
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, file)
                image = load_img(image_path, target_size=image_size, color_mode='grayscale')  # Load image in grayscale
                image = img_to_array(image)  # Convert to array
                images.append(image)
                labels.append(label_map.get(label, -1))  # Convert label to integer using the map
    
    return np.array(images), np.array(labels)
