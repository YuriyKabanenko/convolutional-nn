import idx2numpy
import pandas as pd
import numpy as np  
import pickle
import cnn
import os
from PIL import Image
import matplotlib.pyplot as plt

root_dataset_folder = 'dataset/test'

images_map = {}

def get_channels_array(img_array):
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2] 
    return np.stack([red_channel, green_channel, blue_channel])

for subdir_name in os.listdir(root_dataset_folder):
    img_array = []
    subdir_path = os.path.join(root_dataset_folder, subdir_name)
    subdir_path = subdir_path.replace('\\', '/')
    for file_name in os.listdir(subdir_path):
            file_path = subdir_path + '/' + file_name
            image = Image.open(file_path)
            img_array.append( get_channels_array(np.array(image)))  
    images_map[subdir_path] = img_array 

images = list(images_map.values())
labels = list(images_map.keys())

print( np.array(images[0]).shape) 


# # Read the dataset files
# images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
# labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

my_cnn = cnn.CNN(dense_hidden_size=(128, 64), max_iter=1000,
              alpha=0.0001, solver='adam', random_state=21,
              learning_rate=0.001)

# my_cnn.train(images, labels)

# print(my_cnn.predict(images[37247]))