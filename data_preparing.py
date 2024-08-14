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


# Read the dataset files
images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

my_cnn = cnn.CNN(dense_hidden_size=(128, 64), max_iter=1000,
              alpha=0.0001, solver='adam', random_state=21,
              learning_rate=0.001)

my_cnn.train(images, labels)

print(my_cnn.predict(images[0]))