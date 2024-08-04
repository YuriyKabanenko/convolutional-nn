import numpy as np
import feature_map as fmap
import avg_pool as avg
import idx2numpy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class CNN:
    
    def __init__(self, dense_hidden_size, max_iter, alpha, solver, 
                 learning_rate, random_state):
        self.dense_hidden_size = dense_hidden_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.solver = solver
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.flatten_patched_maps = []
        self.patched_maps = []
        self.feature_map = fmap.FeatureMap(5, 10)
        self.avg_pool = avg.AveragePool(4)
        self.classifier = None
        
    def train(self, images, labels):
        patched_maps = []

        for i in range(len(images)):
            img = images[i]
            maps = self.feature_map.forward(img)
            patches = self.avg_pool.forward(maps)
            patched_maps.append(patches[0])
        
        self.patched_maps = np.array(patched_maps)
        
        self.save_patches_to_file("patched-images.idx3-ubyte")
        
        patched_maps = np.array(patched_maps)
        
        self.flatten_patched_maps = patched_maps.reshape(patched_maps.shape[0], -1)
        
        x_train, x_test, y_train, y_test= train_test_split(self.flatten_patched_maps,labels, test_size=0.2, random_state=21)

        self.classifier = MLPClassifier(hidden_layer_sizes= self.dense_hidden_size,
                                        max_iter= self.max_iter, 
                                   alpha=self.alpha, solver=self.solver,
                                   random_state=self.random_state,
                                   learning_rate_init=self.learning_rate)

        self.classifier.fit(x_train, y_train)
        
    def save_patches_to_file(self, file_name):
        idx2numpy.convert_to_file(file_name, self.patched_maps)
        
        
    def predict(self, image):
        maps = self.feature_map.forward(image)
        patches = self.avg_pool.forward(maps)
       
        flatten_map = patches.reshape(patches.shape[0], -1)
        
        return self.classifier.predict(flatten_map)
        
        
        
        
        
        
        
        
        
        
        
        