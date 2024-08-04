import numpy as np
import utils as ut
import math

class AveragePool:
    def __init__(self, patch_size):
        self.feature_maps = None
        self.patch_size = patch_size
        self.patches = None
    
    def forward(self, feature_maps):
        patches = []
        
        self.feature_maps = feature_maps
        
        for i in range(len(self.feature_maps)):
            subarray = ut.extract_subarrays(self.feature_maps[i], self.patch_size)
            patched_average = []
            
            for j in range(len(subarray)):
                patched_average.append(np.mean(subarray[j]))
            
            
            patched_average = np.array(patched_average)
            
            flatten_patch_length = len(patched_average)
            
            patches.append(patched_average.reshape(int(math.sqrt(flatten_patch_length)), int(math.sqrt(flatten_patch_length))))
        
        self.patches = np.array(patches)
            
        return self.patches   
   
    