import numpy as np
import utils as ut

class FeatureMap:
    def __init__(self, filters_count, filter_size):
        self.filters_count = filters_count
        self.filter_size = filter_size
        filters_rand = np.random.randn(filters_count, filter_size, filter_size)
        self.filters = (filters_rand - filters_rand.min()) / (filters_rand.max() - filters_rand.min())
        self.feature_maps = None
        
    def forward(self, input):
        
        f_Maps = []
        
        
        for i in range(len(self.filters)):
              f_Map = []
              
              
              for img_i in range(input.shape[0] - self.filter_size + 1):
                  row = []
                  
                  for img_j in range(input.shape[1] - self.filter_size + 1):
                      patch = input[img_i:img_i+ self.filter_size, img_j:img_j + self.filter_size]
                      dp = ut.dot_product(patch ,self.filters[i]) 
                      row.append(dp) 
                  
                 
                  f_Map.append(row)   
               
              f_Maps.append(f_Map)                     
                
        
        self.feature_maps = np.array(f_Maps)
        
        return self.feature_maps      
    
    
    
    