import numpy as np
                        
def extract_subarrays(arr, sub_shape):
    subarrays = []
        
    for i in range(0, arr.shape[0], sub_shape):
        for j in range(0, arr.shape[1], sub_shape):
            subarray = arr[i:i + sub_shape, j:j + sub_shape]
            subarrays.append(subarray)
                
    return subarrays 
    
        
def dot_product(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible dimensions for matrix multiplication.")
        
    return np.sum(np.multiply(A, B))