import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

cnn_filter_1 = np.array([[0.9345, 0.5793, 0.1394],
                      [0.7656, 0.4325, 0.0246],
                      [0.3123, 0.1675, 0.8464]]);
cnn_filter_2 = np.array([[0.7385, 0.2453, 0.7894],
                      [0.7656, 0.4325, 0.4462],
                      [0.2423, 0.4578, 0.0413]]);
cnn_filter_3 = np.array([[0.3456, 0.0793, 0.9992],
                      [0.7626, 0.3425, 0.1323],
                      [0.3823, 0.7875, 0.1364]]);

def dot_product_sum(matrix1, matrix2):
    dot_product = np.dot(matrix1, matrix2)
    return np.sum(dot_product)

def get_product_image(mnist_image, image_filter):
    img_row, img_col = mnist_image.shape
    f_row, f_col = image_filter.shape
    out_rows = img_row - f_row + 1
    out_cols = img_col - f_col + 1
    
    result = np.zeros((out_rows, out_cols))
    
    for i in range(out_rows):
        for j in range(out_cols):
            # Extract the region of interest from the input matrix
            roi = mnist_image[i:i+f_row, j:j+f_col]
            # Perform element-wise multiplication with the filter and sum the result
            result[i, j] = np.sum(roi * image_filter)
            
    return result        
    
def angle_between_vectors(vector1, vector2):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)
    
    # Calculate the magnitudes (norms) of the vectors
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)
    
    # Calculate the cosine of the angle between the vectors
    cosine_angle = dot_product / (magnitude_vector1 * magnitude_vector2)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees
            
# Path to your MNIST dataset files
images_file = 'train-images.idx3-ubyte'
labels_file = 'train-labels.idx1-ubyte'

# Read the dataset files
images = idx2numpy.convert_from_file(images_file)
labels = idx2numpy.convert_from_file(labels_file)



# Print the shape of the arrays
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

image_number = 0

image_array = images[image_number]
image_label = labels[image_number]

# # Pass image through few filters in order to get pattern
image_product = get_product_image(image_array, cnn_filter_1)
image_product = get_product_image(image_product, cnn_filter_2)
image_product = get_product_image(image_product, cnn_filter_3)


# arr1 = np.array([1,4,6,7,2,3,4,1,5,6,8,9])

# arr2 = np.array([5,2,1,7,8,4])


# print(angle_between_vectors(arr1, arr2))

# Flatten image

# flatten_image_product = image_product.flatten()

# angles_arr = np.array([])

# for img in images:
#     angles_arr = np.append(angles_arr, angle_between_vectors(flatten_image_product, img.flatten()))
    
# plt.imshow(image_array, cmap='gray')
# plt.title("Label: {}".format(labels[image_number]))

# plt.imshow(image_product, cmap='gray')
# plt.title('Product Label: {}'.format(labels[image_number]))

# plt.show()