import numpy as np
import scipy as sp
import math

class Convolution2D:

    def __init__(self, kernel_count, kernel_size, mode = 'valid'):
        
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size,)

        
        if type(kernel_count) is int and type(kernel_size) is tuple:
            self.kernel_shape = kernel_size
            self.count = kernel_count
        else:
            raise TypeError("Input parameter of convolution layer must be 'int' or 'tuple' (except mode)")

        self.kernel_weights = np.random.random_sample((kernel_count,) + kernel_size)
        self.mode = mode
        self.output = None
        self.output_size = None

    def Compute(self, input_image):

        image_size = input_image.shape

        if self.mode == 'valid':
            self.output_size = (self.count, image_size[1] - self.kernel_shape[0] + 1, image_size[2] - self.kernel_shape[1] + 1)
        elif self.mode == 'same':
            self.output_size = (self.count, image_size[1], image_size[2])
        else:
            raise NameError("Proper mode not defined (use 'same' or 'valid')")
        
        self.output = np.random.random_sample(self.output_size)

        if image_size[1] >= self.kernel_shape[0] and image_size[2] >= self.kernel_shape[1] and self.mode != 'full':

            for i,each_kernel in enumerate(self.kernel_weights):
                self.output[i] = 0
                for each_depth in input_image:
                    self.output[i] = self.output[i] + sp.signal.convolve(each_depth, each_kernel, mode= self.mode)

        else:
            raise ValueError("Kernel size must be smaller than input image (try changing kernel size or give larger image)")
        
class DenseLayer:

    def __init__(self, input_size, node_count, activation_object):

        if type(input_size) is int and type(node_count) is int:

            self.input_size = input_size
            self.node_count = node_count
            self.type = 'Dense'
            self.Activation = activation_object

        else:
            ValueError("Input size and node_coutn must be 'int'")


        self.weights = np.random.random_sample((node_count, input_size))
        self.output_size = None
        self.output = None

    def Compute(self, input):

        self.output = self.weights@input
        self.output_size = self.output.shape

        self.Activation.compute(input)

class Flatten:

    def __init__(self, ):
        self.output = None
        self.output_size = None
        
    def Compute(self, convolution_output):

        self.output = convolution_output.flatten()
        self.output_size = self.output.shape


class Relu:

    def __init__(self, ):
        self.output = None
        self.output_size = None
        self.derivative = None

    def Compute(self,dense_input):
        self.output = np.maximum(0,dense_input)
        self.output_size = self.output.shape

        Temp = self.output >= 0
        Temp.astype(np.int)

        self.derivative = Temp
        del Temp

    
class Signmoid:

    def __init__(self, ):
        self.output = None
        self.output_size = None
        self.derivative = None

    def Compute(self,dense_input):
        self.output = 1.0 / (1.0 + np.exp(-dense_input))
        self.output_size = self.output.shape

        Temp = np.multiply(self.output, (1 - self.output))
        self.derivative = Temp
        del Temp

    
class Softmax:
        
    def __init__(self, ):
        self.output = None
        self.output_size = None
        self.derivative = None

    def Compute(self, dense_input):
        e_x = np.exp(dense_input - np.max(dense_input))
        self.output = e_x / e_x.sum()
        self.output_size = self.output.shape

        
