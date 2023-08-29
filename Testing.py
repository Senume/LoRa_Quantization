from CNN_Env.src.Layers import Convolution2D
import numpy as np 


Object = Convolution2D(13, (3,3))
image = np.random.random_sample((10,5,5))

Object.Compute(image)

# for each in Object.output:
#     print(each, each.shape)

print(Object.output.shape)



