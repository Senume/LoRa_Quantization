import numpy as np


class Gradient:

    def __init__(self, target, input, learning_rate):
        self.delta = []
        self.gradient = []
        self.target = target
        self.input = input
        self.learning_rate = learning_rate

    def Compute_delta(self, Layers):

        for i,layer in enumerate(reversed(Layers)):
            if layer.type == 'Dense':
                if i == 0:
                    self.delta.append(np.multiply(layer.Activation.output - self.target, layer.Activation.derivative))
                else:
                    self.delta.append(np.multiply((Layers[i - 1].weights.T @ self.delta[-1]), layer.Activation.derivative))

    def Compute_gradient(self, Layers):

        for i,layer in enumerate(reversed(Layers)):
            if i == 0:
                self.gradient.append(self.delta[i]@self.input.T)
            else:
                self.gradient.append(self.delta[i]@Layers[i - 1].Activation.output.T)

    
    def AdjustWeights(self, Layers):

        for i,layer in enumerate(reversed(Layers)):
            layer.weights -= self.learning_rate*self.gradient[i]
        
    
