import numpy as np


class Gradient:

    def __init__(self, target, input, learning_rate):
        self.delta = []
        self.gradient = []
        self.target = target
        self.input = input
        self.learning_rate = learning_rate

    def Compute_delta(self, Layers):

        L = list(reversed(Layers))

        for i,layer in enumerate(L):
            if layer.type == 'Dense':
                if i == 0:
                    self.delta.append(np.multiply(layer.Activation.output - self.target, layer.Activation.derivative))
                else:
                    self.delta.append(np.multiply((L[i - 1].weights.T @ self.delta[-1]), layer.Activation.derivative))

    def Compute_gradient(self, Layers):

        L = list(reversed(Layers))

        for i,layer in enumerate(L):
            if i == len(L) - 1:

                self.gradient.append(self.delta[i]@self.input.T)
            else:
                self.gradient.append(self.delta[i]@L[i + 1].Activation.output.T)

    
    def AdjustWeights(self, Layers):
        L = list(reversed(Layers))
        for i,layer in enumerate(L):
            layer.weights -= self.learning_rate*self.gradient[i]
    
    def Run_tape(self, Layers):

        self.Compute_delta(Layers)
        self.Compute_gradient(Layers)
        self.AdjustWeights(Layers)

    
        
    
