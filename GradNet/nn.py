import random

class Neuron:
    def __init__(self, nin):
        # Initialize weights for each input from the previous layer
        self.weights = [random.uniform(-1, 1) for _ in range(nin)]
        self.bias = random.uniform(-1, 1)  
    
    def forward(self, inputs):
        # Multiply each input by its respective weight and sum the results, including bias
        weighted_sum = sum(w * inp for w, inp in zip(self.weights, inputs)) + self.bias
        print("inputs = ", inputs)
        print("weights = ", self.weights)
        print("bias = ", self.bias)
        print("weighted_sum = ", weighted_sum)
        print("\n")
        return  weighted_sum  


class Layer:
    def __init__(self, nin, nout):
        # Create neurons in the layer, each with 'nin' inputs (from previous layer)
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def forward(self, inputs):
        # Pass the inputs to each neuron and collect the outputs
        return [neuron.forward(inputs) for neuron in self.neurons]
    
    
class MLP():

    def __init__(self, num_weight, num_neuron, num_layer):
        self.layers = [Layer(num_weight, num_neuron ) for _ in range(num_layer)]
        
    def forward(self, inputs):
        # Pass the inputs to each neuron in the respective layer and collect the outputs
        for layer in self.layers:
            inputs = layer.forward(inputs)  # Pass the outputs as inputs to the next layer
        return inputs   
        




        