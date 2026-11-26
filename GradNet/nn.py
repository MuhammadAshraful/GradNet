import random
from GradNet.engine import Value

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w,x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        #here as you can see we are only collecting weights and biases
        #Hence when we update we only update weights and biases according to their current gradient
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
        # params = []
        # for neuron in self.neurons:
        #     p = neuron.parameters()
        #     params.extend(p)
        # return params


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range (len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        




# nin = 2 means it creates two weights. So basically Neuron(2) is just one neuron we are introducing that does multiplication of the 
# two weights with two inputs

# x in __call__ function is inputs [2.0, 3.0]
# nout is the number of neurones in a layer