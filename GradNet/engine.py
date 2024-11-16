class Value:
    def __init__(self, num, inputs = (), operation = "" ):
        self.num = num
        self.inputs = inputs
        self.operation = operation
        self.gradient = 0.0
        self._backward = lambda:None
        
        
    def __add__(self, other):   
        if not isinstance(other, Value): other = Value(other)
        out = Value(self.num + other.num, (self, other), "+")
        
        def _backward():
            self.gradient += out.gradient
            other.gradient += out.gradient  
        out._backward = _backward
        
        return out
    
    
    def __mul__(self, other):   
        if not isinstance(other, Value): other = Value(other)
        out = Value(self.num * other.num, (self, other), "*")
        
        def _backward():
            self.gradient += other.num * out.gradient
            other.gradient += self.num * out.gradient
        out._backward = _backward   
        
        return out
    
    
    def __pow__(self, other):   
        if not isinstance(other, Value): other = Value(other)
        out = Value(self.num ** other.num, (self, ), "^")
        
        def _backward():
            self.gradient += ((other.num * out.gradient) ** (other.num - 1)) * out.gradient
           
        out._backward = _backward   
        
        return out
    
    
    def relu(self):
        out = Value(max(0, self.num), (self,), "ReLU")
        
        def _backward():
            self.gradient += (out.num > 0) * out.gradient
        out._backward = _backward
        
        return out
    
    
   
    def backward(self): 
        self._backward()
             
        for val in self.inputs:
            val.backward();
            
    
    def reset_grad(self):   
        self.gradient = 0.0
        
        for val in self.inputs:
            val.reset_grad();
            
            
        
        
    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return Value(other).__sub__(self)
    
    def __repr__(self):
        return f"Value(num={self.num})"