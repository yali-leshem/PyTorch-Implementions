class MyScalar:
    def __init__(self, value, parents=[], to_cache_grads=False):
        # initialization:
        self.value = value
        self.grads_cache = None # no need to cache currently
        self.to_cache_grads = to_cache_grads

        # creating a clone of the parents scalars of current object with a semi-list
        self.parents = []
        for parent, parent_grad in parents:
            self.parents.append((parent, parent_grad))

    def set_to_cache_grads(self, to_cache_grads):
        # set attribute
        self.to_cache_grads = to_cache_grads
        # discard cache in case caching is now disabled
        if to_cache_grads == False:
            self.grads_cache = None

    def get_value(self):
        return self.value # get method for the scalar

    def get_gradient(self): # calculate gradient with respect to previous ones relies on
        # calculate from the beginning derivatives if they are not cached:
        if self.grads_cache == None:
            grads_of_self = {} # to hold the result

            for (parent, parent_grad) in self.parents: # get previous parents through directs
                grads_of_parent = parent.get_gradient()

                for ancestor_id in grads_of_parent.keys(): # taking previous parents derivatives as well
                    if ancestor_id not in grads_of_self.keys():
                        grads_of_self[ancestor_id] = 0.0
                    grads_of_self[ancestor_id] += parent_grad * grads_of_parent[ancestor_id]
                    # = derivative(self)/derivative(parent) * derivative(parent) / derivative(ancestor)
            
            # add entry for 'self' itself:
            grads_of_self[id(self)] = 1.0

            # cache a copy of the derivatives if needed
            if self.to_cache_grads == True:
                self.grads_cache = grads_of_self.copy()
            
            return grads_of_self

        # in case derivatives are cached - return copy of the cache
        else:
            return self.grads_cache.copy()
    
    def __str__(self, d=2):
        return str(round(self.value, d))

    def __float__(self):
        return float(self.value)

    def __add__(self, other, to_cache_grads=False):
        # store parents and calculate gradients
        parents = [(self, 1.0)] # derivative(self+other)/derivative(self)=1
        if type(other) == MyScalar:
          parents.append((other, 1.0)) # derivative(self+other)/derivative(other)=1

        # create and return new MyScalar
        return MyScalar(float(self) + float(other),
                        parents, to_cache_grads)

    def __radd__(self, other, to_cache_grads=False):
        return self.__add__(other, to_cache_grads)

    def __mul__(self, other, to_cache_grads=False):
        # store parents and calculate gradients
        parents = [(self, float(other))] # derivative(self*other)/derivative(self)=other
        if type(other) == MyScalar:
          parents.append((other, float(self))) # derivative(self*other)/derivative(other)=self

        # create and return new MyScalar
        return MyScalar(float(self) * float(other),
                        parents, to_cache_grads)
        
    def __rmul__(self, other, to_cache_grads=False):
        return self.__mul__(other, to_cache_grads)

    def __truediv__(self, other, to_cache_grads=False):
        # store parents and calculate gradients
        parents = [(self, 1/float(other))] # derivative(self/other)/derivative(self)=1/other
        if type(other) == MyScalar:
            parents.append((other,
                          -float(self)/float(other)**2)) # derivative(self/other)/derivative(other)=-self/(other)**2

        # create and return new MyScalar
        return MyScalar(float(self) / float(other),
                        parents, to_cache_grads)
    
    def __rtruediv__(self, other, to_cache_grads=False):
        # store parents and calculate gradients
        parents = [(self, -float(other)/float(self)**2)] # derivative(other/self)/derivative(self)=-other/self**2
        if type(other) == MyScalar:
            parents.append((other, 1/float(self))) # derivative(other/self)/derivative(other)=1/self

        # create and return new MyScalar
        return MyScalar(float(other) / float(self),
                        parents, to_cache_grads)

    def __pow__(self, other, to_cache_grads=False):
        # store parents and calculate gradients
        parents = [(self, float(other) * (float(self)**(float(other)-1)))]
                          # d(self**other)/d(self)=other*(self)**(other-1)
        if type(other) == MyScalar:
            parents.append((other, float(self)**float(other) * math.log(float(self))))
                                  # derivative(self**other)/derivative(other)=self**other*ln(self)

        # create and return new MyScalar
        return MyScalar(float(self)**float(other),
                        parents, to_cache_grads)

    def __rpow__(self, other, to_cache_grads=False):
        # store parents and calculate gradients
        parents = [(self, float(other)**float(self) * math.log(float(other)))]
                          # derivative(other**self)/derivative(self)=other**self*ln(other)
        if type(other) == MyScalar:
            parents.append((other, float(self) * (float(other)**(float(self)-1))))
                                   # derivative(other**self)/derivative(other)=self*(other)**(self-1)
                                 

        # create and return new MyScalar
        return MyScalar(float(other)**float(self),
                        parents, to_cache_grads)
        
    def exp(self, to_cache_grads=False):
        # simply use __rpow__ defined above in order to power exponential 
        return self.__rpow__(math.e, to_cache_grads)

    def cos(self, to_cache_grads=False):
        return MyScalar(math.cos(self.value), # result
                        [(self, -math.sin(self.value))], # = derivative(cos(self))/self = -sin(self)
                        to_cache_grads)
    
    def log(self, to_cache_grads=False):
        return MyScalar(math.log(self.value), # result
                        [(self, 1/self.value)], # = derivative(log(self))/self = 1/self
                        to_cache_grads)
    
    def sin(self, to_cache_grads=False):
        return MyScalar(math.sin(self.value), # result
                        [(self, math.cos(self.value))], # = derivative(sin(self))/self = cos(self)
                        to_cache_grads)
