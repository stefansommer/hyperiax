from .parameter import Parameter

import jax
import jax.numpy as jnp

class UniformParameter(Parameter):
    def __init__(self, value, min=0., max=1., keep_constant=False) -> None:
        super().__init__(value)
        self.min = min
        self.max = max
        self.keep_constant = keep_constant

    def propose(self, key):
        if self.keep_constant:
            return self

        return UniformParameter(jax.random.uniform(key, minval=self.min, maxval=self.max), self.min, self.max, self.keep_constant)
    
    def update(self, value, accepted): 
        if accepted:
            self.value = value
    
    def log_prior(self):
        return 0.
