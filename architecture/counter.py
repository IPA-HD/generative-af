"""
Architecture decorators tracking metadata.
"""
from functools import wraps
from typing import Protocol, Optional
from abc import abstractmethod
import torch

def count_forward_calls(cls):
    """
    Decorator for nn.Module instances which counts the number of forward passes
    """
    counter_name = "fevals"
    original_forward = cls.forward
    
    @wraps(cls.forward)
    def counted_forward(self, *args, **kwargs):
        if not hasattr(self, counter_name):
            setattr(self, counter_name, 0)
        setattr(self, counter_name, getattr(self, counter_name) + 1)
        return original_forward(self, *args, **kwargs)
    
    cls.forward = counted_forward
    
    def reset_forward_count(self):
        setattr(self, counter_name, 0)

    # Method to get the count
    def get_forward_count(self):
        return getattr(self, counter_name, 0)
    
    cls.get_forward_count = get_forward_count
    cls.reset_forward_count = reset_forward_count
    
    return cls

