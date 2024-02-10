# import the used libraries
import numpy as np
import functools

# Define the decorator function
def check_input_size(expected_size):
    """A decorator function that checks the input size of the parameter vector."""

    def decorator(func):
        """A wrapper function that performs the check and calls the original function."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """A nested function that gets the parameter vector and compares its size with the expected size."""

            # Get the parameter vector from the args
            ξ = args[1]

            # Check if it is a numpy array
            if not isinstance(ξ, np.ndarray):
                raise TypeError("The parameter vector must be a numpy array.")
            
            # Check if it is a one-dimensional array
            if ξ.ndim != 1:
                raise ValueError("The parameter vector must be a one-dimensional array.")
            
            # Check if it has the expected size
            if ξ.size != expected_size:
                raise ValueError(f"The parameter vector must have size {expected_size}. Actual size is {ξ.size}.")
            
            # Call the original function with the args and kwargs
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator