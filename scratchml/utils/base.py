import inspect

def clone_estimator(estimator):
    # Get the class of the instance
    cls = estimator.__class__

    # Get the constructor arguments and their values
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    if init is object.__init__:
        # No parameters
        return cls()
    
    # Introspect the constructor arguments to find the model parameters
    init_signature = inspect.signature(init)
    parameters = [p for p in init_signature.parameters.values()
                  if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    
    # Get the values of the parameters from the original estimator
    param_values = {param.name: getattr(estimator, param.name) for param in parameters}

    # Create a new instance with the same parameters
    return cls(**param_values)