import numpy as np 


# ------------------------------------------------------------------------------
# Main activation function
# ------------------------------------------------------------------------------

class ActivationFunctions:
    """
    staticmethod
        - If these methods do not depend on instance attributes
        - Clarifies that these methods can be called on the class itself, without needing to instantiate an object
    """

    def __init__(self):
        """
        Initialize the ActivationFunctions instance.
        This constructor can be extended to include instance-specific attributes if necessary.
        """
    pass

    @staticmethod
    def linear(Z):
        """
        Linear activation
    
        Parameters
        ----------
        Z : numpy array of any shape
        
        Returns
        -------
        A : output of linear(z), same shape as Z
        cache : returns Z (for backpropagation)
        """
        
        A = Z
        assert(A.shape == Z.shape)
        
        cache = Z
        
        return A, cache
    @staticmethod
    def sigmoid(Z):
        """
        Sigmoid activation
    
        Parameters
        ----------
        Z : numpy array of any shape
        
        Returns
        -------
        A : output of sigmoid(z), same shape as Z
        cache : returns Z (for backpropagation)
        """
        
        A = 1/(1+np.exp(-Z))
        assert(A.shape == Z.shape)
        
        cache = Z
        
        return A, cache
    @staticmethod
    def relu(Z):
        """
        Relu function
        
        Parameters
        ----------
        Z : Output of the linear layer, of any shape
      
        Returns
        -------
        A : Post-activation parameter, of the same shape as Z
        cache : a python dictionary containing "Z" (for backpropagation)
    
        """
        A = np.maximum(0, Z)
        assert(A.shape == Z.shape)
        
        cache = Z
        
        return A, cache
    @staticmethod
    def tanh(Z):
        """
        Tanh activation
        
        Parameters
        ----------
        Z : Output of the linear layer, of any shape
    
        Returns
        -------
        A : Post-activation parameter, of the same shape as Z
        cache : a python dictionary containing "Z" (for backpropagation)
    
        """
        A = np.tanh(Z)
        assert(A.shape == Z.shape)
        
        cache = Z
        
        return A, cache
    
    @staticmethod
    def softmax(Z):
        """
        Softmax activation
        
        Parameters
        ----------
        Z : Output of the linear layer, numpy array of any shape
        
        Returns
        -------
        A : Post-activation parameter, of the same shape as Z
        cache : a python dictionary containing "Z" (for backpropagation)
        """
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stabilize softmax by subtracting max from Z
        A = expZ / np.sum(expZ, axis=0, keepdims=True)
        assert(A.shape == Z.shape)
        
        cache = Z
        return A, cache

# ------------------------------------------------------------------------------
# Main Backward Activation
# ------------------------------------------------------------------------------

class BackwardActivation:
    """
    The BackwardActivation class provides static methods for implementing 
    backward propagation steps for different activation functions used in neural networks. 
    """

    def __init__(self):
        """
        Initialize the BackwardActivation instance.
        This constructor can be extended to include instance-specific attributes if necessary.
        """
    pass

    @staticmethod
    def linear_activation_backward(dA, cache):
        """
        Backward propagation for a single LINEAR unit.
        
        Parameters
        ----------
        dA : post-activation gradient, of any shape
        cache : 'Z' (for backpropagation)
    
        Returns
        -------
        dZ : Gradient of the cost with respect to Z
        """
        
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        print("dZ shape:", dZ.shape)
        print("Z shape:", Z.shape)
        # When z <= 0, we should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    
    @staticmethod  
    def sigmoid_backward(dA, cache):
        """
        Backward propagation for a single SIGMOID unit.
    
        Parameters
        ----------
        dA : post-activation gradient, of any shape
        cache : 'Z' (for backpropagation)
    
        Returns
        -------
        dZ : Gradient of the cost with respect to Z
        """
        
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    
    @staticmethod
    def relu_backward(dA, cache):
        """
        Backward propagation for a single RELU unit.
    
        Parameters
        ----------
        dA : post-activation gradient, of any shape
        cache : 'Z' (for backpropagation)
    
        Returns
        -------
        dZ : Gradient of the cost with respect to Z
        """
        
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, we should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    
    @staticmethod
    def tanh_backward(dA, cache):
        """
        Backward propagation for a single TANH unit.
    
        Parameters
        ----------
        dA : post-activation gradient, of any shape
        cache : 'Z' (for backpropagation)
    
        Returns
        -------
        dZ : Gradient of the cost with respect to Z
        """
        Z = cache
        A = np.tanh(Z)  # calculate the tanh of Z
        dZ = dA * (1 - np.power(A, 2))  # derivative of tanh
    
        assert (dZ.shape == Z.shape)
        return dZ
    
    @staticmethod
    def softmax_backward(dA, cache):
        """
        Backward propagation for a single SOFTMAX unit.

        Parameters
        ----------
        dA : post-activation gradient, of any shape
        cache : 'Z' used for the softmax computation

        Returns
        -------
        dZ : Gradient of the cost with respect to Z
        """
        Z = cache
        # Compute softmax from Z
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        A = expZ / np.sum(expZ, axis=0, keepdims=True)
        # Compute dZ
        dZ = dA * A * (1 - A)

        assert (dZ.shape == Z.shape)
        return dZ
