import numpy as np
import copy 
from .activation import ActivationFunctions, BackwardActivation

# ------------------------------------------------------------------------------
# Error Calculations
# ------------------------------------------------------------------------------
class ErrorCalculations:
    """
    The ErrorCalculations class is designed to compute the errors in neural network predictions. 
    It provides methods to calculate the difference between the predicted outputs and the actual values
    """
    @staticmethod
    def squared_error(Y, A):
        """
        Compute the Mean Squared Error loss
    
        Parameters
        ----------
        Y : true labels, numpy array of shape (number of examples, )
        AL : predicted values, numpy array of same shape as Y
    
        Returns
        -------
        mse : Mean Squared Error
        """
        squarred_error = np.square(A - Y)
        return squarred_error   
    @staticmethod
    def classification_error(Y, A):
        """
        Compute the classification error for each example
    
        Parameters
        ----------
        Y : true labels (0 or 1), numpy array of shape (number of examples, )
        AL : predicted probabilities, numpy array of same shape as Y
    
        Returns
        -------
        errors : Classification errors (0 for correct, 1 for incorrect)
        """
        errors = np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))
        return errors

# ------------------------------------------------------------------------------
# Model Parameters Initializer
# ------------------------------------------------------------------------------
class ModelParamInit:
    """
    Provides methods to set up these parameters appropriately before the training process begins.
    """
    @staticmethod
    def initialize_parameters_zeros(layer_dims):
        """
        Parameters
        ----------
        layer_dims : python array (list) containing the size of each layer.

        Returns:
        parameters : python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                - Wl : weight matrix of shape (layers_dims[l], layers_dims[l-1])
                - bl  bias vector of shape (layers_dims[l], 1)
        """
        parameters = {}
        L = len(layer_dims)

        for l in range(L):
            parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        return parameters
         
    @staticmethod
    def initialize_parameters_random(layer_dims):
        """
        Parameters
        ----------
        layer_dims : array (list) containing the dimensions of each layer in the network
        
        Returns
        -------
        parameters : dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl : weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl : bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)
        
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters["b" + str(l)] = np.random.randn((layer_dims[l], 1))
            
            assert(parameters["WS" + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert(parameters["b" + str(l)].shape == (layer_dims[l], 1))
            
        return parameters

    @staticmethod
    def initialize_parameters_He(layer_dims):
        """
        Parameters
        ----------
        layer_dims : python array (list) containing the size of each layer.
        
        Returns
        -------
        parameters : python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        - W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        - b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims) - 1 # integer representing the number of layers
            
        for l in range(1, L + 1):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2./layer_dims[l - 1])
            parameters['b' + str(l)] = np.zeros(layer_dims[l], 1)

            assert(parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert(parameters["b" + str(l)].shape == (layer_dims[l], 1))

        return parameters
# ------------------------------------------------------------------------------
# Forward Propagation
# ------------------------------------------------------------------------------
class ForwardProp:
    """
    The ForwardProp class efficiently handles the forward propagation in a neural network, 
    offering methods for linear computation and activation functions like sigmoid, relu, and tanh, essential for neural network operations.
    """
    @staticmethod
    def linear_forward(A, W, b):
        """
        Linear part of a layer's forward propagation.
    
        Parameters
        ----------
        A : activations from previous layer (or input data): (size of previous layer, number of examples)
        W : weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b : bias vector, numpy array of shape (size of the current layer, 1)
    
        Returns
        -------
        Z : the input of the activation function, also called pre-activation parameter 
        cache : a python tuple containing "A", "W" and "b" (for backpropagation)
        """
        Z = np.dot(W, A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache
    
    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        """
        Forward propagation for the LINEAR->ACTIVATION layer
        
        Parameters
        ----------
        A_prev : activations from previous layer (or input data): (size of previous layer, number of examples)
        W : weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b : bias vector, numpy array of shape (size of the current layer, 1)
        activation : the activation to be used in this layer, stored as a text string: "linear", "sigmoid", "relu" or "tanh"
    
        Returns
        -------
        A : the output of the activation function, also called the post-activation value 
        cache : a python tuple containing "linear_cache" and "activation_cache"; (for backpropagation)
                - linear_cache : A_prev, W, b
                - activation_cache : Z
        """
        if activation == "linear":
            Z, linear_cache = ForwardProp.linear_forward(A_prev, W, b)
            A, activation_cache = ActivationFunctions.linear(Z)           
        elif activation == "sigmoid":
            Z, linear_cache = ForwardProp.linear_forward(A_prev, W, b)
            A, activation_cache = ActivationFunctions.sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = ForwardProp.linear_forward(A_prev, W, b)
            A, activation_cache = ActivationFunctions.relu(Z)
        elif activation == "tanh":
            Z, linear_cache = ForwardProp.linear_forward(A_prev, W, b)
            A, activation_cache = ActivationFunctions.tanh(Z)
        else:
            raise ValueError("Unrecognized activation function: " + activation)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        
        return A, cache
    
    @staticmethod
    def L_model_forward(X, parameters, activation):
        """
        Forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->ACTIVATION computation.
                ACTIVATION = "linear", "sigmoid", "relu" or "tanh"
        
        Parameters
        ----------
        X : data, numpy array of shape (input size, number of examples)
        parameters : output of initialize_parameters_deep()
        activation = list with activation for each layer
        
        Returns
        -------
        AL : activation value from the output (last) layer
        caches : list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
                    - linear_cache : A_prev, W, b
                    - activation_cache : Z
        """
        caches = []
        A = X
        L = len(parameters) // 2   # number of layers in the neural network, divide by 2 because two parameters W, b

        for l in range(1, L):
            A_prev = A
            W, b = parameters["W" + str(l)], parameters["b" + str(l)]
            # activation has a size L-1 
            activation_function = activation[l-1]
            A, cache = ForwardProp.linear_activation_forward(A_prev, W, b, activation_function)
            caches.append(cache)
            """
            # Diagnostic print statements
            print(f"Layer {l}:")
            print(f"A_prev.shape: {A_prev.shape}")
            print(f"W.shape: {W.shape}")
            print(f"b.shape: {b.shape}")
            print(f"A.shape: {A.shape}\n")
            """
        W, b = parameters["W" + str(L)], parameters["b" + str(L)]
        activation_function = activation[L-1]
        AL, cache = ForwardProp.linear_activation_forward(A, W, b, activation_function)
        """
        # Additional diagnostic print for the last layer
        print(f"Layer {L}:")
        print(f"A_prev.shape: {A.shape}")
        print(f"W.shape: {W.shape}")
        print(f"b.shape: {b.shape}")
        print(f"AL.shape: {AL.shape}\n")
        """
        assert(AL.shape == (W.shape[0],X.shape[1])) 
        caches.append(cache)
        
        return AL, caches

    @staticmethod
    def L_model_forward_keep_prob(X, parameters, activation, keep_prob = 0.5):
        """
        Forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->ACTIVATION computation.
                ACTIVATION = "linear", "sigmoid", "relu" or "tanh"
        
        Parameters
        ----------
        X : data, numpy array of shape (input size, number of examples)
        parameters : output of initialize_parameters_deep()
        activation = list with activation for each layer
        keep_prob - probability of keeping a neuron active during drop-out, scalar
        
        Returns
        -------
        AL : activation value from the output (last) layer
        caches : list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
                    - linear_cache : A_prev, W, b
                    - activation_cache : Z
        """
        caches = []
        A = X
        L = len(parameters) // 2   # number of layers in the neural network, divide by 2 because two parameters W, b

        for l in range(1, L):
            A_prev = A
            W, b = parameters["W" + str(l)], parameters["b" + str(l)]
            activation_function = activation[l-1]
            A, cache = ForwardProp.linear_activation_forward(A_prev, W, b, activation)
            D = np.random.rand(A.shape[0], A.shape[1])
            A = A * D
            caches.append(cache)
        
        W, b = parameters["W" + str(L)], parameters["b" + str(L)]
        activation_function = activation[L-1]
        AL, caches = ForwardProp.linear_activation_forward(A_prev, W, b, activation)
        
        assert(AL.shape == (W.shape[0],X.shape[1])) 
        caches.append(cache)
        
        return AL, caches

# ------------------------------------------------------------------------------
# Cost function
# ------------------------------------------------------------------------------
class CostFunction:
    """
    The CostFunction class provides methods for calculating loss metrics, 
    such as mean squared error or cross-entropy, crucial for evaluating the performance of a neural network.
    """
    @staticmethod
    def compute_cost_regression(AL, Y):
        """
        Cost function for regression (linear, relu, tanh).
    
        Parameters
        ----------
        AL : predicted values, numpy array of same shape as (number of examples, )
        Y : true labels, numpy array of shape (number of examples, )
    
        Returns
        -------
        cost : mean squarred error
        """
        m = Y.shape[0]
        errors = ErrorCalculations.squared_error(Y, AL)
        cost = np.sum(errors) / m
        
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        
        return cost
    
    @staticmethod    
    def compute_cost_classification(AL, Y):
        """
        Cost function for classification (linear, sigmoid, relu, tanh).
    
        Parameters
        ----------
        AL : probability vector corresponding to the label predictions, shape (1, number of examples)
        Y : true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    
        Returns
        -------
        cost : cross-entropy cost
        """
        m = Y.shape[1]
        errors = ErrorCalculations.classification_error(Y, AL)
        cost = np.sum(errors) / m
        
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        return cost    

# ------------------------------------------------------------------------------
# Backward Propagation
# ------------------------------------------------------------------------------
class BackwardProp:
    """
    The BackwardProp class facilitates the backward propagation process in neural networks, 
    handling gradient calculations essential for updating model parameters during training.
    """
    @staticmethod
    def linear_backward(dZ, cache):
        """
        Linear portion of backward propagation for a single layer (layer l)
        
        Parameters
        ----------
        dZ : Gradient of the cost with respect to the linear output (of current layer l)
        cache : tuple of values (A_prev, W, b)  (from forward propagation)
        
        Returns
        -------
        dA_prev : Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW : Gradient of the cost with respect to W (current layer l), same shape as W
        db : Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
        
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis = 1, keepdims = True ) / m
        dA_prev = np.dot(W.T, dZ)
       
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
       
        return dA_prev, dW, db
    
    @staticmethod
    def linear_activation_backward(dA, cache, activation):
        """
        Backward propagation for the LINEAR->ACTIVATION layer.
        
        Parameters
        ----------
        dA : post-activation gradient for current layer l 
        cache : tuple of values (linear_cache, activation_cache) (from forward propagation)
                    - linear_cache : A_prev, W, b
                    - activation_cache : Z
        activation : the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" or 'tanh' or "linear"
    
        
        Returns
        -------
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """    
        linear_cache, activation_cache = cache 
        
        if activation == "linear":
            dZ = BackwardActivation.linear_activation_backward(dA, activation_cache)
            dA_prev, dW, db = BackwardProp.linear_backward(dZ, linear_cache)
        
        if activation == "sigmoid":
            dZ = BackwardActivation.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = BackwardProp.linear_backward(dZ, linear_cache)
            
    
        if activation == "relu":
            dZ = BackwardActivation.relu_backward(dA, activation_cache)
            dA_prev, dW, db = BackwardProp.linear_backward(dZ, linear_cache)
            
        if activation == "tanh":
            dZ = BackwardActivation.tanh_backward(dA, activation_cache)
            dA_prev, dW, db = BackwardProp.linear_backward(dZ, linear_cache)
            
        return dA_prev, dW, db

    @staticmethod
    def squared_error_gradient(Y, A):
        """
        Compute the gradient of the squared error
    
        Parameters
        ----------
        Y : true values, numpy array of shape (number of examples, )
        A : predicted values, numpy array of same shape as Y
    
        Returns
        -------
        gradient : Gradient of the squared error with respect to A
        """
        gradient = 2 * (A - Y)
        return gradient
    
    @staticmethod
    def cross_entropy_gradient(Y, A):
        """
        Compute the gradient of the cross-entropy loss for binary classification
    
        Parameters
        ----------
        Y : true labels, numpy array of shape (number of examples, )
        A : predicted probabilities, numpy array of same shape as Y
    
        Returns
        -------
        gradient : Gradient of the cross-entropy loss with respect to A
        """
        gradient = - (np.divide(Y, A) - np.divide(Y, A - 1))    
        return gradient
    
    @staticmethod
    def L_model_backward_classification(AL, Y, caches, activation):
        """
        Backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID, or TANH, or RELU
        
        Parameters
        ----------
        AL : probability vector, output of the forward propagation (L_model_forward())
        Y : true "label" vector
        caches : list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
                    - linear_cache : A_prev, W, b
                    - activation_cache : Z
        activation : list of activation function for each layer

        Returns
        -------
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        current_activation = activation[L-1]
        current_cache = caches[L-1]
        dAL = BackwardProp.cross_entropy_gradient(Y, AL)
        dA_prev_temp, dW_temp, db_temp = BackwardProp.linear_activation_backward(dAL, current_cache, current_activation)
        
        grads["dA"+str(L-1)] = dA_prev_temp
        grads["dW"+str(L)] = dW_temp
        grads["db"+str(L)] = db_temp
        
        for l in reversed(range(L-1)):
            current_activation = activation[l-1]
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = BackwardProp.linear_activation_backward(dA_prev_temp, current_cache, current_activation)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            
        return grads
    
    @staticmethod
    def L_model_backward_regression(AL, Y, caches, activation):
        """
        Backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> LINEAR or TANH or RELU
        
        Parameters
        ----------
        AL : probability vector, output of the forward propagation (L_model_forward())
        Y : true labels, numpy array of shape (number of examples, )
        caches : list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
                    - linear_cache : A_prev, W, b
                    - activation_cache : Z
        activation : list of activation function for each layer

        Returns
        -------
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches)
        print(f"taille du caches {L}")
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        current_activation = activation[L-1]
        current_cache = caches[L-1]
        dAL = BackwardProp.squared_error_gradient(Y, AL)
        dA_prev_temp, dW_temp, db_temp = BackwardProp.linear_activation_backward(dAL, current_cache, current_activation)
        
        grads["dA"+str(L-1)] = dA_prev_temp
        grads["dW"+str(L)] = dW_temp
        grads["db"+str(L)] = db_temp
        
        for l in reversed(range(L-1)):
            current_activation = activation[l-1]
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = BackwardProp.linear_activation_backward(dA_prev_temp, current_cache, current_activation)
            grads["dA"+str(l)] = dA_prev_temp
            grads["dW"+str(l + 1)] = dW_temp
            grads["db"+str(l + 1)] = db_temp
            
            # Diagnostic print statements
            """
            print(f"Layer {l}:")
            print(f"dA_prev.shape : {dA_prev_temp.shape}")
            print(f"dW.shape : {dW_temp.shape}")
            print(f"db.shape : {db_temp.shape} ")
            """
        return grads     
 
# ------------------------------------------------------------------------------
# Update parameters
# ------------------------------------------------------------------------------
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Parameters
    ----------
    params : python dictionary containing the parameters 
    grads -- python dictionary containing the gradients, output of L_model_backward
    
    Returns:
    ----------
    parameters : python dictionary containing your updated parameters 
                  - parameters["W" + str(l)] = ... 
                  - parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(parameters)
    L = int(len(parameters)/2)
    print(L)
    
    for l in range(0, L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
 




    
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     