from utils import ModelParamInit as MPI, ForwardProp as FP, CostFunction as CF, BackwardProp as BP, update_parameters as up
import numpy as np

# ------------------------------------------------------------------------------
# Several layers model
# ------------------------------------------------------------------------------

class NeuralNetworkModel:
    """
    A class to represent a multi-layer neural network model.
    
    Attributes
    ----------
    layer_dims : list
        A list containing the dimensions of each layer in the network.
    activation : list
        A list containing the activation functions to be used in each layer.
    initialization : str
        The method to be used for initializing the weights. Can be "zeros", "random", or "he".
    learning_rate : float
        The learning rate for the gradient descent optimization algorithm.
    num_iterations : int
        The number of iterations for the optimization algorithm.
    lambd : float, optional
        The regularization hyperparameter. The default is 0 (no regularization).
    keep_prob : float, optional
        The probability of keeping a neuron active during dropout. The default is 1 (no dropout).
    
    Methods
    -------
    train(X, Y):
        Trains the neural network using the provided training data.
    """
    
    def __init__(self, layer_dims, activation, initialization, learning_rate, num_iterations, lambd=0, keep_prob=1):
        """
        Constructs all the necessary attributes for the NeuralNetworkModel object.

        Parameters
        ----------
        layer_dims : list
            A list containing the dimensions of each layer in the network.
        activation : list
            A list containing the activation functions to be used in each layer.
        initialization : str
            The method to be used for initializing the weights. Can be "zeros", "random", or "he".
        learning_rate : float
            The learning rate for the gradient descent optimization algorithm.
        num_iterations : int
            The number of iterations for the optimization algorithm.
        lambd : float, optional
            The regularization hyperparameter. The default is 0.
        keep_prob : float, optional
            The probability of keeping a neuron active during dropout. The default is 1.
        """
        self.layer_dims = layer_dims
        self.activation = activation
        self.initialization = initialization
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambd = lambd
        self.keep_prob = keep_prob

    def train(self, X, Y):
        """
        Trains the neural network using the provided training data.

        Parameters
        ----------
        X : numpy.ndarray
            The input data, with shape (n_x, number of examples).
        Y : numpy.ndarray
            The true "label" vector, with shape (n_y, number of examples).

        Returns
        -------
        parameters : dict
            The parameters learned by the model, which can be used for predictions.
        costs : list
            The costs of the model following some iterations and the last one.
        """
        np.random.seed(1)  # Ensuring consistency in random operations
        costs = []  # Keeping track of the cost

        # Initialize parameters based on the chosen method
        if self.initialization == "random":
            parameters = MPI.initialize_parameters_random(self.layer_dims)  
        elif self.initialization == "zeros":
            parameters = MPI.initialize_parameters_zeros(self.layer_dims)
        elif self.initialization == "he":
            parameters = MPI.initialize_parameters_He(self.layer_dims)
        
        # Loop (optimization) over num_iterations
        for i in range(self.num_iterations):
            # Forward propagation: [LINEAR -> ACTIVATION] * (L-1) -> LINEAR -> SOFTMAX.
            AL, caches = FP.L_model_forward(X, parameters, self.activation)
            
            # Compute cost.
            J = CF.compute_cost_regression(AL, Y) 
            
            # Backward propagation.
            grads = BP.L_model_backward_regression(AL, Y, caches, self.activation)
            
            # Update parameters.
            parameters = up(parameters, grads, self.learning_rate)
            
            # Print the cost every 10000 iterations
            if i % 10000 == 0 or i == self.num_iterations - 1:
                print(f"For iteration {i}: \n  cost = {np.squeeze(J)} \n")
                costs.append(J)

        return parameters, costs
    
    




        



        
# ------------------------------------------------------------------------------
# Performance metrics
# ------------------------------------------------------------------------------
class PerfMetrics:
    """
    A class to compute performance metrics for machine learning models.
    """

    @staticmethod
    def accuracy(predictions, labels):
        """
        Calculate the accuracy of predictions against true labels.

        Parameters:
        - predictions: A list of predicted values.
        - labels: A list of actual values.

        Returns:
        - Accuracy as a float.
        """
        if len(predictions) != len(labels):
            raise ValueError("Length of predictions and labels must be the same.")

        correct = sum(p == l for p, l in zip(predictions, labels))
        return correct / len(labels)


# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------
def predict(X, Y, parameters, activation) :
    """
    This function is used to predict the results of a  L-layer neural network.

    Parameters
    ----------
    X : input data, of shape (n_x, number of examples).
    parameters : parameters learnt by the model.    
    layer_dims : list containing the input size and each layer size, of length (number of layers + 1).
    activation : list containing the activation function for each layer, of length (number of layers + 1).
    learning_rate : learning rate of the gradient descent update rule.
    num_iterations : number of iterations of the optimization loop.

    Returns 
    -------

    costs : cost of the model following some iterations and the last one.
    """   
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network

    AL, caches = FP.L_model_forward(X, parameters, activation)

    print("Accuracy: "  + Accuracy())

    return AL