import numpy as np

def linearRegression(X: np.array, Y: np.array, lr: float, lambda_: float, convergence_threshold: float = 1e-8):
    """
    Parameters:
    - X: Input feature matrix (NumPy array)
    - Y: Target vector (NumPy array)
    - lr: Learning rate (float)
    - lambda_: L1 regularization coefficient (float)

    Returns:
    - weights: Learned model parameters
    """
    
    # I'm considing weights[0] to be the constant term
    weights = np.random.randn(X.shape[1] + 1)
    # add a column of ones to X for the bias term, so we can treat it as a weight
    x = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term

    # I track the losses so that the model stops after mse doesn't change significantly
    # Also I can make neat diagrams later :)
    losses = []
    while True:
        pred = x @ weights
        mse = np.mean((pred - Y) ** 2)
        # del C / del w_i = 2 (pred - y) * x_i
        grads = 2 * (pred - Y).reshape(-1, 1) * x
        # regularisation term = lambda_ * sum(abs(weights)) so in derivative we have sign(weights)
        l1_term = lambda_ * np.sign(weights)
        l1_term[0] = 0 # don't regularise the constant
        grad = np.mean(grads, axis=0) + l1_term
        weights -= lr * grad
        # print(mse)

        if len(losses) > 0 and np.abs(mse - losses[-1]) < convergence_threshold:
            break
        losses.append(mse)
    return weights

if __name__ == '__main__':
    X = np.loadtxt('X.npy')
    Y = np.loadtxt('Y.npy')
    linearRegression(X, Y, lr=0.01, lambda_=0.1)
