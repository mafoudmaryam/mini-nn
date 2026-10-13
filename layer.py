# layer.py - one fully connected layer for a mini neural-network library (port of your MATLAB fc.m / bc.m).

import numpy as np  # numpy gives us matrices and the matrix-multiply operator @ (like MATLAB's *)


class Layer:  # a "class" bundles a layer's data (weights) and actions (forward, backward, update) together
    def __init__(self, n_in, n_out, activation="sigmoid", init_scale=0.5):  # runs when you write Layer(2, 10, "sigmoid", init_scale=2.0); init_scale defaults to 0.5
        self.n_in = n_in  # remember how many inputs this layer receives
        self.n_out = n_out  # remember how many neurons (outputs) this layer has
        self.activation = activation  # remember which activation: "sigmoid" or "linear"
        if activation not in ("sigmoid", "linear"):  # guard against typos in the activation name
            raise ValueError("activation must be 'sigmoid' or 'linear'")  # stop with a clear error message
        self.W = np.random.randn(n_out, n_in + 1) * init_scale  # random weights scaled by init_scale (bigger = larger starting weights), shape (n_out, n_in+1); the LAST column is the bias (as in fc.m's w)
        self.a_prev = None  # will hold the input saved by forward(); backward() needs it
        self.z = None  # will hold the weighted sums saved by forward(); the network needs them for f'(z) (bc.m's z argument)
        self.grad = None  # will hold the gradient computed by backward(); update() uses it

    def _f(self, z):  # apply the activation function to z; MATLAB: the handle f passed into fc.m
        if self.activation == "sigmoid":  # case 1: sigmoid squashes values into (0, 1)
            return 1.0 / (1.0 + np.exp(-z))  # sigmoid formula, applied to every element
        return z  # case 2: "linear" activation just passes z through unchanged

    def df(self, z):  # derivative f'(z) of the activation; MATLAB: the handle df passed into bc.m
        if self.activation == "sigmoid":  # case 1: derivative of the sigmoid
            s = self._f(z)  # compute sigmoid(z) once
            return s * (1.0 - s)  # sigmoid'(z) = s * (1 - s), element-wise like MATLAB's .*
        return np.ones_like(z)  # case 2: derivative of a linear function is 1 everywhere

    def forward(self, a_prev):  # forward pass; a_prev has shape (n_in, m): one column per sample. MATLAB: fc.m
        self.a_prev = a_prev  # save the input for the backward pass
        ones_row = np.ones((1, a_prev.shape[1]))  # a row of 1s, one per sample; MATLAB (fc.m line 5): ones(1, size(a, 2))
        a_aug = np.vstack([a_prev, ones_row])  # stack the 1s under the inputs so the bias column of W gets multiplied by 1; MATLAB: [a; ones(1, size(a, 2))]
        self.z = self.W @ a_aug  # weighted sums + bias, shape (n_out, m); MATLAB (fc.m line 5): z_next = w * [a; ones(...)]
        return self._f(self.z)  # apply the activation and return the layer output; MATLAB (fc.m line 6): a_next = f(z_next)

    def backward(self, dA):  # backward pass; dA = error at THIS layer's OUTPUT (after the activation), shape (n_out, m). MATLAB: bc.m
        delta = dA * self.df(self.z)  # this layer applies its own activation derivative to get the error at z; MATLAB (bc.m line 5): df(z) .* (...)
        m = self.a_prev.shape[1]  # number of samples, used to average the gradient
        ones_row = np.ones((1, m))  # the same row of 1s used in forward()
        a_aug = np.vstack([self.a_prev, ones_row])  # rebuild [a_prev; ones] from the saved input
        self.grad = delta @ a_aug.T / m  # gradient of the loss w.r.t. W, shape (n_out, n_in+1), averaged over samples
        return self.W[:, :-1].T @ delta  # error at the previous layer's OUTPUT: W without bias column, transposed, times delta; MATLAB (bc.m line 5): w(:, 1:end-1)' * delta_next
        # The previous layer's backward() will multiply this by its own df(z), so the caller just passes it along:
        # dA_prev = layer.backward(dA)

    def update(self, alpha):  # gradient-descent step with learning rate alpha
        self.W -= alpha * self.grad  # move the weights against the gradient: W = W - alpha*grad


if __name__ == "__main__":  # small test; runs only when you execute layer.py directly
    from data import X_train  # reuse the 2 x 121 training inputs from data.py
    layer = Layer(2, 10, "sigmoid", init_scale=2.0)  # a layer with 2 inputs and 10 sigmoid neurons, larger starting weights
    out = layer.forward(X_train)  # run the forward pass on all 121 samples at once
    print("output shape:", out.shape)  # expected: (10, 121)
    dA_prev = layer.backward(np.ones_like(out))  # backward pass with a dummy output error of all ones
    print("grad shape:", layer.grad.shape, "dA_prev shape:", dA_prev.shape)  # expected: (10, 3) and (2, 121)
