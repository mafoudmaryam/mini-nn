# network.py - a stack of Layer objects trained with mini-batch gradient descent (port of your MATLAB main.m training loop).
# I could not find your original main.m, so the MATLAB equivalents in the comments follow your description of it.

import numpy as np  # numpy gives us arrays, random numbers and the mean function


class Network:  # a "class" bundles the layers and the actions on them (predict, loss, fit) together
    def __init__(self, layers):  # runs when you write Network([layer1, layer2]); 'layers' is a Python list of Layer objects
        self.layers = layers  # keep the list; layers[0] is the first hidden layer, layers[-1] is the output layer; MATLAB: a cell array of weight matrices {w1, w2}
        self.history = {"train": [], "val": []}  # dictionary that will hold the loss after every epoch, one list for training and one for validation

    def predict(self, X):  # run the inputs forward through every layer; X has shape (n_in, m), one column per sample
        a = X  # the "activation" entering the first layer is just the input data; MATLAB: a = X;
        for layer in self.layers:  # visit the layers in order, first to last; MATLAB: for k = 1:numel(w)
            a = layer.forward(a)  # each layer turns its input into its output; MATLAB: [a, z] = fc(a, w{k}, f);
        return a  # the output of the last layer is the network's prediction, shape (n_out, m)

    def loss(self, X, Y):  # mean squared error with the 0.5 factor, so its derivative is just (prediction - target)
        return 0.5 * np.mean((self.predict(X) - Y) ** 2)  # predict, subtract targets, square each error, average over all entries, times 0.5; MATLAB: 0.5*mean((pred - Y).^2, 'all')

    def fit(self, X, Y, alpha, epochs, batch_size, X_val=None, Y_val=None, seed=None):  # train the network; alpha = learning rate, epochs = passes over the data
        rng = np.random.default_rng(seed)  # random number generator for shuffling; the same seed gives the same shuffles every run; MATLAB: rng(seed)
        m = X.shape[1]  # number of training samples (columns); MATLAB: m = size(X, 2);
        for epoch in range(1, epochs + 1):  # repeat for epochs = 1, 2, ..., epochs; MATLAB: for epoch = 1:epochs
            order = rng.permutation(m)  # a random ordering of the sample indices 0..m-1; MATLAB: idx = randperm(m);
            for start in range(0, m, batch_size):  # step through the shuffled samples one mini-batch at a time; MATLAB: for s = 1:batch_size:m
                idx = order[start:start + batch_size]  # indices of this mini-batch (the last batch may be smaller); MATLAB: b = idx(s:min(s+batch_size-1, m));
                xb = X[:, idx]  # pick those columns of the inputs; MATLAB: xb = X(:, b);
                yb = Y[:, idx]  # pick the matching columns of the targets; MATLAB: yb = Y(:, b);
                pred = self.predict(xb)  # forward pass; every layer also saves what backward() needs; MATLAB: the fc.m calls
                dA = pred - yb  # error at the output = derivative of 0.5*(pred-y)^2 w.r.t. pred; MATLAB: da = a - y;
                for layer in reversed(self.layers):  # go through the layers last to first (back-propagation); MATLAB: for k = numel(w):-1:1
                    dA = layer.backward(dA)  # layer computes its own gradient and returns the error for the layer before it; MATLAB: [da, grad{k}] = bc(...)
                for layer in self.layers:  # now that every gradient is ready, change the weights
                    layer.update(alpha)  # gradient-descent step W = W - alpha*grad; MATLAB: w{k} = w{k} - alpha*grad{k};
            self.history["train"].append(self.loss(X, Y))  # record the loss on the whole training set after this epoch
            if X_val is not None and Y_val is not None:  # only if validation data was given
                self.history["val"].append(self.loss(X_val, Y_val))  # record the loss on data the network never trains on
            if epoch % 1000 == 0:  # '%' is the remainder, so this is true every 1000th epoch; MATLAB: if mod(epoch, 1000) == 0
                msg = f"epoch {epoch:5d}  train loss {self.history['train'][-1]:.6f}"  # build the text to print (f-string puts values inside {})
                if self.history["val"]:  # if we have been recording validation loss
                    msg += f"  val loss {self.history['val'][-1]:.6f}"  # add it to the text
                print(msg)  # show progress; MATLAB: fprintf / disp
        return self.history  # hand the recorded losses back to the caller as well
