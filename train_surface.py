# train_surface.py - train a 2-10-1 network on z = sin(x)cos(y) using data.py, then plot the results.

import numpy as np  # numpy for arrays and random numbers
import matplotlib  # import the package itself so we can pick a backend
matplotlib.use("Agg")  # draw to files only, no window pops up
import matplotlib.pyplot as plt  # matplotlib replaces MATLAB's plotting functions

from data import X_train, Y_train, X_test, Y_test  # the 2x121 training set and 2x900 test set
from layer import Layer  # one fully connected layer
from network import Network  # the stack of layers with fit/predict/loss

SEED = 1  # one seed so every run gives identical results
np.random.seed(SEED)  # Layer() draws its starting weights from numpy's global generator, so seed it before building layers

net = Network([  # a 2-10-1 network
    Layer(2, 10, "sigmoid", init_scale=2.0),  # hidden layer: 2 inputs -> 10 sigmoid neurons, large starting weights
    Layer(10, 1, "linear"),  # output layer: 10 -> 1 linear neuron (can output any real number)
])

net.fit(X_train, Y_train, alpha=0.5, epochs=8000, batch_size=11, X_val=X_test, Y_val=Y_test, seed=SEED)  # train; test set is used only to monitor loss

train_mse = np.mean((net.predict(X_train) - Y_train) ** 2)  # plain MSE (no 0.5 factor) on the training grid
test_mse = np.mean((net.predict(X_test) - Y_test) ** 2)  # plain MSE on the unseen 30x30 grid
print(f"final train MSE: {train_mse:.6f}")  # report training error
print(f"final test  MSE: {test_mse:.6f}")  # report test error

# --- plot 1: true surface vs learned surface on the 30x30 test grid ---
n = 30  # test grid is 30 x 30
xs = X_test[0].reshape(n, n, order="F") * 3  # rebuild the x grid (data.py flattened column-major) and undo the /3 scaling
ys = X_test[1].reshape(n, n, order="F") * 3  # rebuild the y grid the same way
true_z = Y_test[0].reshape(n, n, order="F")  # true heights as a grid
learned_z = net.predict(X_test)[0].reshape(n, n, order="F")  # network's heights as a grid

fig = plt.figure(figsize=(12, 5))  # one wide figure with two panels
for i, (z, title) in enumerate([(true_z, "True: z = sin(x)cos(y)"), (learned_z, "Learned by 2-10-1 network")], start=1):  # loop over both panels
    ax = fig.add_subplot(1, 2, i, projection="3d")  # panel i of 2, 3-D axes
    ax.plot_surface(xs, ys, z, cmap="viridis")  # draw the surface
    ax.set_zlim(-1.1, 1.1)  # same height range in both panels so they compare fairly
    ax.set_title(title)  # panel title
    ax.set_xlabel("x")  # axis labels
    ax.set_ylabel("y")
fig.tight_layout()  # avoid overlapping labels
fig.savefig("surface_comparison.png", dpi=150)  # save instead of showing
plt.close(fig)  # free the figure

# --- plot 2: loss curve ---
fig = plt.figure(figsize=(7, 5))  # new figure
plt.semilogy(net.history["train"], label="train (11x11)")  # training loss per epoch, log scale on y
plt.semilogy(net.history["val"], label="test (30x30)")  # test loss per epoch
plt.xlabel("epoch")  # axis labels and title
plt.ylabel("loss (0.5 * MSE)")
plt.title("Loss curve")
plt.legend()  # show which line is which
plt.grid(True, which="both", alpha=0.3)  # light grid
fig.savefig("loss_curve.png", dpi=150)  # save to file
plt.close(fig)  # free the figure
print("saved surface_comparison.png and loss_curve.png")  # tell the user where the plots are
