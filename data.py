# data.py - Python port of the MATLAB dataset for the surface z = sin(x)*cos(y).
# I could not find your original .m file, so this follows your description; MATLAB equivalents are in comments.

import numpy as np  # numpy replaces MATLAB's built-in matrix/array support
import matplotlib.pyplot as plt  # matplotlib replaces MATLAB's figure/plot3/scatter3 functions


def target(x, y):  # target surface; MATLAB: f = @(x,y) sin(x).*cos(y);
    return np.sin(x) * np.cos(y)  # '*' is element-wise on numpy arrays, like MATLAB's '.*'


def make_set(n):  # build one dataset on an n-by-n grid; MATLAB: a local function or a script section
    g = np.linspace(-3, 3, n)  # n evenly spaced points on [-3, 3]; MATLAB: g = linspace(-3, 3, n);
    xx, yy = np.meshgrid(g, g)  # n-by-n coordinate grids; MATLAB: [xx, yy] = meshgrid(g, g); (same 'xy' layout)
    zz = target(xx, yy)  # surface height at every grid point; MATLAB: zz = sin(xx).*cos(yy);
    x = np.vstack([xx.ravel(order="F"), yy.ravel(order="F")]) / 3  # 2 x n^2 inputs scaled by 3; MATLAB: X = [xx(:)'; yy(:)'] / 3;  (order="F" = column-major flatten, like MATLAB's (:))
    y = zz.ravel(order="F").reshape(1, -1)  # 1 x n^2 targets, one column per sample; MATLAB: Y = zz(:)';
    return x, y  # hand back inputs and targets


X_train, Y_train = make_set(11)  # 11x11 training grid -> X_train 2x121, Y_train 1x121; MATLAB: [X_train, Y_train] = make_set(11);
X_test, Y_test = make_set(30)  # 30x30 testing grid -> X_test 2x900, Y_test 1x900; MATLAB: [X_test, Y_test] = make_set(30);


if __name__ == "__main__":  # run only when executed directly (MATLAB scripts always run top to bottom)
    print("X_train", X_train.shape, "Y_train", Y_train.shape)  # MATLAB: disp(size(X_train)); disp(size(Y_train));
    print("X_test ", X_test.shape, "Y_test ", Y_test.shape)  # MATLAB: disp(size(X_test)); disp(size(Y_test));

    fig = plt.figure(figsize=(8, 6))  # new figure window; MATLAB: figure;
    ax = fig.add_subplot(projection="3d")  # 3-D axes; MATLAB: axes (scatter3 creates them automatically)
    ax.scatter(X_test[0] * 3, X_test[1] * 3, Y_test[0], s=6, c="tab:blue", alpha=0.4, label="test (30x30)")  # test points, x/y rescaled back to [-3, 3]; MATLAB: scatter3(X_test(1,:)*3, X_test(2,:)*3, Y_test, 6, 'b'); hold on;
    ax.scatter(X_train[0] * 3, X_train[1] * 3, Y_train[0], s=30, c="tab:red", label="train (11x11)")  # training points, larger and red; MATLAB: scatter3(X_train(1,:)*3, X_train(2,:)*3, Y_train, 30, 'r', 'filled');
    ax.set_xlabel("x")  # MATLAB: xlabel('x');
    ax.set_ylabel("y")  # MATLAB: ylabel('y');
    ax.set_zlabel("z = sin(x)cos(y)")  # MATLAB: zlabel('z = sin(x)cos(y)');
    ax.set_title("Dataset: z = sin(x)cos(y)")  # MATLAB: title('Dataset: z = sin(x)cos(y)');
    ax.legend()  # MATLAB: legend('test (30x30)', 'train (11x11)');
    plt.show()  # display the figure; MATLAB shows it automatically, Python needs this call
