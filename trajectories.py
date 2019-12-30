import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import spatial


def kernel1(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def normalise(y, mu, std):
    y_new = (y - y.mean())/y.std()
    y_new = y_new*std
    y_new = y_new+mu
    print(y_new.mean())
    print(y_new.std())
    return y_new

def get_trajectories(mu_desired, std_desired, num_of_trajectories, nb_of_samples=5000):
    # Independent variable samples
    X = np.expand_dims(np.linspace(-30, 30, nb_of_samples), 1)
    Σ = kernel1(X, X)  # Kernel of data points

    # Assume a mean of 0 for simplicity
    ys = np.random.multivariate_normal(
        mean=np.zeros(nb_of_samples), cov=Σ,
        size=num_of_trajectories)
    ynew = []
    for yi in ys:
        y_new_i = normalise(yi, mu_desired, std_desired)
        ynew.append(y_new_i)
    return np.array(ynew)

def plot_them(trajs):
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title("Trajectories of gaussian process with mu=" + str(trajs[0].mean()) + ", std= " + str(trajs[0].std()))
    for yi in trajs:
        plt.plot(yi)
        print(yi.mean())
        print(yi.std())
    plt.show()

###############################################
