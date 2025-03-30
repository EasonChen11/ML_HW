import numpy as np
import argparse
import os
from DataLoader import DataLoader
import matplotlib.pyplot as plt


def Linear_Regression(DataLoader: DataLoader):
    """
    Do the Linear_Regression here on your own.
    weights -> 2 * 1 resulted weight matrix  
    Ein -> The in-sample error
    """
    weights = np.zeros(2)
    Ein = 0
    DataLoader.data = np.array(DataLoader.data)
    ############ START ##########
    N = len(DataLoader.data)
    X = np.array([[xi, 1] for xi in DataLoader.data[:,0]])
    Y = np.array(DataLoader.data)[:,1]
    # print(X.shape,X.T.shape,Y.shape)
    weights = np.linalg.inv(X.T @ X) @ X.T @ Y
    Ein = np.sum((X @ weights - Y) ** 2) / N
    # GradientEin = 2/N(x^t . x . w - x^t . y)
    GradientEin = 2 * (X.T @ X @ weights - X.T @ Y) / N
    print(f"Regression line: y = {weights[0]}x + {weights[1]}")
    print(f"In-sample error 𝐸𝑖𝑛 = {Ein}")
    print(f"Gradient Ein: ∇𝐸𝑖𝑛 = {GradientEin}")
    ############ END ############
    return weights, Ein


def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    weights, Ein = Linear_Regression(DataLoader=Loader)

    # This part is for plotting the graph
    plt.title(
        'Linear Regression_%s, Ein = %.2f, W = [%.2f,%.2f]' % (os.path.splitext(args.path)[0],Ein,weights[0],weights[1]))
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    Data = np.array(Loader.data)
    plt.scatter(Data[:, 0], Data[:, 1], c='b', label='data')

    x = np.linspace(-100, 100, 10)
    # This is your regression line
    y = weights[0]*x + weights[1]
    plt.plot(x, y, 'g', label='regression line', linewidth='1')
    plt.legend()
    plt.show()
    plt.savefig(f"LR_{os.path.splitext(args.path)[0]}.png")


if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        description='Place the .tx.T file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)
