import numpy as np
import argparse
import os
from DataLoader import DataLoader
import matplotlib.pyplot as plt
import time

# def sign(wx:np) -> int:
#     return 1 if wx >= 0 else -1
    
def PLA(DataLoader: DataLoader) -> np.ndarray:
    """
    Do the PLA here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  

    """
    weight_matrix = np.zeros(3)
    s = time.time()
    ############ START ##########
    np.set_printoptions(suppress=True)
    n = len(DataLoader.data)
    random_index = np.random.permutation(n)
    DataLoader.data = [DataLoader.data[i] for i in random_index]
    DataLoader.label = [DataLoader.label[i] for i in random_index]
    iterations = 0
    while True:
        mistake = False
        for i in range(n):
            x = np.array(DataLoader.data[i])
            y = DataLoader.label[i]
            iterations += 1
            if np.sign(np.dot(weight_matrix,x)) != y:
                weight_matrix += y * x
                mistake = True
                break
        if not mistake:
            break
    ############ END ############
    e = time.time()
    print("ex time : %f" % (e-s))
    print("iterations: %d"% iterations)
    print(f"weight_matrix: {weight_matrix}")
    return weight_matrix


def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    updated_weight = PLA(DataLoader=Loader)

    # This part is for plotting the graph
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.scatter(Loader.cor_x_pos, Loader.cor_y_pos,
                c='b', label='pos data')
    plt.scatter(Loader.cor_x_neg, Loader.cor_y_neg,
                c='r', label='neg data')

    x = np.linspace(-1000, 1000, 100)
    # This is the base line
    y1 = 3*x+5
    # This is your split line
    y2 = (updated_weight[1]*x + updated_weight[0]) / (-updated_weight[2])
    plt.plot(x, y1, 'g', label='base line', linewidth='1')
    plt.plot(x, y2, 'y', label='split line', linewidth='1')
    plt.legend()
    plt.show()
    plt.savefig(f"PLA_{os.path.splitext(args.path)[0]}.png")


if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)
