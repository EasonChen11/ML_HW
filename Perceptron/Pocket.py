import numpy as np
import argparse
from DataLoader import DataLoader
import os
import matplotlib.pyplot as plt
import time

def Data_error(w:np, DataLoader:DataLoader)->int:
    return (np.sign(np.dot(DataLoader.data,w)) != DataLoader.label).sum() # count True

def pocket(DataLoader: DataLoader) -> np.ndarray:
    """
    Do the Pocket algorithm here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  

    """
    weight_matrix = np.zeros(3)
    s = time.time()
    ############ START ##########
    np.set_printoptions(suppress=True)
    n = len(DataLoader.data)
    # random sufform
    random_index = np.random.permutation(n)
    DataLoader.data = np.array([DataLoader.data[i] for i in random_index])
    DataLoader.label = np.array([DataLoader.label[i] for i in random_index])
    error_count = Data_error(weight_matrix,DataLoader)
    weight_error = [(weight_matrix,error_count)]
    iterations = 0
    while True:
        weight_change = False
        for i in range(n):
            if error_count == 0:
                break

            x = DataLoader.data[i]
            y = DataLoader.label[i]
            if np.sign(np.dot(weight_matrix,x)) != y:
                tmp_weight_matrix = weight_matrix+y*x
                weight_error.append((tmp_weight_matrix,Data_error(tmp_weight_matrix,DataLoader))) 
        for new_weight_matrix ,new_error in weight_error:
            if error_count > new_error:
                error_count = new_error
                weight_matrix = new_weight_matrix
                weight_change = True
        weight_error = [(weight_matrix,error_count)]
        iterations += 1
        if not weight_change:
            break
    ############ END ############
    e = time.time()
    print("ex time = %f" % (e-s))
    print("iterations: %d"% iterations)
    print(f"weight_matrix: {weight_matrix}")
    print(f"error points: {error_count}")
    return weight_matrix


def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    updated_weight = pocket(DataLoader=Loader)

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
    plt.savefig(f"Pocket_{os.path.splitext(args.path)[0]}.png")


if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)
