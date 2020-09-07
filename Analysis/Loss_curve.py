import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def main(train_txt_file,valid_txt_file):
    train = np.loadtxt(train_txt_file)
    valid = np.loadtxt(valid_txt_file)
    plt.plot(range(1,12),train,label='train')
    plt.plot(range(1,12),valid,label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == '__main__':
    train_txt_file = '/nfs/masi/yangq6/CD_DWI/result/train.txt'
    valid_txt_file = '/nfs/masi/yangq6/CD_DWI/result/valid.txt'
    main(train_txt_file,valid_txt_file)
