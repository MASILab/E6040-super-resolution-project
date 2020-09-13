import os
import numpy as np
import nibabel as nib
import sys
from tqdm import tqdm

def main(label_dir,output_555_dir,output_225_dir,input_555_dir,input_225_dir):
    input_555_loss = 0
    input_225_loss = 0
    output_555_loss = 0
    output_225_loss = 0

    for file in tqdm(sorted(os.listdir(label_dir))):
        label_file = os.path.join(label_dir,file)
        input_555_file = os.path.join(input_555_dir,file)
        input_225_file = os.path.join(input_225_dir,file)
        output_555_file = os.path.join(output_555_dir,file)
        output_225_file = os.path.join(output_225_dir,file)

        input_555_loss += mse_loss(label_file,input_555_file)
        input_225_loss = mse_loss(label_file,input_225_file)
        output_555_loss += mse_loss(label_file,output_555_file)
        output_225_loss += mse_loss(label_file,output_225_file)

    print('input 225 loss',input_225_loss / len(os.listdir(label_dir)))
    print('input 555 loss',input_555_loss / len(os.listdir(label_dir)))
    print('output 225 loss',output_225_loss / len(os.listdir(label_dir)))
    print('output 555 loss',output_555_loss / len(os.listdir(label_dir)))

def mse_loss(file1,file2):
    img1 = nib.load(file1).get_fdata()
    img2 = nib.load(file2).get_fdata()

    return np.square(img1 - img2).mean()


        





if __name__ == '__main__':
    label_dir = '/nfs/masi/yangq6/CD_DWI/eval/norm_label'
    output_555_dir = '/nfs/masi/yangq6/CD_DWI/eval/norm_output_555'
    output_225_dir = '/nfs/masi/yangq6/CD_DWI/eval/norm_output_225'
    input_225_dir = '/nfs/masi/yangq6/CD_DWI/eval/norm_input_225'
    input_555_dir = '/nfs/masi/yangq6/CD_DWI/eval/norm_input_555'
    main(label_dir,output_555_dir,output_225_dir,input_555_dir,input_225_dir)