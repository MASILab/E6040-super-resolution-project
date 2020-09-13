import nibabel as nib 
import numpy as np
import os
import sys
import glob
from tqdm import tqdm

def main(test_file):
    label_dir      = '/nfs/masi/yangq6/CD_DWI/data/production/HR'
    input_555_dir  = '/nfs/masi/yangq6/CD_DWI/data/production/LR'
    input_225_dir  = '/nfs/masi/yangq6/CD_DWI/data/production/LR225'
    output_555_dir = '/nfs/masi/yangq6/CD_DWI/infer/iso'
    output_225_dir = '/nfs/masi/yangq6/CD_DWI/infer/225'

    test = np.loadtxt(test_file,dtype=str,delimiter=':')
    for img_file in tqdm(test[:,1]):
        img_base_name = os.path.basename(img_file)
        norm_img(img_base_name,label_dir,input_555_dir,input_225_dir,output_555_dir,output_225_dir)
        # exit(125)


def norm_img(img_base_name,label_dir,input_555_dir,input_225_dir,output_555_dir,output_225_dir):
    label_file      = glob.glob(os.path.join(label_dir,'*{}'.format(img_base_name)))[0]
    input_555_file  = glob.glob(os.path.join(input_555_dir,'*{}'.format(img_base_name)))[0]
    input_225_file  = glob.glob(os.path.join(input_225_dir,'*{}'.format(img_base_name)))[0]
    output_555_file = glob.glob(os.path.join(output_555_dir,'*{}'.format(img_base_name)))[0]
    output_225_file = glob.glob(os.path.join(output_225_dir,'*{}'.format(img_base_name)))[0]

    save_imgs(img_base_name,label_file,input_555_file,input_225_file,output_555_file,output_225_file)

def save_imgs(img_base_name,label_file,input_555_file,input_225_file,output_555_file,output_225_file):
    base_name = img_base_name
    thre = np.percentile(nib.load(label_file).get_fdata(),99)
    save_label_dir      = '/nfs/masi/yangq6/CD_DWI/eval/norm_label'
    save_input_555_dir  = '/nfs/masi/yangq6/CD_DWI/eval/norm_input_555'
    save_input_225_dir  = '/nfs/masi/yangq6/CD_DWI/eval/norm_input_225'
    save_output_555_dir = '/nfs/masi/yangq6/CD_DWI/eval/norm_output_555'
    save_output_225_dir = '/nfs/masi/yangq6/CD_DWI/eval/norm_output_225'
    
    save_img(thre,base_name,label_file,save_label_dir,False)
    save_img(thre,base_name,input_555_file,save_input_555_dir,False)
    save_img(thre,base_name,input_225_file,save_input_225_dir,False)
    save_img(thre,base_name,output_555_file,save_output_555_dir,True)
    save_img(thre,base_name,output_225_file,save_output_225_dir,True)

def save_img(thre,base_name,img_file,save_dir,norm_bool):
    nii = nib.load(img_file)
    img = nii.get_fdata()
    if (norm_bool):
        img = img * thre
    # img[img > thre] = thre
    # img[img < 0] = 0
    save_nii = nib.Nifti1Image(img,nii.affine,nii.header)
    nib.save(save_nii,os.path.join(save_dir,base_name))




if __name__ == '__main__':
    test_file = '/nfs/masi/yangq6/CD_DWI/data/test.list'
    main(test_file)

