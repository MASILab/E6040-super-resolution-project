import numpy as np
import os
import torch
import sys
import os
import torch.nn as nn
from ecbm6040.model.mDCSRN_WGAN import Generator
from tqdm import tqdm
import nibabel as nib
import argparse

def main(options):
    test_file_arrs = np.loadtxt(options.test_file,dtype=str)
    for test_file in tqdm(test_file_arrs):
        volume_dir = get_patch_by_model(test_file,options)
        # volume_dir = '/nfs/masi/yangq6/CD_DWI/infer/cdmri_0015_volume_225/0500'
        merge_volume(test_file,volume_dir)
        exit(125)

def merge_volume(test_file,test_dir):
    nii = nib.load(test_file)
    x,y,z = nii.get_fdata().shape
    template = np.zeros([x,y,z+8]) + 1
    merge = np.zeros([x,y,z+8])
    template[7:71,13:77,4:60] = 0
    for i in range(2):
        for j in range(2):
            xy_range = get_range(template.shape,i,j)
            tmp = template[xy_range[0]:xy_range[0] + 64,xy_range[1]:xy_range[1]+64,:]
            basename = os.path.basename(test_dir)
            patch_img = nib.load(os.path.join(test_dir,'{}_{}_{}.nii.gz'.format(i,j,basename))).get_fdata()
            merge[xy_range[0]:xy_range[0] + 64,xy_range[1]:xy_range[1]+64,:] += tmp * patch_img
    
    merge[7:71,13:77,:] = nib.load(os.path.join(test_dir,'2_2_{}.nii.gz'.format(os.path.basename(test_dir)))).get_fdata()
    save_file =os.path.join(os.path.dirname(test_dir),basename + '.nii.gz')
    save_nii = nib.Nifti1Image(merge[:,:,4:60],nii.affine,nii.header)
    nib.save(save_nii,save_file)


def get_patch_by_model(test_file,options):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    model = Generator(1).to(device)
    model.load_state_dict(torch.load(options.model_file)['model_state_dict'])
    model.eval()
    print('Begin to evaluate model')
    with torch.no_grad():
        for i in range(2):
            for j in range(2):
                input_tensor, thre = get_input_tensor_by_id(test_file,i,j,options)
                output = torch.squeeze(model(input_tensor.to(device)).cpu()).numpy()
                save_file = get_save_file(test_file,options,i,j)
                save_nii_img(test_file,save_file,output)
                print(i,j)
        input_tensor,thre = get_input_tensor_by_id(test_file,2,2,options)
        output = torch.squeeze(model(input_tensor.to(device)).cpu()).numpy()
        save_file = get_save_file(test_file,options,2,2)
        save_nii_img(test_file,save_file,output)
        print(2,2)
    return os.path.dirname(save_file)
                

def get_save_file(test_file,options,i,j):
    basename = os.path.basename(test_file).split('.')[0]
    prefix = '{}_{}_'.format(i,j)
    save_sub_dir = os.path.join(options.save_folder,basename)
    save_file = os.path.join(save_sub_dir,prefix + basename + '.nii.gz') 
    if (not os.path.isdir(save_sub_dir)):
        os.mkdir(save_sub_dir)
    return save_file


def save_nii_img(ref_file,save_file,img):
    nii = nib.load(ref_file)
    save_nii = nib.Nifti1Image(img,nii.affine)
    nib.save(save_nii,save_file)

def get_input_tensor_by_id(test_file,i,j,options):
    mask = nib.load(options.mask_file).get_fdata()
    test_img = nib.load(test_file).get_fdata()
    thre = np.percentile(test_img[mask == 3],99.99)
    test_img = test_img / thre
    test_img = np.pad(test_img,((0,0),(0,0),(4,4)),'constant',constant_values=(0,0))
    test_img[test_img >= 1] = 1
    if (i ==2 and j == 2):
        input_tensor = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(test_img[7:71,13:77,:]),0),0).type(torch.FloatTensor)
        return input_tensor, thre
    xy_range = get_range(test_img.shape,i,j)
    input_tensor = get_input_tensor(test_img,xy_range)
    return input_tensor,thre

def get_input_tensor(img,xy_range):
    input_img = img[xy_range[0]:xy_range[0]+64,xy_range[1]:xy_range[1] + 64,:]
    input_tensor = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(input_img),0),0)
    return input_tensor.type(torch.FloatTensor)

def get_range(shape,i,j):
    x_start = [0,shape[0]-64]
    y_start = [0,shape[1]-64]
    return x_start[i],y_start[j]
    
    
def get_option(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parse command")
    parser.add_argument('--test_file',default='', type=str,help='The file contains complete volume')
    parser.add_argument('--model_file',default='',type=str,help='Which model file you want to load')
    parser.add_argument('--save_folder',default='',type=str,help='Where to save the tested file')
    parser.add_argument('--type',default='',type=str,help='isotropic or not isotropic')
    parser.add_argument('--mask_file',default='',type=str,help='the structure mask which can be used to determine threshold')

    options = parser.parse_args(args)

    return options

if __name__ == '__main__':
    options = get_option()
    print(options)
    main(options)

