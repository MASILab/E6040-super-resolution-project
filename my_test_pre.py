import torch
import sys
import os
import torch.nn as nn
import argparse
import nibabel as nib
import torch.optim as optim
from torch.utils.data import dataloader
from my_model_loader import dataset
from ecbm6040.model.mDCSRN_WGAN import Generator
from tqdm import tqdm
import numpy as np

def main(options):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers':4,'pin_memory':True} if use_cuda else {}
    test_dataset = dataset(options.test_file)

    lr = 1e-6
    
    model = Generator(1)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=1, 
                                        shuffle=False)
    finished_model_path = options.model_file
    model.load_state_dict(torch.load(finished_model_path)['model_state_dict'])
    model = model.to(device)
    print('begin test')
    test_model(model,test_loader,device,options.save_folder)

def test_model(model,data_loader,device,save_folder):
    model.eval()
    with torch.no_grad():
        for i,sampler in enumerate(data_loader):
            HR = sampler['label']
            LR = sampler['data']
            HR_nii_file = sampler['nii']
            save_nii_file = os.path.join(save_folder + os.path.basename(HR_nii_file[0]))
            print(i,save_nii_file)
            LR = LR.to(device)
            output = model(LR)
            output = np.squeeze(output.cpu().numpy())
            print(output.shape)
            save_nii_img(HR_nii_file,save_nii_file,output)
            # exit(125)

def save_nii_img(HR_nii_file,save_nii_file,output):
    nii = nib.load(HR_nii_file[0])
    save_nii = nib.Nifti1Image(output,nii.affine)
    nib.save(save_nii,save_nii_file)
    
def get_option(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parse command")
    parser.add_argument('--test_file',default='', type=str,help='The file records test data')
    parser.add_argument('--model_file',default='',type=str,help='Which model file you want to load')
    parser.add_argument('--save_folder',default='',type=str,help='Where to save the tested file')

    options = parser.parse_args(args)

    return options


if __name__ == '__main__':
    options = get_option()
    print(options)
    main(options)