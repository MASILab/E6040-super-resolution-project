import os
import numpy as np
import nibabel as nib
import sys
from tqdm import tqdm

def main(test_file,infer_folder):
    test = np.loadtxt(test_file,dtype=str,delimiter=':')
    infer_folder = '/nfs/masi/yangq6/CD_DWI/infer'
    mses = 0
    for HR_file in tqdm(test[:,0]):
        HR_img = nib.load(HR_file).get_fdata()
        thre = np.percentile(HR_img,99)
        HR_img[HR_img > thre] = thre
        HR_img[HR_img < 0] = 0
        infer_file = os.path.join(infer_folder,'0_' +  os.path.basename(HR_file))
        infer_img = nib.load(infer_file).get_fdata() * thre
        infer_img[infer_img < 0] = 0
        infer_img[infer_img > thre] = thre
        mses += np.square(HR_img - infer_img).mean()
    print('inference',mses / test.shape[0])
    mses = 0
    for i in tqdm(range(test.shape[0])):
        HR_file = test[i,0]
        LR_file = test[i,1]
        LR_img = nib.load(LR_file).get_fdata()
        HR_img = nib.load(HR_file).get_fdata()
        thre = np.percentile(HR_img,99)
        HR_img[HR_img > thre] = thre
        HR_img[HR_img < 0] = 0
        LR_img[LR_img < 0] = 0
        LR_img[LR_img > thre] = thre
        mses += np.square(HR_img - LR_img).mean()
    print('input',mses / test.shape[0])
        
        





if __name__ == '__main__':
    test_file = '/nfs/masi/yangq6/CD_DWI/data/test.list'
    infer_folder = '/nfs/masi/yangq6/CD_DWI/infer'
    main(test_file,infer_folder)