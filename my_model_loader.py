import numpy as np
import nibabel as nib
import os
import sys
import torch
from torch.utils.data import Dataset
from glob import glob

class dataset(Dataset):
    def __init__(self,data_file):
        self.data_file = data_file
        self.data = np.loadtxt(self.data_file,dtype=str,delimiter=':')
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,index):
        ind = index % len(list(self.data))
        HR_data_file = self.data[ind,0]
        LR_data_file = self.data[ind,1]
        
        HR_img = nib.load(HR_data_file).get_fdata()
        HR_img = self.normlize(HR_img)
        # print(np.max(HR_img))
    
        HR_img_tensor = torch.from_numpy(HR_img).type(torch.FloatTensor)
        HR_img_tensor = torch.unsqueeze(HR_img_tensor,0)

        LR_img = nib.load(LR_data_file).get_fdata()
        LR_img = self.normlize(LR_img)

        LR_img_tensor = torch.from_numpy(LR_img).type(torch.FloatTensor)
        LR_img_tensor = torch.unsqueeze(LR_img_tensor,0)

        return {'data':LR_img_tensor,'label':HR_img_tensor,'nii':HR_data_file}

    def normlize(self,img):
        thre = np.percentile(img,99)
        img[img >= thre] = thre
        img = img / thre
        return img

def get_data_file():
    HR_dir = '/nfs/masi/yangq6/CD_DWI/data/production/HR'
    LR_dir = '/nfs/masi/yangq6/CD_DWI/data/production/LR'
    train_subj = ['11','12','13','14']
    with open('/nfs/masi/yangq6/CD_DWI/data/train.list','w') as f:
        files = sorted(os.listdir(HR_dir))
        for file in files:
            subj = file.split('_')[0]
            if (subj in train_subj):
                f.write('{}:{}\n'.format(os.path.join(HR_dir,file),os.path.join(LR_dir,file)))

    with open('/nfs/masi/yangq6/CD_DWI/data/valid.list','w') as f:
        files = sorted(os.listdir(HR_dir))
        count = 0
        for i,file in enumerate(files):
            subj = file.split('_')[0]
            ses = file.split('_')[1]
            if (not subj in train_subj):
                count += 1
                if (count < 500):
                    f.write('{}:{}\n'.format(os.path.join(HR_dir,file),os.path.join(LR_dir,file)))

    with open('/nfs/masi/yangq6/CD_DWI/data/test.list','w') as f:
        files = sorted(os.listdir(HR_dir))
        count = 0
        for i,file in enumerate(files):
            subj = file.split('_')[0]
            ses = file.split('_')[1]
            if (not subj in train_subj):
                count += 1
                if (count > 500):
                    f.write('{}:{}\n'.format(os.path.join(HR_dir,file),os.path.join(LR_dir,file)))
    
if __name__ == '__main__':
    get_data_file()
        
        

        
        
    




