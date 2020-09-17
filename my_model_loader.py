import numpy as np
import nibabel as nib
import os
import sys
import torch
from torch.utils.data import Dataset
from glob import glob

class dataset(Dataset):
    def __init__(self,data_file,type):
        self.data_file = data_file
        self.data = np.loadtxt(self.data_file,dtype=str,delimiter=':')
        self.type = type
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,index):
        ind = index % len(list(self.data))
        HR_data_file = self.data[ind,0]
        LR_data_file = self.data[ind,1]

        LR_img, HR_img, thre = self.normalize(HR_data_file,LR_data_file)

        HR_img_tensor = torch.from_numpy(HR_img).type(torch.FloatTensor) 
        HR_img_tensor = torch.unsqueeze(HR_img_tensor,0)

        LR_img_tensor = torch.from_numpy(LR_img).type(torch.FloatTensor)
        LR_img_tensor = torch.unsqueeze(LR_img_tensor,0)

        return {'data':LR_img_tensor,'label':HR_img_tensor,'nii':HR_data_file,'thre':thre}

    def normalize_divide_thre(self,HR_data_file,LR_data_file):
        basename = os.path.basename(HR_data_file)
        subj = basename.split('_')[0]
        prefix = 'cdmri00'
        anat_mask_file = '/nfs/masi/yangq6/CD_DWI/data/production/{}{}_{}.nii.gz'.format(prefix,subj,self.type)
        mask = nib.load(anat_mask_file).get_fdata()
        LR_img = nib.load(LR_data_file).get_fdata()
        HR_img = nib.load(HR_data_file).get_fdata()

        thre = self.cal_thre(LR_img,mask)

        LR_img = LR_img / thre
        HR_img = HR_img / thre
        # print(np.max(HR_img),np.max(LR_img))

        return LR_img, HR_img, thre

    def normalize(self,HR_data_file,LR_data_file):
        basename = os.path.basename(HR_data_file)
        subj = basename.split('_')[0]
        prefix = 'cdmri00'
        anat_mask_file = '/nfs/masi/yangq6/CD_DWI/data/production/{}{}_{}.nii.gz'.format(prefix,subj,self.type)
        mask = nib.load(anat_mask_file).get_fdata()
        LR_img = nib.load(LR_data_file).get_fdata()
        HR_img = nib.load(HR_data_file).get_fdata()

        thre = self.cal_thre(LR_img,mask)

        LR_img = LR_img / thre
        LR_img[LR_img > 1] = 1
        HR_img = HR_img / thre
        HR_img[HR_img > 1] = 1
        # print(np.max(HR_img),np.max(LR_img))

        return LR_img, HR_img, thre

    def cal_thre(self,LR_img,mask):
        thre = np.percentile(LR_img[mask==3],99.99)
        return thre

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
    # get_data_file()
    data_file = '/nfs/masi/yangq6/CD_DWI/data/train_225.list'
    type = '225'
    data = dataset(data_file,type)
    for i in range(100):
        print(data[i]['nii'])
        
        

        
        
    




