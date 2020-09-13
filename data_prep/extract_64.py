import numpy as np
import nibabel as nib
import os
import sys

def main(img_path,save_path):
    src_nii = nib.load(img_path)
    src_img = src_nii.get_fdata()
    
    pad_img = src_img[7:71,13:77,:]
    dst_img = np.pad(pad_img,((0,0),(0,0),(4,4)),'constant',constant_values=(0,0))
    print(src_img.shape)

    dst_nii = nib.Nifti1Image(dst_img,src_nii.affine)
    nib.save(dst_nii,save_path)



if __name__ == '__main__':
    img_root = '/nfs/masi/yangq6/CD_DWI/data/scratch/volumes'
    cdmris = ['cdmri0011','cdmri0012','cdmri0013','cdmri0014','cdmri0015']
    for cdmri in cdmris:
        mri_dir = os.path.join(img_root,cdmri,'2.5_5')
        for file in sorted(os.listdir(mri_dir)):
            img_path = os.path.join(mri_dir,file)
            subj = cdmri[-2:]
            save_path = '/nfs/masi/yangq6/CD_DWI/data/production/LR225/{}_{}'.format(subj,file)
            print(img_path,save_path)
            main(img_path,save_path)

    
    # for cdmri in cdmris:
    #     for folder in sorted(os.listdir(os.path.join(img_root,cdmri))):
    #         for file in sorted(os.listdir(os.path.join(img_root,cdmri,folder))):
    #             # print(cdmri,folder,file)
    #             img_path = os.path.join(img_root,cdmri,folder,file)
    #             print(img_path)
    #             img_name = os.path.basename(img_path)
    #             subj = os.path.basename(os.path.dirname(os.path.dirname(img_path)))[-2:]
    #             if ('2.5' in img_path):
    #                 res = 'HR'
    #             else:
    #                 res = 'LR'
    #             save_path = '/nfs/masi/yangq6/CD_DWI/data/production/{}/{}_{}'.format(res,subj,img_name)
    #             main(img_path,save_path)