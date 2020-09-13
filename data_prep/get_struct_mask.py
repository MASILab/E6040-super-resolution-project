import nibabel as nib
import numpy as np
import os
import sys

def main(seg_225_file,seg_555_file,save_225_file,save_555_file):
    img_225_nii = nib.load(seg_225_file)
    img_555_nii = nib.load(seg_555_file)

    ROI_img_225 = img_225_nii.get_fdata()[7:71,13:77,:]
    ROI_pad_img_255 = np.pad(ROI_img_225,((0,0),(0,0),(4,4)),'constant',constant_values=(0,0))
    ROI_img_555 = img_555_nii.get_fdata()[7:71,13:77,:]
    ROI_pad_img_555 = np.pad(ROI_img_225,((0,0),(0,0),(4,4)),'constant',constant_values=(0,0))

    print(ROI_img_555.shape)

    save_ROI_225_nii = nib.Nifti1Image(ROI_pad_img_255,img_225_nii.affine)
    save_ROI_555_nii = nib.Nifti1Image(ROI_pad_img_555,img_555_nii.affine)

    nib.save(save_ROI_225_nii,save_225_file)
    nib.save(save_ROI_555_nii,save_555_file)

if __name__ == '__main__':
    cdmris = ['cdmri0011','cdmri0012','cdmri0013','cdmri0014','cdmri0015']
    for cdmri in cdmris:
        seg_225_file = os.path.join('/nfs/masi/hansencb/CDMRI_2020/challenge_data',cdmri,'seg','2.5_2.5_5_seg.nii.gz')
        seg_555_file = os.path.join('/nfs/masi/hansencb/CDMRI_2020/challenge_data',cdmri,'seg','5iso_seg.nii.gz') 
        save_225_file = os.path.join('/nfs/masi/yangq6/CD_DWI/data/production','{}_225.nii.gz'.format(cdmri))
        save_555_file = os.path.join('/nfs/masi/yangq6/CD_DWI/data/production','{}_555.nii.gz'.format(cdmri))
        main(seg_225_file,seg_555_file,save_225_file,save_555_file)



