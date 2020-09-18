import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation 

def main(volume_file,save_file,max_value,min_value):
    img = nib.load(volume_file).get_fdata()
    num_slices=img.shape[2]
    print(num_slices)
    list_of_slices = []
    fig = plt.figure()
    fig.tight_layout()
    for i in range(num_slices):
        plt.axis('off')
        a_slice = plt.imshow(img[:,:,i].T, 
                        cmap="gray", 
                         animated=True,
                         vmax=max_value,
                         vmin=min_value
                        )
        
        list_of_slices.append([a_slice])
        
    brain_animation = ArtistAnimation(fig, list_of_slices, interval=10, blit=True,
                                repeat_delay=1000)

    brain_animation.save(save_file)
    plt.show()

if __name__ == '__main__':
    # label_file = '/nfs/masi/yangq6/CD_DWI/data/scratch/volumes/cdmri0015/5_iso/0500.nii.gz'
    label_file='/nfs/masi/yangq6/CD_DWI/infer/cdmri_0015_volume_555/0500.nii.gz'
    # label_file = '/nfs/masi/yangq6/CD_DWI/data/scratch/volumes/cdmri0015/2.5_5/0500.nii.gz'
    save_file = '5output0500.gif'
    # max_value = np.max(nib.load(label_file).get_fdata())
    # min_value = np.min(nib.load(label_file).get_fdata())
    max_value = 1
    min_value = 0
    print(max_value,min_value)
    main(label_file,save_file,max_value,min_value)
