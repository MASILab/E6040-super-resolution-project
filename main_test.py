from ecbm6040.model.mDCSRN_WGAN import Generator
import torch 
import re

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# Create the generator
ngpu = 1
netG = Generator(ngpu).cuda(device)
# Print the model
print(netG)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
pretrained = '/home-nfs2/local/VANDERBILT/yangq6/myProject/cd_dwi/E6040-super-resolution-project/models/pretrained_G_step250000'
netG.load_state_dict(torch.load(pretrained))
step = int(re.sub("\D", "", pretrained)) 
print(step)
