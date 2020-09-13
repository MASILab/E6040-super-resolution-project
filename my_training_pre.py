import torch
import sys
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import dataloader
from my_model_loader import dataset
from ecbm6040.model.mDCSRN_WGAN import Generator
from tqdm import tqdm

def main(options):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers':4,'pin_memory':True} if use_cuda else {}
    train_dataset = dataset(options.train_file,options.type)
    valid_dataset = dataset(options.valid_file,options.type)

    lr = 1e-6
    
    model = Generator(1)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=options.batch_size, 
                                        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=options.batch_size,
                                               shuffle=False)
            
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pretrained = '/home-nfs2/local/VANDERBILT/yangq6/myProject/cd_dwi/cdmri/models/pretrained_G_step250000'
    model.load_state_dict(torch.load(pretrained))
    model = model.to(device)
    print('run')
    for epoch in range(options.epoch):
        train_loss = train_model(model,device,epoch,optimizer,criterion,train_loader)
        torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict()},options.model_prefix + '_{}'.format(epoch))
        valid_loss = valid_model(model,device,criterion,epoch,valid_loader)
        with open(options.train_res,'a') as f:
            f.write('{}\n'.format(train_loss))
        with open(options.valid_res,'a') as f:
            f.write('{}\n'.format(valid_loss))


def train_model(model,device,epoch,optimizer,criterion,train_loader):
    model.train()
    total_loss = 0
    with tqdm(total=len(train_loader)) as pbar:
        for i,sampler in enumerate(train_loader):
            HR = sampler['label']
            LR = sampler['data']

            HR,LR = HR.to(device),LR.to(device)
            output = model(LR)
            loss = criterion(HR,output)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_description("Epoch {} Avg Loss {:4f}".format(epoch,total_loss/(i+1)))
            # pbar.update(1)

    return total_loss / len(train_loader)

def valid_model(model,device,criterion,epoch,valid_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(total=len(valid_loader)) as pbar:
            for i,sample in enumerate(valid_loader):
                HR = sample['label']
                LR = sample['data']

                HR,LR = HR.to(device),LR.to(device)
                output = model(LR)
                loss = criterion(HR,output)

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_description("Valid {} Avg Loss {:4f}".format(epoch,total_loss/(i+1)))
                
    return total_loss / len(valid_loader)
def get_option(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parse command")
    parser.add_argument('--train_file',default='',type=str,help='The file records train data')
    parser.add_argument('--valid_file',default='',type=str,help='The file records valid data')
    parser.add_argument('--test_file',default='', type=str,help='The file records test data')
    parser.add_argument('--epoch',default='',type=int,help='How many epochs which we need to train')
    parser.add_argument('--batch_size',default=2,type=int,help='The batch size we need to feed into model')
    parser.add_argument('--model_prefix',default='',type=str,help='Where to store models')
    parser.add_argument('--type',default='',type=str,help='The type is used to decide whether is isotropic or not')
    parser.add_argument('--train_res',default='',type=str,help='The train result file used to save loss')
    parser.add_argument('--valid_res',default='',type=str,help='The valid result file used to save loss')

    options = parser.parse_args(args)

    return options


if __name__ == '__main__':
    options = get_option()
    print(options)
    main(options)