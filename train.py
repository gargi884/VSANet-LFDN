import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from utils.LFDataset import LoadTrainData
from utils.Functions import *
from utils.DeviceParameters import to_device
from model.vsanet import VSANet

import mat73
from skimage import metrics 
import itertools,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
os.makedirs('./log',exist_ok=True)
fh = logging.FileHandler('./log/Training_%s.log' % datetime.now().strftime("%Y%m%d%H%M"))
log.addHandler(fh)


# Training settings
parser = argparse.ArgumentParser(description="Hybrid Light Field Restoration")
parser.add_argument("--model_name", type=str, default='VSANet', help="Path for saving training log ")
parser.add_argument("--learningRate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--sigma", type=int, default=10, help="Noise level")
parser.add_argument("--angResolution", type=int, default=5, help="The angular resolution of original LF")

parser.add_argument("--batchSize", type=int, default=4, help="Batch size")
parser.add_argument("--sampleNum", type=int, default=70, help="The number of LF in training set")
parser.add_argument("--patchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--epochNum", type=int, default=10000, help="The number of epoches")
parser.add_argument("--num_steps", type=int, default=2500, help="The number of step size reduce learning rate")

parser.add_argument("--summaryPath", type=str, default='./log/', help="Path for saving training log ")
parser.add_argument("--saveCheckpointsDir", type=str, default='./checkpoints/', help="Path for saving training log ")
parser.add_argument("--dataPath", type=str, default='datasets/train_noiseLevel_10-20-50_4-11_color_5x5.mat', help="Path for loading training data ")

parser.add_argument('--resume',type=str, default = False, help='Resume training from saved checkpoint(s).')
parser.add_argument('--modelPath',type=str, default = './checkpoints/model_sigma_50.pth', help='Resume training from saved checkpoint(s).')
parser.add_argument('--optimizerPath',type=str, default = './checkpoints/optimizer.pth', help='Resume optimizer from saved optimizer(s).')

parser.add_argument("--n_groups", type=int, default=5, help="The number of HGAG groups")
parser.add_argument("--n_blocks", type=int, default=5, help="The number of HFEB blocks")
parser.add_argument("--n_channels", type=int, default=32, help="The number of convolution filters")

opt = parser.parse_args()
logging.info(opt)

# make save ckpt
saveCheckpointsDir = opt.saveCheckpointsDir + '/' + opt.model_name + '_' + str(opt.sigma)
if not os.path.exists(saveCheckpointsDir): 
        os.makedirs(saveCheckpointsDir) 

if __name__ == '__main__':

    lfDataset = LoadTrainData(opt)
    dataloader = DataLoader(lfDataset, batch_size=opt.batchSize,shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model=VSANet(opt)
    
    
    to_device(model,device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Training parameters: %d" %total_trainable_params)

    criterion = torch.nn.L1Loss() # Loss 
    
    optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=opt.learningRate) #optimizer
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.num_steps, gamma=0.5)
    writer = SummaryWriter(opt.summaryPath)

    if opt.resume:
        print('Resume training from' + opt.modelPath)
        model.load_state_dict(torch.load(opt.modelPath),strict=False)

        optimizerPath = torch.load(opt.optimizerPath)
        optimizer.load_state_dict(optimizerPath['optimizer_state_dict'])
        epoch_start = optimizerPath['epoch'] + 1
        model.train()
    else:
        print('Start training from scratch.')
        epoch_start = 1
    
    lossLogger = defaultdict(list)

    for epoch in range(epoch_start,opt.epochNum):
        batch = 0
        lossSum = 0
        for _,sample in enumerate(dataloader):
            batch = batch +1
            LF=sample['LFPatch']
            noiLF=sample['noiLFPatch']
            LF = to_device(LF,device) # label:[u v c x y] 
            noiLF= to_device(noiLF,device)  # input:[u v c x/s y/s]

            b,u,v,k,h,w,c = noiLF.shape
            noiLF = noiLF.permute(0, 1, 2, 4, 5, 6,3).contiguous()
            noiLF = noiLF.view(b, u, v,h,w,c*k)
            noiLF = noiLF.permute(0, 1, 2, 5, 3, 4).contiguous()

            LF = LF.permute(0, 1, 2, 4, 5, 6,3).contiguous()
            LF = LF.view(b, u, v,h,w,c*k).contiguous() 
            LF = LF.permute(0, 1, 2, 5, 3, 4).contiguous()
            
            estimatedLF=model(noiLF)
            loss = criterion(estimatedLF,LF)
            
            lossSum += loss.item()            
            
            writer.add_scalar('loss', loss, opt.sampleNum//opt.batchSize*epoch+batch)
            print("Epoch: %d Batch: %d Loss: %.6f" %(epoch,batch,loss.item()))
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            save_ckpt_path = saveCheckpointsDir + '/model_sigma_%d_epoch_%02d.pth' % (
                opt.sigma, epoch + 1)
            torch.save(model.state_dict(),save_ckpt_path)

            save_optimizer_path = saveCheckpointsDir + '/optimizer.pth'
            torch.save({'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict()},save_optimizer_path)
     
        log.info("Epoch: %d Loss: %.6f" %(epoch,lossSum/len(dataloader)))
        scheduler.step()
        
        #Record the training loss
        lossLogger['Epoch'].append(epoch)
        lossLogger['Loss'].append(lossSum/len(dataloader))
        plt.figure()
        plt.title('Loss')
        plt.plot(lossLogger['Epoch'],lossLogger['Loss'])
        plt.savefig('./log/Training_{}.jpg'.format(opt.sigma))
        plt.close()
    

