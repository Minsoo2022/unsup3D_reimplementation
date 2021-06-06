import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

def get_loader(opt, phase) :
    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
        ])
        shuffle = False
    if opt.dataset == 'CelebA' :
        opt.data_dir = '../unsup3d/data/celeba_cropped/'
        dataset = CelebA_dataset(transform=transform, opt=opt, phase=phase)
    elif opt.dataset == 'Cat':
        opt.data_dir = '../unsup3d/data/cat_combined/'
        dataset = Cat_dataset(transform=transform, opt=opt, phase=phase)
    elif opt.dataset == 'Synface':
        opt.data_dir = '../unsup3d/data/synface/'
        dataset = Synface_dataset(transform=transform, opt=opt, phase=phase)
    else:
        raise NotImplementedError

    if not opt.debug:
        dataloader = data.DataLoader(dataset=dataset,batch_size=opt.batch_size,shuffle=shuffle,num_workers=opt.workers)
    else:
        dataloader = data.DataLoader(dataset=dataset,batch_size=2,shuffle=shuffle,num_workers=0)

    return dataloader

class CelebA_dataset(data.Dataset):
    def __init__(self, transform=None, opt=None, phase=None):
        super(CelebA_dataset,self).__init__()
        self.transform = transform
        data_dir = opt.data_dir
        #self.image_path = os.path.join(data_dir, f'{phase}')
        self.image_path = os.path.join(data_dir, f'paint')
        self.image_list = glob.glob(os.path.join(self.image_path,'*.jpg'))
        self.normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = Image.open(self.image_list[item])
        return self.normalization(self.transform(image))

class Cat_dataset(data.Dataset):
    def __init__(self, transform=None, opt=None, phase=None):
        super(Cat_dataset,self).__init__()
        self.transform = transform
        data_dir = opt.data_dir
        self.image_path = os.path.join(data_dir, f'{phase}')
        self.image_list = glob.glob(os.path.join(self.image_path,'*.jpg'))
        self.image_croping = transforms.CenterCrop(170)
        self.normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = Image.open(self.image_list[item])
        return self.normalization(self.transform(self.image_croping(image)))


class Synface_dataset(data.Dataset):
    def __init__(self, transform=None, opt=None, phase=None):
        super(Synface_dataset,self).__init__()
        self.transform = transform
        data_dir = opt.data_dir
        self.image_path = os.path.join(data_dir, f'{phase}', 'image')
        self.image_list = glob.glob(os.path.join(self.image_path,'*.png'))
        self.image_list.sort()
        self.depth_path = os.path.join(data_dir, f'{phase}', 'depth')
        self.depth_list = glob.glob(os.path.join(self.depth_path, '*.png'))
        self.depth_list.sort()
        self.image_croping = transforms.CenterCrop(170)
        self.image_normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = Image.open(self.image_list[item])
        depth = Image.open(self.depth_list[item])
        return self.image_normalization(self.transform(self.image_croping(image))), self.transform(self.image_croping(depth))