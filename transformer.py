import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Dataprepare():
    def __init__(self,batch_size):
        self.batch_size = batch_size
        
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    


    def data_download(self):
        
        self.train_dataset = datasets.CIFAR100(root="data",
                                               train=True,
                                               transform=self.normalize,
                                               download=True,
                                               )
        
        self.test_dataset = datasets.CIFAR100(root="data",
                                              train=False,
                                              transform=self.normalize,
                                              download=True,
                                              ) 
    
    

    def data_prepare (self):
        
        self.train_prepare = DataLoader(dataset=self.train_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        drop_last=False)
        
        self.test_prepare = DataLoader(dataset=self.test_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       drop_last=True)

        

