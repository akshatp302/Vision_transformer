import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor



# batch = 32
# epoch = 10
# learning_rate = 0.001
# patch_size = 4
# number_of_classes = 100
# image_size = 32
# num_channels = 3    
embed_dim = 64
# num_heads = 4
# num_layers = 6
# mlp_size = 128
# dropout_rate = 0.1  







class Trainer (nn.Module):
    def __init__(self):
        super().__init__()
        
        self.patch_size = 4
        self.embed_dim = 256
        
        self.projection = nn.Conv2d(in_channels=3,
                                    
                                    out_channels=self.embed_dim,
                                    
                                    kernel_size=self.patch_size,
                                    
                                    stride=self.patch_size)
        
        total_patches = (32 // self.patch_size) ** 2
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        

    pass






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

        

A = Dataprepare(batch_size=32)
A.data_download()
A.data_prepare()


print(A.train_dataset)
print(A.test_dataset)

print(f"the length of the total batches are {len(A.train_prepare)} with batch size {A.batch_size}   ")
print(f"the length of the total batches are {len(A.test_prepare)} with batch size {A.batch_size}   ")

c = next(iter(A.train_prepare))
print(c[0].shape)