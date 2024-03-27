import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import torch
import random
import os
import codecs
import numpy as np
import random
import pandas as pd
import nibabel as nib
from PIL import Image


class ixi(data.Dataset):



    def __init__(self, indexes, root: str,
            task,  seed,N, data_split_path):
        super().__init__()
        self.paths = []
        self.targets=[]
        self.indexes = []
        self.root_dir = root
        self.task = task
        self.N =N



        check = pd.read_csv(data_split_path + 'df_seed_' + str(seed) + '_n_' +str(N))
        train_files = check['file'].loc[check['split'] =='train']

        if task =='train':
            for i,f in enumerate(train_files):
                self.paths.append(root+ '/normal/' + f + '.nii.gz')
                self.targets.append(torch.FloatTensor([0]))
                self.indexes.append(i)

        else:


            val = os.listdir(root + '/anom/')
            val_files = pd.DataFrame(val).iloc[:,0].apply(lambda x: x.split('.nii.gz')[0]).drop_duplicates().reset_index(drop=True)
            for f in val_files:
                if f not in train_files.tolist():

                    self.paths.append(root+ '/anom/' + f + '.nii.gz')
                    self.targets.append(torch.FloatTensor([1]))



            val = os.listdir(root + '/normal/')
            val_files = pd.DataFrame(val).iloc[:,0].apply(lambda x: x.split('.nii.gz')[0]).drop_duplicates().reset_index(drop=True)
            for f in val_files:
                if f not in train_files.tolist():

                    self.paths.append(root+ '/normal/' + f + '.nii.gz')
                    self.targets.append(torch.FloatTensor([0]))




    def __len__(self):

        return len(self.paths)



    def __getitem__(self, index: int, seed = 1, base_ind=-1):


        base=False
        target = self.targets[index]
        paths = self.paths[index]


        file_path = self.paths[index]
        img = np.array(nib.load(file_path).get_fdata()) / 5012.0

        img = torch.FloatTensor(img)[int(np.ceil(img.shape[0] /2 ) - 10) : int(np.ceil(img.shape[0] /2 ) + 10), : , : ]
        assert img.shape[0] == 20

        img = torch.stack((img,img,img),1)

        if self.task == 'train':
            np.random.seed(seed)
            ind = np.random.randint(len(self.indexes) )
            c=1
            while (ind == index):
                np.random.seed(seed * c)
                ind = np.random.randint(len(self.indexes) )
                c=c+1

            if ind == base_ind:
              base = True

            target2 = int(self.targets[ind])
            file_path2 = self.paths[ind]
            img2 = np.array(nib.load(file_path2).get_fdata()) / 5012.0

            img2 = torch.FloatTensor(img2)[int(np.ceil(img2.shape[0] /2 ) - 10) : int(np.ceil(img2.shape[0] /2 ) + 10), : , : ]

            img2 = torch.stack((img2,img2,img2),1)


            label = torch.FloatTensor([0])
        else:
            img2 = torch.Tensor([1])
            label = target

        return img, img2, label, base, 1, paths[0].split('T2')[0]


