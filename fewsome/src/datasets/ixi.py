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
        self.data=[]
        self.N =N
        normals = os.listdir(root+ '/normal/')
        normals_ids = pd.DataFrame(normals.copy()).iloc[:,0].apply(lambda x: x.split('_')[2])
        normals_ids.columns=['file']
        normals_ids=normals_ids.drop_duplicates().reset_index(drop=True)

        random.seed(seed)
        ind = random.sample(range(0, len(normals_ids)), N)
        train_files = np.array(normals_ids)[ind] #names of training

        check = pd.read_csv(data_split_path + 'df_seed_' + str(seed) + '_n_' +str(N))
        train_files = check['file'].loc[check['split'] =='train']


        if task =='train':
            for i,f in enumerate(train_files):
                path_temp=[]
                for file in normals:
                    if  f in file.split('_')[2] : #
                        path_temp.append(root+ '/normal/' + file)
                self.paths.append(path_temp)
                self.targets.append(torch.FloatTensor([0]))
                self.data.append(f)
                self.indexes.append(i)

        else:


            val = os.listdir(root + '/anom/')
            val_files = pd.DataFrame(val).iloc[:,0].apply(lambda x: x.split('_')[2]).drop_duplicates().reset_index(drop=True)
            for f in val_files:
                if f not in train_files:

                    path_temp = []
                    for file in val:
                        if f in file:
                            path_temp.append(root+ '/anom/' + file)

                    self.data.append(f)
                    self.paths.append(path_temp)
                    self.targets.append(torch.FloatTensor([1]))



            val = os.listdir(root + '/normal/')
            val_files = pd.DataFrame(val).iloc[:,0].apply(lambda x: x.split('_')[2]).drop_duplicates().reset_index(drop=True)
            for f in val_files:
                if f not in train_files:
                    path_temp = []
                    for file in val:
                        if f in file: #
                            path_temp.append(root+ '/normal/' + file)

                    self.data.append(f)
                    self.paths.append(path_temp)
                    self.targets.append(torch.FloatTensor([0]))




    def __len__(self):

        return len(self.paths)



    def __getitem__(self, index: int, seed = 1, base_ind=-1):


        base=False
        target = self.targets[index]
        paths = self.paths[index]
        images=[]
        for i,file_path in enumerate(paths):
            images.append(np.asarray(Image.open(file_path)))

        img = np.stack(images, axis =0)
        if img.shape[0] >= 20:
            img = torch.FloatTensor(img)[10:30,:,:]
        else:
            img = torch.FloatTensor(img)

        img = torch.stack((img,img,img),1)

        if self.task == 'train':
            np.random.seed(seed)
            ind = np.random.randint(len(self.data) )
            c=1
            while (ind == index):
                np.random.seed(seed * c)
                ind = np.random.randint(len(self.indexes) )
                c=c+1

            if ind == base_ind:
              base = True

            target2 = int(self.targets[ind])
            paths = self.paths[ind]
            images=[]
            for i,file_path in enumerate(paths):
                images.append(np.asarray(Image.open(file_path)))

            img2 = np.stack(images, axis =0)
            if img2.shape[0] >= 20:
                img2 = torch.FloatTensor(img2)[10:30,:,:]
            else:
                img2 = torch.FloatTensor(img2)

            img2 = torch.stack((img2,img2,img2),1)


            label = torch.FloatTensor([0])
        else:
            img2 = torch.Tensor([1])
            label = target

        return img, img2, label, base, 1
