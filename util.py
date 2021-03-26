import torch
from torchvision import transforms
import os
from PIL import Image
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, filelist, filepath, transform=None):
        self.filelist = filelist
        self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return int(len(self.filelist))

    def __getitem__(self, index):
        imgpath = os.path.join(self.filepath, self.filelist[index])
        img = Image.open(imgpath)

        if "cat." in imgpath:
            label = 0
        else:
            label = 1
        if self.transform is not None:
            img = self.transform(img)

        return (img, label)

def dataload():
    script_dir = os.path.dirname(__file__)
    train_dir = os.path.join(script_dir, 'train')
    test_dir = os.path.join(script_dir, 'test1')

    train_list = os.listdir(train_dir)  ## train data(.jpg) list
    test_list = os.listdir(test_dir)  ## test data(.jpg) list

    # transform
    transform = transforms.Compose(
        [transforms.Resize((64,64)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train = Dataset(train_list,train_dir,transform)
    test_set = Dataset(test_list,test_dir,transform)
    # train_set_bal = [train.__getitem__(x)[1] for x in range(len(train))]
    # print("train set label : ", train_set_bal)

    # split
    train_set,val_set = torch.utils.data.random_split(train,[20000,5000])
    print("train set size : ", len(train_set))
    print("validation set size : ",len(val_set))
    print("test set size : ",len(test_set))
    return train_set, val_set, test_set

