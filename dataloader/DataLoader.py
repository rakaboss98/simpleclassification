import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


class LoadItem(DataLoader):
    def __init__(self):

        # define the data folder
        self.data_dir = "/Users/rakshitbhatt/Documents/GalaxEye /Disease Classification/Potato/"

        # define the different classes in the dataset
        self.data_dict = {'0': "Potato___Early_blight/",
                          '1': "Potato___healthy/",
                          '2': "Potato___Late_blight/"}

        # define the data transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomRotation(degrees=60),

        ])

        # define an empty list containing all the files
        self.files = []

    def __len__(self):
        __, __, files_1 = next(os.walk(self.data_dir + self.data_dict.get('0')))
        self.files.extend(files_1)  # attach all the files to the list
        self.id1 = len(files_1)

        __, __, files_2 = next(os.walk(self.data_dir + self.data_dict.get('1')))
        self.files.extend(files_2)  # attach all the files to the list
        self.id2 = len(files_2)

        __, __, files_3 = next(os.walk(self.data_dir + self.data_dict.get('2')))
        self.files.extend(files_3)  # attach all the files to the list
        self.id3 = len(files_3)

        return self.id1 + self.id2 + self.id3

    def __getitem__(self, idx):
        if idx < self.id1:
            image = Image.open(self.data_dir+self.data_dict.get('0')+self.files[idx])
            image = self.transforms(image)
            return [0, image]
        elif idx < self.id1 + self.id2:
            image = Image.open(self.data_dir + self.data_dict.get('1') + self.files[idx])
            image = self.transforms(image)
            return [1, image]
        else:
            image = Image.open(self.data_dir+self.data_dict.get('2')+self.files[idx])
            image = self.transforms(image)
            return [2, image]