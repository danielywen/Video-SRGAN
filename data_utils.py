from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, transforms
import torch
import os


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, sequence_length=5):
        super(TrainDatasetFromFolder, self).__init__()
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.sequence_length = sequence_length
        self.image_filenames = []
        
        # Collect all image filenames
        for dirpath, dirnames, filenames in os.walk(dataset_dir):
            filenames = [os.path.join(dirpath, f) for f in filenames if f.endswith(('.png', '.jpg', '.jpeg'))]
            filenames.sort()
            if len(filenames) >= self.sequence_length:
                self.image_filenames.extend([filenames[i:i + self.sequence_length] for i in range(0, len(filenames) - self.sequence_length + 1)])

        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image_sequence = self.image_filenames[index]
        lr_sequence = []
        hr_sequence = []

        for image_path in image_sequence:
            hr_image = Image.open(image_path)
            lr_image = hr_image.resize(
                (hr_image.width // self.upscale_factor, hr_image.height // self.upscale_factor), 
                Image.BICUBIC
            )

            if self.transform:
                hr_image = self.transform(hr_image)
                lr_image = self.transform(lr_image)

            hr_sequence.append(hr_image)
            lr_sequence.append(lr_image)

        lr_sequence = torch.stack(lr_sequence)
        hr_sequence = torch.stack(hr_sequence)

        return lr_sequence, hr_sequence

    def __len__(self):
        return len(self.image_filenames)
    
class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, sequence_length=5):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.sequence_length = sequence_length
        self.image_filenames = []
        
        # Collect all image filenames
        for dirpath, dirnames, filenames in os.walk(dataset_dir):
            filenames = [os.path.join(dirpath, f) for f in filenames if f.endswith(('.png', '.jpg', '.jpeg'))]
            filenames.sort()
            if len(filenames) >= self.sequence_length:
                self.image_filenames.extend([filenames[i:i + self.sequence_length] for i in range(0, len(filenames) - self.sequence_length + 1)])

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        image_sequence = self.image_filenames[index]
        lr_sequence = []
        hr_sequence = []

        for image_path in image_sequence:
            hr_image = Image.open(image_path)
            lr_image = hr_image.resize(
                (hr_image.width // self.upscale_factor, hr_image.height // self.upscale_factor), 
                Image.BICUBIC
            )

            if self.transform:
                hr_image = self.transform(hr_image)
                lr_image = self.transform(lr_image)

            hr_sequence.append(hr_image)
            lr_sequence.append(lr_image)

        lr_sequence = torch.stack(lr_sequence)
        hr_sequence = torch.stack(hr_sequence)

        return lr_sequence, hr_sequence

    def __len__(self):
        return len(self.image_filenames)

    
# class TrainDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, crop_size, upscale_factor):
#         super(TrainDatasetFromFolder, self).__init__()
#         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
#         crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
#         self.hr_transform = train_hr_transform(crop_size)
#         self.lr_transform = train_lr_transform(crop_size, upscale_factor)

#     def __getitem__(self, index):
#         hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
#         lr_image = self.lr_transform(hr_image)
#         return lr_image, hr_image

#     def __len__(self):
#         return len(self.image_filenames)


# class ValDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(ValDatasetFromFolder, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

#     def __getitem__(self, index):
#         hr_image = Image.open(self.image_filenames[index])
#         w, h = hr_image.size
#         crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
#         lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
#         hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
#         hr_image = CenterCrop(crop_size)(hr_image)
#         lr_image = lr_scale(hr_image)
#         hr_restore_img = hr_scale(lr_image)
#         return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

#     def __len__(self):
#         return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        # self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        # self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.lr_path = "/data/daniel/CSE244C/RealVSR/RealVSR/LQ_test_one_dir"
        self.hr_path = "/data/daniel/CSE244C/RealVSR/RealVSR/GT_test_one_dir"
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
