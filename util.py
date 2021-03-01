import random
from glob import glob
import os

from PIL import Image
import numpy as np
from skimage.io import imsave, imread
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def save_samples(image_tensor, sample_dir, filename, image_size, num_tiles):
    img = image_tensor.to("cpu").detach()
    img = img.numpy().transpose(0, 2, 3, 1)
    result = np.zeros((image_size*num_tiles, image_size*num_tiles, 3))
    img = img * 255
    
    for i in range(num_tiles):
        for j in range(num_tiles):
            result[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size, :] = img[num_tiles*i+j, :, :, :]
    result = result.astype(np.uint8)
    imsave(f"{sample_dir}/{filename}", result)

def count_pixels(image_tensor):
    _, _, h, w = image_tensor.shape
    return h * w

def split_feature(tensor, type):
    assert type in ["split", "cross"]
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


class ImageDataset(Dataset):
    def __init__(self, params):
        super(ImageDataset, self).__init__()
        data_dir = params["data_dir"]
        dataset_name = params["dataset_name"]
        image_size = params["image_shape"][0]
        self.on_memory = params["on_memory"]
        self.image_files = sorted(glob(os.path.join(data_dir, dataset_name) + '/*.*'))
        #self._size_check(image_size)
        random.seed(42)
        random.shuffle(self.image_files)
        if self.on_memory:
            self.images = [Image.open(f) for f in self.image_files]

        self.transform = transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1),
                            transforms.Resize(int(image_size*1.1)),
                            transforms.CenterCrop(image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor()])
    
    def __getitem__(self, index):
        if not self.on_memory:
            image = Image.open(self.image_files[index])
        else:
            image = self.images[index]
        if image.mode == "L":
            image = image.convert("RGB")
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_files)

    def _size_check(self, image_size):
        image_size = image_size[0]
        new_list = []
        for f in self.image_files:
            image = Image.open(f)
            if min(image.width, image.height) >= image_size:
                new_list.append(f)
        self.image_files = new_list