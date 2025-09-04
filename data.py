import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]
        return image, mask


def get_data(type_, image_nums):

    data = []

    for image_num in image_nums:

        image = cv2.imread(f'./images-1024x768/{type_}/image-{image_num}.png')
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype = torch.float32)
        image = image / 255.0

        mask = cv2.imread(f'./masks-1024x768/{type_}/mask-{image_num}.png', cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype('int')
        mask = torch.tensor(mask, dtype = torch.long)
        data.append([image, mask])

    return data

def view(image):

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    plt.imshow(image, cmap = "binary")
    plt.colorbar()
    plt.title("View Image")
    plt.show()

