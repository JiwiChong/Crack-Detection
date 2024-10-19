import torch
import os
import numpy as np
import cv2
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from pathlib import Path
import glob


def augmented_crack_images(crack_image_path_list, nocrack_image_path_list):
    crack_image_path_list_ = []
    factor = len(nocrack_image_path_list) // len(crack_image_path_list)
    for _ in range(factor):
        crack_image_path_list_ += crack_image_path_list
    
    return crack_image_path_list_


def get_mean_std(loader):
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std
# https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4

class SDNETImageDataset(Dataset):
    def __init__(self, augment=None, validate=None, data_part=None, mode=None, transform=None, image_form=None, target_transform=None):

        self.deck_crack_image_path_list = sorted(glob.glob(os.path.join('./dataset/SDNET2018/SDNET2018/D/CD', "*.jpg")))
        self.deck_nocrack_image_path_list = sorted(glob.glob(os.path.join('./dataset/SDNET2018/SDNET2018/D/UD', "*.jpg")))

        self.wall_crack_image_path_list = sorted(glob.glob(os.path.join('./dataset/SDNET2018/SDNET2018/W/CW', "*.jpg")))
        self.wall_nocrack_image_path_list = sorted(glob.glob(os.path.join('./dataset/SDNET2018/SDNET2018/W/UW', "*.jpg")))

        self.pavement_crack_image_path_list = sorted(glob.glob(os.path.join('./dataset/SDNET2018/SDNET2018/P/CP', "*.jpg")))
        self.pavement_nocrack_image_path_list = sorted(glob.glob(os.path.join('./dataset/SDNET2018/SDNET2018/P/UP', "*.jpg")))

        if augment:
            self.deck_crack_image_path_list = augmented_crack_images(self.deck_crack_image_path_list, self.deck_nocrack_image_path_list)
            self.wall_crack_image_path_list = augmented_crack_images(self.wall_crack_image_path_list, self.wall_nocrack_image_path_list)
            self.pavement_crack_image_path_list = augmented_crack_images(self.pavement_crack_image_path_list, self.pavement_nocrack_image_path_list)
        else:
            self.deck_crack_image_path_list = self.deck_crack_image_path_list
            self.wall_crack_image_path_list = self.wall_crack_image_path_list
            self.pavement_crack_image_path_list = self.pavement_crack_image_path_list

        self.entire_crack_image_path_list = self.deck_crack_image_path_list + self.wall_crack_image_path_list + self.pavement_crack_image_path_list
        self.entire_noncrack_image_path_list = self.deck_nocrack_image_path_list + self.wall_nocrack_image_path_list + self.pavement_nocrack_image_path_list
        self.entire_image_path_list = self.entire_crack_image_path_list + self.entire_noncrack_image_path_list

        self.deck_image_path_list = self.deck_crack_image_path_list + self.deck_nocrack_image_path_list
        self.wall_image_path_list = self.wall_crack_image_path_list + self.wall_nocrack_image_path_list
        self.pavement_image_path_list = self.pavement_crack_image_path_list + self.pavement_nocrack_image_path_list

        self.deck_image_labels = [os.path.basename(os.path.dirname(p)) for p in self.deck_image_path_list]
        self.wall_image_labels = [os.path.basename(os.path.dirname(p)) for p in self.wall_image_path_list]
        self.pavement_image_labels = [os.path.basename(os.path.dirname(p)) for p in self.pavement_image_path_list]

        self.deck_label_idxs = [0 if l == 'UD' else 1 for l in self.deck_image_labels]
        self.wall_label_idxs = [0 if l == 'UW' else 1 for l in self.wall_image_labels]
        self.pavement_label_idxs = [0 if l == 'UP' else 1 for l in self.pavement_image_labels]
        self.label_idxs = self.deck_label_idxs + self.wall_label_idxs + self.pavement_label_idxs

        indices = list(range(len(self.entire_image_path_list)))

        if validate:
            split_idx1 = int(np.floor(.8 * len(self.entire_image_path_list)))
            split_idx2 = int(np.floor(.1 * len(self.entire_image_path_list)))
        else:
            split_idx1 = int(np.floor(.75 * len(self.entire_image_path_list)))
            split_idx2 = int(np.floor(.25 * len(self.entire_image_path_list)))

        shuffle_dataset = True
        random_seed = 42
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        if validate:
            train_idx, val_idx, test_idx = indices[:split_idx1], indices[split_idx1:split_idx1+split_idx2], indices[-split_idx2-1:]
        else:
            train_idx, test_idx = indices[:split_idx1], indices[-split_idx2-1:]
        
        pickle.dump(test_idx, open(f'./dataset/SDNET2018/SDNET2018/indexes/{mode}/test_idx.pickle', 'wb'))

        self.data_part = data_part
        if self.data_part == 'train':
            self.entire_image_path_list = [self.entire_image_path_list[i] for i in train_idx] 
        elif self.data_part == 'valid':
            self.entire_image_path_list = [self.entire_image_path_list[i] for i in val_idx] 
        else:
            self.entire_image_path_list = [self.entire_image_path_list[i] for i in test_idx] 

        self.transform = transform
        self.target_transform = target_transform
        self.image_form = image_form
        
    def __len__(self):
        return len(self.entire_image_path_list)

    def __getitem__(self, idx):
        image_path = self.entire_image_path_list[idx]
        
        # TODO
        if self.image_form == 'rgb':
            image = np.array(Image.open(image_path).convert("RGB"))
            image = torch.Tensor(np.moveaxis(image, [0,1,2], [1,2,0]))
        elif self.image_form == 'grayscale':
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = torch.Tensor(np.expand_dims(image, 0))

        label = self.label_idxs[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    


class MetalImageDataset(Dataset):
    def __init__(self, transform=None, image_form=None, target_transform=None):

        # self.img_dir = img_dir
        
        self.accepted_imgs, self.unaccepted_imgs = [], []
        for l in ['a','b']:
            self.accepted_imgs.extend(sorted(glob.glob('./dataset/steel/*/{}.jpg'.format(l))))

        for ul in ['c','d','e']:
            self.unaccepted_imgs.extend(sorted(glob.glob('./dataset/steel/*/{}.jpg'.format(ul))))
        self.entire_image_path_list = self.accepted_imgs + self.unaccepted_imgs 
 
        self.acc_image_labels = [1 for _ in range(len(self.accepted_imgs))]
        self.uacc_image_labels = [0 for _ in range(len(self.unaccepted_imgs))]
        self.label_idxs = self.acc_image_labels + self.uacc_image_labels
        
        self.transform = transform
        self.target_transform = target_transform
        self.image_form = image_form
        
    def __len__(self):
        return len(self.entire_image_path_list) 

    def __getitem__(self, idx):
        image_path = self.entire_image_path_list[idx]
        
        # TODO
        if self.image_form == 'rgb':
            image = np.array(Image.open(image_path).convert('RGB'))
            image = torch.Tensor(np.moveaxis(image, [0,1,2], [1,2,0]))

        elif self.image_form == 'grayscale':
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = torch.Tensor(np.expand_dims(image, 0))
        label = self.label_idxs[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def save_plots(epoch_train_loss, epoch_valid_loss, epochs, run_num):
    plt.figure(figsize=(14, 5))

  # Accuracy plot
    plt.subplot(1, 2, 1)
    train_loss_plot, = plt.plot(epochs, epoch_train_loss, 'r')
    val_loss_plot, = plt.plot(epochs, epoch_valid_loss, 'b')
    plt.title('Training and Validation Loss')
    plt.legend([train_loss_plot, val_loss_plot], ['Training Loss', 'Validation Loss'])
    plt.savefig('./plots/CF0/run_{}/loss_plots.jpg'.format(run_num))


def save_plots_wo_validation(epoch_train_loss, epochs, run_num):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    train_loss_plot, = plt.plot(epochs, epoch_train_loss, 'r')
    plt.title('Training Loss')
    plt.legend([train_loss_plot], ['Training Loss'])
    plt.savefig('./plots/CF0/run_{}/loss_plots.jpg'.format(run_num))
    
    
def save_checkpoint(save_path, model, loss, val_used=None):
    if save_path == None:
        return

    loss_txt = 'val_loss' if val_used else 'train_loss'
    state_dict = {'model_state_dict': model.state_dict(),
                  loss_txt: loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
    

def calculate_metric(y, y_pred):
    metric = accuracy_score(y, y_pred)
    return metric