import torch
import os 
import glob
import gc
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils import SDNETImageDataset, save_plots, save_plots_wo_validation, save_checkpoint, get_mean_std
from models import CNN_0FC, LeNet5, Ensemble, MultiScaleNet
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")


def config():
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--input_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,help='learning rate')
    parser.add_argument('--run_num', type=str, help='run number')
    parser.add_argument('--validate', type=bool, default=False, help='whether to validate the model or not')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--epochs', type=int, help='Num of epochs')
    parser.add_argument('--n_classes', type=int, default=2, help='Num of classes')
    parser.add_argument('--rgb', type=bool, help='Is image RGB or not')
    parser.add_argument('--half', type=bool, default=False, help='use half Model size or not')
    parser.add_argument('--augment', type=bool, default=False, help='whether augment dataset or not')
    parser.add_argument('--use_cbam', type=bool, default=False, help='use CBAM for CF0 or not')
    parser.add_argument('--first_channel', type=bool, default=64, help='Channel dim of first feature')
    parser.add_argument('--first_dim', type=bool, default=62, help='Img dim of first feature')
    parser.add_argument('--second_channel', type=bool, default=128, help='Channel dim of second feature')
    parser.add_argument('--second_dim', type=bool, default=29, help='Img dim of second feature')
    args = parser.parse_args()
    return args

# Function to run in case we are validating the model during training 
def run_with_validation(args, model, train_dataloaders, valid_dataloaders, optimizer, plot_path, writer):

    # Training loop
    epochs = []                
    epoch_train_loss = []
    epoch_valid_loss = []
    best_loss = np.inf
    best_path = './saved_models/{}'.format(args.model_name) + '/' + 'best_model_num_{}.pt'.format(args.run_num)
    loss_func = nn.BCELoss()
    print('------------  Training started! --------------')
    num_epochs = args.epochs
    for epoch in tqdm(range(num_epochs)):
        model.train()

        b_train_loss = []
        b_train_y = []
        b_train_y_hat = []
        for i, (inputs, labels) in enumerate(train_dataloaders):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            outputs = model(inputs)

            new_label_tensor = np.zeros((len(labels), 2))
            for i, e in enumerate(list(labels)):
                new_label_tensor[i,int(e)] = 1
            labels__ = torch.tensor(new_label_tensor, dtype=torch.float32)
        
            loss = loss_func(outputs, labels__.to(device))
            b_train_loss.append(loss.item())

            del inputs
            del labels
            torch.cuda.empty_cache()
            gc.collect()
            
            loss.backward()
            optimizer.step()
        
        epoch_train_loss.append(np.mean(b_train_loss))
        print('Epoch: {}'.format(epoch+1))
        print('Training Loss: {}'.format(np.mean(b_train_loss)))

        model.eval()

        with torch.no_grad():
            b_valid_loss = []
            b_valid_y = []
            b_valid_y_hat = []
            for i, (inputs, labels) in enumerate(valid_dataloaders):
                inputs = inputs.to(device)

                outputs = model(inputs)
                new_label_tensor = np.zeros((len(labels), 2))
                for i, e in enumerate(list(labels)):
                    new_label_tensor[i,int(e)] = 1
                labels__ = torch.tensor(new_label_tensor, dtype=torch.float32)
                val_loss = loss_func(outputs, labels__.to(device))

                b_valid_loss.append(val_loss.item())

            print('Validation Loss {}'.format(np.mean(b_valid_loss)))
            print('-' * 40)
            epoch_valid_loss.append(np.mean(b_valid_loss))
            
            if np.mean(b_valid_loss) < best_loss:
                best_loss = np.mean(b_valid_loss)
                save_checkpoint(best_path, model, np.mean(b_valid_loss), args.validate)
        epochs.append(epoch+1)

    print("Training complete!")
    save_plots(epoch_train_loss, epoch_valid_loss, epochs, args.run_num)

# Function to run in case we are NOT validating the model during training 
def run_without_validation(args, model, train_dataloaders, optimizer, plot_path, writer):

    # Training loop
    epochs = []                
    epoch_train_loss = []
    best_loss = np.inf
    best_path = './saved_models/{}'.format(args.model_name) + '/' + 'best_model_num_{}.pt'.format(args.run_num)
    loss_func = nn.CrossEntropyLoss()
    print('------------  Training started! --------------')
    num_epochs = args.epochs
    for epoch in tqdm(range(num_epochs)):
        model.train()

        b_train_loss = []
        b_train_y = []
        b_train_y_hat = []
        for i, (inputs, labels) in enumerate(train_dataloaders):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            outputs = model(inputs)

            new_label_tensor = np.zeros((len(labels), 2))
            for i, e in enumerate(list(labels)):
                new_label_tensor[i,int(e)] = 1
            labels__ = torch.tensor(new_label_tensor, dtype=torch.float32)
        
            loss = loss_func(outputs, labels__.to(device))
            b_train_loss.append(loss.item())

            del inputs
            del labels
            torch.cuda.empty_cache()
            gc.collect()
            
            loss.backward()
            optimizer.step()

        if np.mean(b_train_loss) < best_loss:
            best_loss = np.mean(b_train_loss)
            save_checkpoint(best_path, model, np.mean(b_train_loss), args.validate)
        
        epoch_train_loss.append(np.mean(b_train_loss))
        print('Epoch: {}'.format(epoch+1))
        print('Training Loss: {}'.format(np.mean(b_train_loss)))

        epochs.append(epoch+1)

    print("Training complete!")
    save_plots(epoch_train_loss, epochs, args.run_num)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = config()
    mode = 'augmented' if args.augment else 'regular'

    # print('Torch Cuda is available:', torch.cuda.is_available())
    # print('Model is:', args.model_name)
    # print('data mode is {} !!'.format(mode))
    # print('To be trained for {} epochs !!'.format(args.epochs))
    # print('Are we validating?:', args.validate)

    train_transformer = transforms.Compose([
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Resize((args.input_size, args.input_size))
                                        ])
    
    val_transformer = transforms.Compose([
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Resize((args.input_size, args.input_size))
                                        ])
    

    batch_size = 8
    if args.validate:
        train_dataset = SDNETImageDataset(augment=args.augment, validate=args.validate, data_part='train' , mode= mode, 
                                    transform = train_transformer, image_form='rgb', target_transform=None)
        val_dataset = SDNETImageDataset(augment=args.augment, validate=args.validate, data_part='valid' , mode= mode, 
                                    transform = val_transformer, image_form='rgb', target_transform=None)
        
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, sampler=None, num_workers=10)
        valid_dataloaders = DataLoader(val_dataset, batch_size=args.batch_size, sampler=None, num_workers=10)
    else:
        train_dataset = SDNETImageDataset(augment=args.augment, validate=args.validate, data_part='train' , mode= mode, 
                                    transform = train_transformer, image_form='rgb', target_transform=None)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, sampler=None, num_workers=10)
        
    
 
    plot_path=f'./plots/{args.model_name}/run_{args.run_num}/'
    model_path=f'./saved_models/{args.model_name}/run_{args.run_num}/'
    if not os.path.exists(path=plot_path) :
        os.mkdir(plot_path) 
        os.mkdir(model_path) 
    else:
        print(f"The files {plot_path} and {model_path} already exist.")


    if args.model_name == 'Lenet':
        model = LeNet5(args = args).to(device)
    elif args.model_name == 'CF0':
        model = CNN_0FC(args=args).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    elif args.model_name == 'ensemble':
        model = Ensemble(args = args).to(device)
    elif args.model_name == 'MultiScaleNet':
        model = MultiScaleNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)


    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = None
    if args.validate:
        run_with_validation(args, model, train_dataloaders, valid_dataloaders, optimizer, plot_path, writer)
    else:
        run_without_validation(args, model, train_dataloaders, optimizer, plot_path, writer)
   

# command:
# python main_dev.py --run_num 1 --model_name MultiScaleNet --epochs 30 --rgb True