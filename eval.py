import torch
import os 
import argparse
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
from utils import SDNETImageDataset, MetalImageDataset, save_plots, save_checkpoint, calculate_metric, get_mean_std
from models import CNN_0FC, LeNet5, Ensemble, MultiScaleNet
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, roc_auc_score
print(torch.cuda.is_available())
import warnings
warnings.filterwarnings("ignore")


def config():
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--input_size', type=int, default=128, help='batch size')
    parser.add_argument('--run_num', type=str, help='run number')
    parser.add_argument('--validate', type=bool, default=False, help='whether to validate the model or not')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--data', type=str, help='Dataset')
    parser.add_argument('--n_classes', type=int, default = 2, help='Num of classes')
    parser.add_argument('--rgb', type=bool, default=True, help='Is image RGB or not')
    parser.add_argument('--image_form', type=str, help='Image form')
    parser.add_argument('--augment', type=bool, default=False, help='whether augment dataset or not')
    parser.add_argument('--half', type=bool, default=False, help='whether use half channel size or not')
    parser.add_argument('--num_workers', type=bool, help='number of workers')
    parser.add_argument('--use_cbam', type=bool, default=False ,help='use CBAM for CF0 or not')
    parser.add_argument('--first_channel', type=bool, default=64, help='Channel dim of first feature')
    parser.add_argument('--first_dim', type=bool, default=62, help='Img dim of first feature')
    parser.add_argument('--second_channel', type=bool, default=128, help='Channel dim of second feature')
    parser.add_argument('--second_dim', type=bool, default=29, help='Img dim of second feature')

    args = parser.parse_args()
    return args

# Testing model
if __name__ == '__main__':
    args = config()

    # data = 'sdnet'
    test_transformer = transforms.Compose([
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Resize((args.input_size, args.input_size))
                                        ])
    
    if args.data == 'sdnet':
        mode = 'augmented' if args.augment else 'regular'
        dataset = SDNETImageDataset(
                                    augment=args.augment, validate=args.validate, data_part='test', mode= mode, 
                                    transform = test_transformer, image_form=args.image_form, target_transform=None)

        test_dataloaders = DataLoader(dataset, batch_size= args.batch_size, num_workers=args.num_workers)# had sampler =

    elif args.data == 'metal':
        dataset = MetalImageDataset(
                                    transform = test_transformer, image_form = 'rgb', target_transform=None)

        test_dataloaders = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # print('testing with dataset of size:', len(test_dataloaders))
    # print('data used is:', args.data)
    # print('Model used is:', args.model_name)
    # print('Run number used is:', args.run_num)
    # print('Image form is:', args.image_form)
    # print('RGB? :', args.rgb)
    # print('Are we validating?', args.validate)
    # print('Half model:', args.half)
    # print('Using CBAM:', args.use_cbam)

    # criterion = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model_name == 'Lenet':
        model = LeNet5(args.n_classes, rgb=args.rgb).to(device)
    elif args.model_name == 'CF0':
        model = CNN_0FC(args = args).to(device)
    elif args.model_name == 'ensemble':
        model = Ensemble(args.n_classes, rgb=args.rgb).to(device)
    elif args.model_name == 'MultiScaleNet':
        model = MultiScaleNet().to(device)


    model.load_state_dict(torch.load('./saved_models/{}/best_model_num_{}.pt'.format(args.model_name, args.run_num), map_location=device)['model_state_dict'])
    
    model.eval()
    all_y, all_y_hat = [], []
    for batch in test_dataloaders:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            # Add batch to GPU
            outputs = model(inputs)
            y = labels.detach().cpu().numpy().tolist()
            y_pred = outputs.detach().cpu().numpy().tolist()
            all_y.extend(y)
            all_y_hat.extend(y_pred)


    final_all_y_pred = np.argmax(all_y_hat, axis=1)
    print(f'Test Accuracy is: {round(calculate_metric(all_y, final_all_y_pred), 2)}')
    print(f'Test Precision Score is: {round(precision_score(all_y, final_all_y_pred), 2)}')
    print(f'Test ROC AUC Score is: {round(roc_auc_score(all_y, final_all_y_pred), 2)}')
    print(f'Test F1 Score is: {round(f1_score(all_y, final_all_y_pred), 2)}')
    



# command:
# python eval.py --run_num 1 --model_name MultiScaleNet --data sdnet --image_form rgb --num_workers 10 