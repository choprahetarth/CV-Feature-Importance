import numpy as np
import pandas as pd
import os
import torch 
import torchvision 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import FashionMNISTModel
from torch import nn 
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from models import FashionMNISTModel, SVHNModel, FashionMNISTPretrained, SVHNPretrained
import cv2

def create_dataloader(dataset_name, root='data', train=True, download=True, transform=torchvision.transforms.ToTensor(), target_transform=None, batch_size=128, shuffle=False):
    if dataset_name == 'SVHN':
        data = getattr(torchvision.datasets, dataset_name)(root=root,
                                                           split='train' if train else 'test',
                                                           download=download,
                                                           transform=transform,
                                                           target_transform=target_transform)
    else:
        data = getattr(torchvision.datasets, dataset_name)(root=root,
                                                           train=train,
                                                           download=download,
                                                           transform=transform,
                                                           target_transform=target_transform)
    dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)
    return dataloader, data

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(model, train_data_loader, test_data_loader, loss_fn, optimizer, accuracy_fn, device, epoch, epochs, calculate_feat_imp=False):
    model.train()
    train_loss, train_acc = 0, 0
    feat_imp_list = []


    loop = tqdm(enumerate(train_data_loader), total=len(train_data_loader), leave=False)
    for batch, (X, y) in loop:
        X, y = X.to(device), y.to(device)

        if calculate_feat_imp and X.requires_grad == False:
            X.requires_grad = True

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        if calculate_feat_imp:
            # Perform feature importance calculation before backward pass
            with torch.no_grad():
                gradients = torch.autograd.grad(outputs=y_pred, inputs=X, grad_outputs=torch.ones_like(y_pred), only_inputs=True, retain_graph=True)[0]
                feat_imp = torch.abs(gradients).mean(dim=0)
                feat_imp_list.append(feat_imp.cpu().numpy())

        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item(), acc=accuracy_fn(y, y_pred.argmax(dim=1)))

    train_loss /= len(train_data_loader)
    train_acc /= len(train_data_loader)
    print(f"\nTrain Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}%")

    # Evaluation on test data
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in test_data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_acc += accuracy_fn(y, y_pred.argmax(dim=1))
    test_loss /= len(test_data_loader)
    test_acc /= len(test_data_loader)
    print(f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")

    return (model, feat_imp_list) if calculate_feat_imp else model


def create_video(feat_imp_list,name):
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # Infer the image size from the first feature importance map in the list
    image_size = (feat_imp_list[0].shape[1], feat_imp_list[0].shape[2])
    video = cv2.VideoWriter(f'{name}.mp4', fourcc, 18, image_size)

    for feat_imp in feat_imp_list:
        image2 = feat_imp.squeeze()
        image2 = cv2.normalize(image2, None, 255,0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Check if the image is grayscale or RGB
        if feat_imp.shape[0] == 1:  # Grayscale
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        elif feat_imp.shape[0] == 3:  # RGB
            # Transpose the image to get the channels last
            image2 = np.transpose(image2, (1, 2, 0))
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR format
        else:
            raise ValueError("Unsupported number of channels. The image should be either grayscale or RGB.")
        
        image2 = cv2.applyColorMap(image2, cv2.COLORMAP_JET)
        video.write(image2)

    video.release()



def choose_model(dataset_name, use_pretrained, train_data=None, class_name=None):
    if dataset_name == 'FashionMNIST':
        if use_pretrained:
            model = FashionMNISTPretrained(len(class_name))
        else:
            image, label = train_data[0]
            model = FashionMNISTModel(image.shape[0],10,len(class_name))
    else:
        if use_pretrained:
            model  = SVHNPretrained(len(class_name))
        else:
            model = SVHNModel(len(class_name))
    return model


def get_class_name(dataset_name, train_data=None):
    if dataset_name == 'SVHN':
        class_name = [str(i) for i in range(10)]  # SVHN has 10 classes, representing digits 0-9
    else:
        class_name = train_data.classes
    return class_name


def create_mask(feat_imp_list_all, top_n_percent_features):
    feat_imp = feat_imp_list_all[-1]
    # Assuming feat_imp is a numpy array with shape (1, 28, 28) and already calculated
    feat_imp_flat = feat_imp.flatten()  # Flatten the feature importance map
    # Determine the threshold for the top 10%
    k = int((1-top_n_percent_features) * len(feat_imp_flat))  
    threshold = np.sort(feat_imp_flat)[k]
    # Create a mask for top 10% of features
    mask = feat_imp > threshold  # This will be True for top 10%, False otherwise
    return mask