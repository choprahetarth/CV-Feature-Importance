import cv2
import numpy as np
from models import FashionMNISTModel, SVHNModel, FashionMNISTPretrained, SVHNPretrained
from torch import nn 
import torch
import torchvision
from torchvision.transforms import Lambda
import pickle
from utils import create_dataloader, train_step, accuracy_fn, create_video
import matplotlib.pyplot as plt

# # # Choose dataset
dataset_name = 'FashionMNIST'  # Change this to 'SVHN' for SVHN dataset
use_pretrained=True
use_last_feature_importance=True
get_video = True
top_n_percent_features=0.4
epochs = 20
batch_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if not use_last_feature_importance:
    train_dataloader, train_data = create_dataloader(dataset_name, train=True, batch_size=batch_size)
    test_dataloader, test_data = create_dataloader(dataset_name, train=False, batch_size=batch_size)
    print(f"DataLoaders: {train_dataloader, test_dataloader}")
    print(f"Length of the Train DataLoader: {len(train_dataloader)}")
    print(f"Length of the Test DataLoader: {len(test_dataloader)}")
    image, label = train_data[0]
    # Define class names for SVHN
    if dataset_name == 'SVHN':
        class_name = [str(i) for i in range(10)]  # SVHN has 10 classes, representing digits 0-9
    else:
        class_name = train_data.classes
    # Choose model
    if dataset_name == 'FashionMNIST':
        if use_pretrained:
            model = FashionMNISTPretrained(len(class_name))
        else:
            model = FashionMNISTModel(image.shape[0],10,len(class_name))

    else:
        if use_pretrained:
            model  = SVHNPretrained(len(class_name))
        else:
            model = SVHNModel(len(class_name))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    # Training loop with progress tracking and feature importance calculation
    feat_imp_list_all = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model,feat_imp_list_epoch = train_step(model, train_dataloader, test_dataloader, loss_fn, optimizer, accuracy_fn, device, epoch, epochs, calculate_feat_imp=True)
        if feat_imp_list_epoch is not None:
            feat_imp_list_all.extend(feat_imp_list_epoch)
    # Calculate the variance from the last value in the list
    variances = [np.var(feat_imp_list_all[i-1] - feat_imp_list_all[i]) for i in range(1, len(feat_imp_list_all))]

    # Plot the variance with respect to the index
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(variances) + 1), variances)
    plt.xlabel('Index')
    plt.ylabel('Variance from the previous value')

    # Save the plot
    plt.savefig("variance_plot.png")

    if get_video:
        create_video(feat_imp_list_all, name=f'feat_imp_evolution_{dataset_name}_{batch_size}')

    # Save feat_imp_list_all
    with open('feat_imp_list_all.pkl', 'wb') as f:
        pickle.dump(feat_imp_list_all, f)
    # Save the model
    torch.save(model.state_dict(), f'pretrained_model_{dataset_name}_pretrained.pth')

else:
    with open('feat_imp_list_all.pkl', 'rb') as f:
        feat_imp_list_all = pickle.load(f)

feat_imp = feat_imp_list_all[-1]
# # Assuming feat_imp is a numpy array with shape (1, 28, 28) and already calculated
feat_imp_flat = feat_imp.flatten()  # Flatten the feature importance map

# # Determine the threshold for the top 10%
k = int((1-top_n_percent_features) * len(feat_imp_flat))  
threshold = np.sort(feat_imp_flat)[k]

# # Create a mask for top 10% of features
mask = feat_imp > threshold  # This will be True for top 10%, False otherwise
# mask = mask.astype(float)

def apply_mask(img):
    return img * mask

masked_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    Lambda(apply_mask)
])

train_dataloader, train_data = create_dataloader(dataset_name, train=True, transform=masked_transform, batch_size=batch_size)
test_dataloader, test_data = create_dataloader(dataset_name, train=False, transform=masked_transform, batch_size=batch_size)
print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(f"Length of the Train DataLoader: {len(train_dataloader)}")
print(f"Length of the Test DataLoader: {len(test_dataloader)}")
image, label = train_data[0]
# Define class names for SVHN
if dataset_name == 'SVHN':
    class_name = [str(i) for i in range(10)]  # SVHN has 10 classes, representing digits 0-9
else:
    class_name = train_data.classes
# Choose model
if dataset_name == 'FashionMNIST':
    if use_pretrained:
        model = FashionMNISTPretrained(len(class_name))
    else:
        model = FashionMNISTModel(image.shape[0],10,len(class_name))
else:
    if use_pretrained:
        model  = SVHNPretrained(len(class_name))
    else:
        model = SVHNModel(len(class_name))
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model = train_step(model, train_dataloader, test_dataloader, loss_fn, optimizer, accuracy_fn, device, epoch, epochs, calculate_feat_imp=False)
