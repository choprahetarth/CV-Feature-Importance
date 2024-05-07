import argparse
import cv2
import numpy as np
from torch import nn 
import torch
import torchvision
from torchvision.transforms import Lambda
import pickle
from utils import create_dataloader, train_step, accuracy_fn, create_video, choose_model, get_class_name, create_mask
import matplotlib.pyplot as plt
from tqdm import tqdm
from sewar.full_ref import mse, rmse, psnr, ssim, uqi, scc
from math import sqrt
import cv2
import random

def calculate_difference_average(image_list):
    mse_values = []
    rmse_values = []
    psnr_values = []
    ssim_values = []
    uqi_values = []
    scc_values = []
    print("evaluating...")
    for i in tqdm(range(1, len(image_list))):
        img1 = cv2.resize((image_list[i-1] * 255).astype(np.uint8), (256, 256))
        img2 = cv2.resize((image_list[i] * 255).astype(np.uint8), (256, 256))

        mse_value = mse(img1, img2)
        mse_values.append(mse_value)

        rmse_value = sqrt(mse_value)
        rmse_values.append(rmse_value)

        psnr_value = psnr(img1, img2)
        psnr_values.append(psnr_value)

        ssim_value = ssim(img1, img2)[0]
        ssim_values.append(ssim_value)

        uqi_value = uqi(img1, img2)
        uqi_values.append(uqi_value)

        scc_value = scc(img1, img2)
        scc_values.append(scc_value)

    return mse_values, rmse_values, psnr_values, ssim_values, uqi_values, scc_values

def calculate_difference_normal(image_list):
    mse_values = []
    rmse_values = []
    psnr_values = []

    print("evaluating...")
    for i in tqdm(range(1, len(image_list))):
        img1 = cv2.resize((image_list[i-1] * 255).astype(np.uint8), (256, 256))
        img2 = cv2.resize((image_list[i] * 255).astype(np.uint8), (256, 256))

        mse_value = mse(img1, img2)
        mse_values.append(mse_value)

        rmse_value = sqrt(mse_value)
        rmse_values.append(rmse_value)

        psnr_value = psnr(img1, img2)
        psnr_values.append(psnr_value)

    return mse_values, rmse_values, psnr_values


def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='FashionMNIST', type=str, help='Dataset name')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--use_last_feature_importance', action='store_true', help='Use last feature importance')
    parser.add_argument('--use_average', action='store_true', help='Use average')
    parser.add_argument('--get_video', action='store_true', help='Get video')
    parser.add_argument('--top_n_percent_features', default=0.4, type=float, help='Top N percent features')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--seed', default=42, type=int, help='random_seed')
    args = parser.parse_args()

    # Use the arguments
    dataset_name = args.dataset_name
    use_pretrained = args.use_pretrained
    use_last_feature_importance = args.use_last_feature_importance
    use_average = args.use_average
    get_video = args.get_video
    top_n_percent_features = args.top_n_percent_features
    epochs = args.epochs
    batch_size = args.batch_size
    loss_fn=nn.CrossEntropyLoss()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    if not use_last_feature_importance:
        train_dataloader, train_data = create_dataloader(dataset_name, train=True, batch_size=batch_size)
        test_dataloader, test_data = create_dataloader(dataset_name, train=False, batch_size=batch_size)    # Define class names for SVHN
        class_name = get_class_name(dataset_name, train_data)
        model = choose_model(dataset_name, use_pretrained, train_data, class_name)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        feat_imp_list_all = []
        feat_imp_list_average = []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            model,feat_imp_list_epoch = train_step(model, train_dataloader, test_dataloader, loss_fn, optimizer, accuracy_fn, device, epoch, epochs, calculate_feat_imp=True)
            feat_imp_avg_epoch = np.mean(feat_imp_list_epoch, axis=0)

            if feat_imp_list_epoch is not None and feat_imp_avg_epoch is not None:
                feat_imp_list_all.extend(feat_imp_list_epoch)
                feat_imp_list_average.append(feat_imp_avg_epoch)

        if get_video:
            if use_average:
                print(feat_imp_list_average[0].shape)
                create_video(feat_imp_list_average, name=f'./videos/feat_imp_evolution_{dataset_name}_pretrained_{use_pretrained}_{batch_size}_average')
            else:
                create_video(feat_imp_list_all, name=f'./videos/feat_imp_evolution_{dataset_name}_pretrained_{use_pretrained}_{batch_size}')

        if use_average:
            mse_values, rmse_values, psnr_values, ssim_values, uqi_values, scc_values = calculate_difference_average(feat_imp_list_average)
            plt.figure(figsize=(10, 6))
            plt.plot(mse_values, label='MSE')
            plt.xlabel('Frames')
            plt.ylabel('MSE Values')
            plt.title('MSE as a function of frames')
            plt.legend()
            plt.savefig(f'./plots/mse_values_{dataset_name}_pretrained_{use_pretrained}_{batch_size}_average.png')

            plt.figure(figsize=(10, 6))
            plt.plot(rmse_values, label='RMSE')
            plt.xlabel('Frames')
            plt.ylabel('RMSE Values')
            plt.title('RMSE as a function of frames')
            plt.legend()
            plt.savefig(f'./plots/rmse_values_{dataset_name}_pretrained_{use_pretrained}_{batch_size}_average.png')

            plt.figure(figsize=(10, 6))
            plt.plot(psnr_values, label='PSNR')
            plt.xlabel('Frames')
            plt.ylabel('PSNR Values')
            plt.title('PSNR as a function of frames')
            plt.legend()
            plt.savefig(f'./plots/psnr_values_{dataset_name}_pretrained_{use_pretrained}_{batch_size}_average.png')

            plt.figure(figsize=(10, 6))
            plt.plot(ssim_values, label='SSIM')
            plt.xlabel('Frames')
            plt.ylabel('SSIM Values')
            plt.title('SSIM as a function of frames')
            plt.legend()
            plt.savefig(f'./plots/ssim_values_{dataset_name}_pretrained_{use_pretrained}_{batch_size}_average.png')

            plt.figure(figsize=(10, 6))
            plt.plot(uqi_values, label='UQI')
            plt.xlabel('Frames')
            plt.ylabel('UQI Values')
            plt.title('UQI as a function of frames')
            plt.legend()
            plt.savefig(f'./plots/uqi_values_{dataset_name}_pretrained_{use_pretrained}_{batch_size}_average.png')

            plt.figure(figsize=(10, 6))
            plt.plot(scc_values, label='SCC')
            plt.xlabel('Frames')
            plt.ylabel('SCC Values')
            plt.title('SCC as a function of frames')
            plt.legend()
            plt.savefig(f'./plots/scc_values_{dataset_name}_pretrained_{use_pretrained}_{batch_size}_average.png')
        else:
            mse_values, rmse_values, psnr_values = calculate_difference_normal(feat_imp_list_all)
            plt.figure(figsize=(10, 6))
            plt.plot(mse_values, label='MSE')
            plt.xlabel('Frames')
            plt.ylabel('MSE Values')
            plt.title('MSE as a function of frames')
            plt.legend()
            plt.savefig(f'./plots/mse_values_{dataset_name}_pretrained_{use_pretrained}_{batch_size}.png')

            plt.figure(figsize=(10, 6))
            plt.plot(rmse_values, label='RMSE')
            plt.xlabel('Frames')
            plt.ylabel('RMSE Values')
            plt.title('RMSE as a function of frames')
            plt.legend()
            plt.savefig(f'./plots/rmse_values_{dataset_name}_pretrained_{use_pretrained}_{batch_size}.png')

            plt.figure(figsize=(10, 6))
            plt.plot(psnr_values, label='PSNR')
            plt.xlabel('Frames')
            plt.ylabel('PSNR Values')
            plt.title('PSNR as a function of frames')
            plt.legend()
            plt.savefig(f'./plots/psnr_values_{dataset_name}_pretrained_{use_pretrained}_{batch_size}.png')

        # Save feat_imp_list_all
        with open('feat_imp_list_all.pkl', 'wb') as f:
            pickle.dump(feat_imp_list_all, f)
        torch.save(model.state_dict(), f'pretrained_model_{dataset_name}_{batch_size}_pretrained.pth')

    else:
        with open('feat_imp_list_all.pkl', 'rb') as f:
            feat_imp_list_all = pickle.load(f)

    # Create a mask using the feature importance list
    mask = create_mask(feat_imp_list_all, top_n_percent_features)

    def apply_mask(img):
        return img * mask
    
    masked_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        Lambda(apply_mask)
    ])

    train_dataloader, train_data = create_dataloader(dataset_name, train=True, transform=masked_transform, batch_size=batch_size)
    test_dataloader, test_data = create_dataloader(dataset_name, train=False, transform=masked_transform, batch_size=batch_size)
    image, label = train_data[0]
    class_name = get_class_name(dataset_name, train_data)
    model = choose_model(dataset_name, use_pretrained, train_data, class_name)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model = train_step(model, train_dataloader, test_dataloader, loss_fn, optimizer, accuracy_fn, device, epoch, epochs, calculate_feat_imp=False)

if __name__ == "__main__":

    main()

