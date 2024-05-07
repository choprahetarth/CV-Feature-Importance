@echo off
setlocal

set "batch_sizes_1=8 64 256 1024 2048"
set "batch_sizes_2=8 1024 4096"
set "percentage_features=0.1 0.3 0.6 0.8 0.9"
set "seed=10 42 64"

for %%B in %batch_sizes_1% do (
    echo ==============================Starting Ablation Study 1 (Resnet/Average/Video) batch_size=%%B ==================== >> training_log_new.txt
    python testing-script.py --dataset_name FashionMNIST --use_pretrained --use_average --get_video --top_n_percent_features 0.4 --epochs 30 --batch_size %%B >> training_log_new.txt

    echo ==============================Starting Ablation Study 2 (Normal/Average/Video) batch_size=%%B ==================== >> training_log_new.txt
    python testing-script.py --dataset_name FashionMNIST --use_average --get_video --top_n_percent_features 0.4 --epochs 30 --batch_size %%B >> training_log_new.txt

    echo ==============================Starting Ablation Study 3 (Resnet/Average/Video/SVHN) batch_size=%%B ==================== >> training_log_new.txt
    python testing-script.py --dataset_name SVHN --use_pretrained --use_average --get_video --top_n_percent_features 0.4 --epochs 30 --batch_size %%B >> training_log_new.txt
)

for %%B in %batch_sizes_2% do (
    echo ==============================Starting Ablation Study 4 (Resnet/NonAverage/Video) batch_size=%%B ==================== >> training_log_new.txt
    python testing-script.py --dataset_name FashionMNIST --use_pretrained --get_video --top_n_percent_features 0.4 --epochs 5 --batch_size %%B >> training_log_new.txt
)

for %%s in %seed% do (
    echo ==============================Starting Ablation Study 5 (Resnet/NonAverage/Video) batch_size=%%B ==================== >> training_log_new.txt
    python testing-script.py --dataset_name FashionMNIST --use_pretrained --get_video --top_n_percent_features 0.4 --epochs 5 --batch_size 1024 --seed%%s >> training_log_new.txt
)


for %%N in %percentage_features% do (
    echo ==============================Starting Ablation Study 6 (Normal/NonAverage/NoVideo/SVHN) percentage_features=%%N ==================== >> training_log_new.txt
    python testing-script.py --dataset_name SVHN --top_n_percent_features %%N --epochs 20 --batch_size 64 >> training_log_new.txt

    echo ==============================Starting Ablation Study 7 (Normal/NonAverage/NoVideo/FashionMNIST) percentage_features=%%N ==================== >> training_log_new.txt
    python testing-script.py --dataset_name FashionMNIST --top_n_percent_features %%N --epochs 20 --batch_size 64 >> training_log_new.txt

    echo ==============================Starting Ablation Study 8 (Resnet/NonAverage/NoVideo/SVHN) percentage_features=%%N ==================== >> training_log_new.txt
    python testing-script.py --dataset_name SVHN --use_pretrained --top_n_percent_features %%N --epochs 20 --batch_size 64 >> training_log_new.txt

    echo ==============================Starting Ablation Study 9 (Resnet/NonAverage/NoVideo/FashionMNIST) percentage_features=%%N ==================== >> training_log_new.txt
    python testing-script.py --dataset_name FashionMNIST --use_pretrained --top_n_percent_features %%N --epochs 20 --batch_size 64 >> training_log_new.txt
)

endlocal