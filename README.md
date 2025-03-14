# CCNext# Project Title

This project is based on the KITTI and DrivingStereo datasets. It includes scripts for training and evaluating models.

## Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/alelopes/CCNext.git
    cd CCNext
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

**Important:** the requirements assume that a GPU compatible with CUDA 12.1 is available.

## Training

To train the model for the KITTI dataset, run the following command:

```bash
python train.py --batch-size <batch_size> --height <height> --width <width> --epochs <epochs> --dataset KITTI --device <device> --workers <workers> --train-files <train_files> --val-files <val_files> --data-path <data_path> --save-path <save_path>
```

For example:

```bash
python train.py --batch-size 8 --height 384 --width 1280 --epochs 20 --dataset KITTI --device 0 --workers 10 --train-files splits/eigen_full/train_files.txt --val-files splits/eigen_full/val_files.txt --data-path raw/ --save-path out_kitti/
```

To train the model for the DrivingStereo dataset, run the following command:

```bash
python train.py --batch-size <batch_size> --height <height> --width <width> --epochs <epochs> --dataset DrivingStereo --device <device> --workers <workers> --train-files <train_files> --val-files <val_files> --data-path <data_path> --drivingstereo-train-images <train_images> --drivingstereo-train-depth <train_depth> --drivingstereo-val-images <val_images> --drivingstereo-val-depth <val_depth> --save-path <save_path>
```

For example:

```bash
python train.py --batch-size 8 --height 288 --width 640 --epochs 20 --dataset DrivingStereo --device 0 --workers 10 --train-files drivingstereo_splits/stereodrive_train.txt --val-files drivingstereo_splits/stereodrive_val.txt --data-path drivingstereo/ --drivingstereo-train-images StereoDrivingTraining/ --drivingstereo-train-depth depth_maps/train-depth-map --drivingstereo-val-images StereoDrivingTraining/ --drivingstereo-val-depth depth_maps/train-depth-map --save-path out_drivingstereo/
```

Replace the arguments as needed:
- `--batch-size`: Size of each training batch.
- `--height`: Height of the input images.
- `--width`: Width of the input images.
- `--epochs`: Number of training epochs.
- `--dataset`: Specify the dataset to use (`KITTI` or `DrivingStereo`).
- `--device`: Device to run the training on (e.g., `0` for the first GPU).
- `--workers`: Number of data loading workers.
- `--train-files`: Path to the file containing the list of training images.
- `--val-files`: Path to the file containing the list of validation images.
- `--data-path`: Path to the dataset.
- `--save-path`: Path to save the trained model.
- `--drivingstereo-train-images`: Path to the DrivingStereo training images (only for DrivingStereo training).
- `--drivingstereo-train-depth`: Path to the DrivingStereo training depth maps (only for DrivingStereo training).
- `--drivingstereo-val-images`: Path to the DrivingStereo validation images (only for DrivingStereo training).
- `--drivingstereo-val-depth`: Path to the DrivingStereo validation depth maps (only for DrivingStereo training).

**IMPORTANT** We expect the raw kitti dataset to added in a single folder, with all sequences in each individual folder. The structure is:

```
<folder_name>/
└── sequences/
    └── sequences_sync/
        ├── image_02/
        │   └── data/
        ├── image_03/
        │   └── data/
        ├── ...
```

For the DrivingStereo, we expect it to be like:

```
<folder_name>/
└── Training/
    ├── train-left-image/
    ├── train-right-image/
└── Depth_Maps/
    ├── train-depth-maps/
    ├── test-depth-maps/
```

The training set includes all training images and depth maps, which are split into training and validation sets. These splits are available in the `drivingstereo_splits` folder.


## Evaluation

To evaluate the model, run the `evaluate.py` script with the appropriate arguments:

```bash
To evaluate the model for the DrivingStereo dataset, run the `evaluate_drivingstereo_model.py` script with the appropriate arguments:

```bash
python evaluate_drivingstereo_model.py --model-path <model_path> --dataset-path <dataset_path> --filenames <test_files> --drivingstereo-test-images <test_images> --drivingstereo-test-depth <test_depth> --eval-split none --window-size 0.26 --device <device> --reduced-decoder --decoder-path <decoder_path>
```

For example:

```bash
python evaluate_drivingstereo_model.py --model-path drivingstereo_weights/ --dataset-path drivingstereo/ --filenames drivingstereo_splits/stereodrive_test.txt --drivingstereo-test-images StereoDrivingTesting/ --drivingstereo-test-depth depth_maps/test-depth-map --eval-split none --window-size 0.26 --device cuda:0 --reduced-decoder --decoder-path decoder_reduced.pt
```
```

To evaluate the model for the KITTI dataset, run the `evaluate_kitti_model.py` script with the appropriate arguments:

```bash
python evaluate_kitti_model.py --model-path kitti_weights/ --dataset-path raw/ --filenames kitti_splits/eigen/test_files.txt  --gt-path gt_depths.npz --window-size 0.26 --device cuda:0 --reduced-decoder --decoder-path decoder_reduced.pt
```

Replace the arguments as needed:
- `--model-path`: Path to the trained model weights.
- `--dataset-path`: Path to the dataset.
- `--filenames`: Path to the file containing the list of test images.
- `--drivingstereo-test-images`: Path to the DrivingStereo test images (only for DrivingStereo evaluation).
- `--drivingstereo-test-depth`: Path to the DrivingStereo test depth maps (only for DrivingStereo evaluation).
- `--gt-path`: Path to the ground truth depth maps (only for KITTI evaluation).
- `--eval-split`: Evaluation split (none for DrivingStereo evaluation results and you can use the default for garg crop).
- `--window-size`: Size of the windowed cross-attention.
- `--device`: Device to run the evaluation on (e.g., `cuda:0`, `cpu`).
- `--reduced-decoder`: Use a reduced decoder.
- `--decoder-path`: Path to the reduced decoder file.

## License

This project is licensed under the MIT License.

## Acknowledgements

- KITTI dataset
- DrivingStereo dataset
- Parts of the code are based on the [monodepth2](https://github.com/nianticlabs/monodepth2) repository.
- Thanks to the authors of [H-Net]() for sharing the code for some parts of the model