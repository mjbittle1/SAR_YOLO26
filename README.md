# SARDet-100K Object Detection with YOLOv10

This project trains and evaluates a YOLOv10 object detection model on the SARDet-100K dataset (Synthetic Aperture Radar images). The model is designed to detect 6 classes: Aircraft, Ship, Car, Bridge, Tank, and Harbor.

## Project Structure

* **`train.py`**: Script to train the YOLOv10 model. Automatically logs metrics to Weights & Biases (W&B).
* **`evaluate.py`**: Script to evaluate a trained model on the validation or test data split.
* **`dataset.yaml`**: The YOLO configuration file defining class names and relative paths to the training/validation/test images.
* **`dataset/`**: Directory containing data processing scripts.
  * `convert_coco_to_yolo.py`: Converts original COCO-formatted JSON annotations into YOLO-formatted `.txt` label files.
  * `extract.py`: Extracts the downloaded SARDet-100K dataset.
  * `debug_filenames.py`: A utility script to check for mismatches between image filenames and annotation logs.

## Setup

1. **Install Requirements**: Ensure you have the `ultralytics` (YOLO) and `wandb` packages installed (see `requirements.txt`).
2. **Prepare Data**: 
   * Extract your SARDet-100K dataset inside the `dataset/` directory.
   * Run the conversion script from the `dataset/` folder to generate YOLO labels:
     ```bash
     cd dataset
     python convert_coco_to_yolo.py
     cd ..
     ```

## Usage

### Training

**Local / Single GPU Training (YOLOv10n):**
To begin training the model on the SARDet dataset using the base parameters:

```bash
python train.py
```

**Cluster / Multi-GPU Training (YOLO26 Suite):**
To train the specialized YOLO26 models (`n`, `s`, `m`, `l`, `x`) using Distributed Data Parallel (DDP) on a GPU cluster:

```bash
# Example: Train the small ('s') model on 4 GPUs
python train_YOLO26.py --size s --device "0,1,2,3" --cache
```
*(AutoBatch is enabled by default to automatically maximize your available VRAM, so no `--batch` parameter is necessary).*

### Evaluation

To evaluate a previously trained model (e.g., `best.pt` or `runs/train/yolo/weights/best.pt`) on the test dataset:

```bash
python evaluate.py --model path/to/best.pt
```