# Surgical Agent

This repository contains the training and data processing pipeline for a surgical agent. The agent processes surgery videos (e.g., from CholecT50) to recognize actions using a rendezvous model, and then segments the relevant tools and tissues using a SegFormer-B2 model trained on CholecSeg8k.

## Training Performance & Logs

We use [Weights & Biases (WandB)](https://wandb.ai) to track our model training and evaluation metrics. 

You can view the training curves, hyperparameters, and evaluation metrics for our **best performing SegFormer-B2 run** here:

* **[https://wandb.ai/vai-jss-university/surgical-agent-segformer/runs/djqur8be/overview?nw=nwuserimvaibhavrana]**

## Setup and Installation

1. Install dependencies for the data pipeline:
   ```bash
   pip install -r data/requirements.txt
   ```

2. Install dependencies for the training pipeline:
   ```bash
   pip install -r training/requirements.txt
   ```

3. Configure your environment:
   - Copy `.env.example` to `.env`
   - Add your `WANDB_API_KEY` to the `.env` file to enable logging.

## Usage

Each script can be run from the command line with the appropriate arguments. Use the `--help` flag for detailed usage instructions.

### Data Splitting
```bash
python data/data_splitting.py --dataset_root /path/to/cholecseg8k --output_dir splits/
```

### Data Exploration
```bash
python data/data_exploration.py --train_json splits/train.json --test_json splits/test.json --output_dir dataset_analysis/
```

### Training
```bash
python training/train_seg.py --train_json splits/train.json --test_json splits/test.json --save_dir segformer_finetuned/
```

## References & Acknowledgements
- **Rendezvous Model**: Action recognition is powered by the [Rendezvous Model](https://github.com/CAMMA-public/rendezvous). Nwoye et al., "Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos." Medical Image Analysis, 78 (2022) 102433.
- **Datasets**: 
  - [CholecT50](https://camma.u-strasbg.fr/datasets/) for surgical action triplets.
  - **CholecSeg8k** for semantic segmentation of surgical tools and tissues.
