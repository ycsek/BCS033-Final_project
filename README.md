# BCS033-Final_project: Defense Against Member Inference Attacks and Explainability Analysis Based on DP-SGD

## Environment Dependencies

This project uses Conda for environment management.

* **Python**: `3.12.12`
* **PyTorch**: `torch==2.6.0+cu126`, `torchvision==0.21.0+cu126`

##  Installation & Setup

1. Clone this repo：
   ```bash
   git clone git@github.com:ycsek/BCS033-Final_project.git
   cd BCS033-Final_project
   ```
2. Create environment:
    ```bash
    conda env create -f environment.yaml
    conda activate janus
    ```

## Usage
```bash
python main.py --config config.yaml
```