# Single Image Super Resolution

A brief description of what the project does and who it's for.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Run Commands](#run-commands)

## Installation

Instructions for setting up the environment, including dependencies. We recommend creating a virtual enviroment.

```bash
# Clone the repository
git clone https://github.com/alonhelvits/Single-Image-Super-Resolution.git

# Navigate into the project directory
cd yourproject

# Install dependencies (if applicable)
pip install -r requirements.txt
```

## Usage

Download the processed dataset folder from here : https://drive.google.com/drive/u/2/folders/125CbInmLhFyBo0fOgFij4nayQDvXt15Z

From the same link you can also download the CBSR model checkpoints to test the CBSR model pipeline.

- **Project**
  - All python files from this repo
  - dataset
    - combined_500
      - train
      - val
      - test
  - classifier_100_epochs.pth
  - man_made_800_model.pth
  - nature_800_model.pth

## Run Commands
To run a full training of a baseline model:
```bash
python3 main.py --baseline --dataset_path dataset/combined_500
```

If you downloaded the 3 model checkpoints:
  - classifier_100_epochs.pth
  - man_made_800_model.pth
  - nature_800_model.pth

You can run the test on the data:
```bash
python3 main.py --CBSR --dataset_path dataset/combined_500
```
