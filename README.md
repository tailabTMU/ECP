# Evidential Uncertainty Sets in Deep Classifiers Using Conformal Prediction
#### Paper: Hamed Karimi, and Reza Samavi. "Evidential Uncertainty Sets in Deep Classifiers Using Conformal Prediction." 
##### Accepted in The 13th Symposium on Conformal and Probabilistic Prediction with Applications (COPA 2024).

## Getting Started
This is a brief description on the implementation files for the paper.

### Prerequisites

*** Required Python Libraries and Packages (with their latest versions) to be Installed (All Already Imported in `utils.py`) ***

- `torch`
- `torchvision`
- `scipy`
- `numpy`
- `matplotlib`
- `sklearn`
- `tqdm`
- `random`
- `os`
- `multiprocessing`

***For downloading and installing ImageNet datasets (Imagenet-Val and Imagenet-V2):***
`pip install git+https://github.com/modestyachts/ImageNetV2_pytorch `

***

### Python Scripts

1. `utils.py`: This file contains all the imported Python and PyTorch packages, the helper functions, data loading, plots, etc.

- Instead of "your path" and "your inner path" in "os.chdir('your path')" and "root_path = os.path.join(os.getcwd(), 'your inner path')" 
to set "root_path" in the first part of utils.py file, you need to set your own path to the your directory containing two empty folders named: 
"data" for downloading and storing the input datasets, and "results" for saving the graphs and figures of results. 
Under "results" folder, you should also create an empty folder named "data", for storing the raw results produced during experiments.
- You can also set your "SEED" for randomness and reproducability in the first part of utils.py.
 
2. `loss.py`: This file contains other helper functions related to loss function.

3. `ecp_load.py`: This is a main file to validate the selected pretrained model over Imagenet-Val or Imagenet-V2 datasets, and
print the results for ECP method and other SOTA methods.

4. `adapt_comp.py`: This is a main file to validate all the pretrained models over Imagenet-Val or Imagenet-V2 datasets, and
print the results for set size and adaptiveness metrics in ECP method and other SOTA methods.

- Note that ImageNet-Val_DATASET_SIZE = 50000 and ImageNet-V2_DATASET_SIZE = 10000.
- In "ecp_load.py", you can choose (uncomment) only one desired pretrained model for validation and applying the CP methods.
- Also, in both `ecp_load.py` and `adapt_comp.py`, you can set/change the values for the following variables: 
	- `input_data = 'imagenet' # OR 'imagenetv2'`
	- `n_calib = 15000 # OR 3000 for imagenetv2`
	- `alpha_val = 0.1 # user-specified coverage error level`




