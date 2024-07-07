# Generic Title of Project

#### Write a short general description here (optional). Note that you will start from trustworthy machine learning perspective.


## Project Subtitle 1

<div align="justify">Write your abstract or description here. (1-2 justified paragraphs; 300-400 words)</div>

You will add one or two infographics, figures, gifs, or animations here after your description using the following syntax:

- Using an existing image in the Internet:

![alt text](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*leJxy5cZqSxbN5cSSJRpeA.png)

- Or you can link directly to the image if it is stored in your repository:

![alt text](https://github.com/tailabTMU/ECP/blob/main/test/image.webp)

### Related Publications
You can list your related publications (with proper and complete citation information) here. You will use bullet points (-) or numbering for the list items.

## Collaborators

- [Name 1]
  - Academic Position and/or Affiliation
- [Name 2]
  - Academic Position and/or Affiliation

---

## MarkDown File Syntax and Commands


**Required Python Libraries and Packages (with their latest versions) to be Installed (All Already Imported in `utils.py`)**

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

**For downloading and installing ImageNet datasets (*Imagenet-Val* and *Imagenet-V2*):**

`pip install git+https://github.com/modestyachts/ImageNetV2_pytorch `

***

### Python Scripts

1. `utils.py`: This file contains all the imported Python and PyTorch packages, the helper functions, data loading, plots, etc.

- Instead of `your path` and `your inner path` in `os.chdir('your path')` and `root_path = os.path.join(os.getcwd(), 'your inner path')` 
to set `root_path` in the first part of `utils.py` file, you need to set your own path to the your directory containing two empty folders named: 
"data" for downloading and storing the input datasets, and "results" for saving the graphs and figures of results. 
Under "results" folder, you should also create an empty folder named "data", for storing the raw results produced during experiments.

- You can also set your `SEED` for randomness and reproducability in the first part of `utils.py`.
 
2. `loss.py`: This file contains other helper functions related to loss function.

3. `ecp_load.py`: This is a main file to validate the selected pretrained model over Imagenet-Val or Imagenet-V2 datasets, and
print the results for ECP method and other SoTA methods.

4. `adapt_comp.py`: This is a main file to validate all the pretrained models over Imagenet-Val or Imagenet-V2 datasets, and
print the results for set size and adaptiveness metrics in ECP method and other SoTA methods.

- Note that ImageNet-Val_DATASET_SIZE = 50000 and ImageNet-V2_DATASET_SIZE = 10000.
- In `ecp_load.py`, you can choose (uncomment) only one desired pretrained model for validation and applying the CP methods.
- Also, in both `ecp_load.py` and `adapt_comp.py`, you can set/change the values for the following variables:
```
 input_data = 'imagenet'    # OR 'imagenetv2'
 n_calib = 15000    # OR 3000 for imagenetv2
 alpha_val = 0.1    # user-specified coverage error level
```

## Citation
Karimi, H., & Samavi, R. (2024). Evidential Uncertainty Sets in Deep Classifiers Using Conformal Prediction. arXiv preprint [arXiv:2406.10787](https://arxiv.org/abs/2406.10787) ([BibTeX](https://scholar.googleusercontent.com/scholar.bib?q=info:-Xtg9_TC0l8J:scholar.google.com/&output=citation&scisdr=ClHThqE0EInapi4G4T8:AFWwaeYAAAAAZocA-T_QDOFB9Ot3-ZLzwBjva18&scisig=AFWwaeYAAAAAZocA-XghJuYAkw1mPHw1BpbBO1g&scisf=4&ct=citation&cd=-1&hl=en)).
