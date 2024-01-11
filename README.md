
# DLBS - AgroLuege

This project is a part of the [DLBS - AgroLuege](https://gitlab.fhnw.ch/thomas.mandelz/dlbs-crop-segmentation/) at [Data Science FHNW](https://www.fhnw.ch/en/degree-programmes/engineering/bsc-data-science).

In Switzerland, farmers manually enter crop information on the Agri-Portal for federal payments, verified on-site by the Federal Office for Agriculture. Our goal is to streamline this process using Sentinel-2 satellites and GIS data, providing farmers with a pre-filled crop basis for validation.

This project closely aligns with [Challenge X - AgroLuege](https://gitlab.fhnw.ch/thomas.mandelz/AgroLuege). Within the dlbs module, we test alternative model architectures on smaller datasets, optimizing efficiency with fewer computing resources.

## Project Status: Completed

## Project Intro/Objective

### Methods Used

* Deep Learning
* Machine Learning
* Semantic Segmentation
* Crop Classification

### Technologies

* Python
* PyTorch
* wandb
* numpy
* pandas
* sklearn
* h5py

## Featured Files

* Our Exploratory Data Analysis can be found in this notebook: [Exploratory Data Analysis Notebook](eda/eda.ipynb)
* Our Random Forest baseline training can be found in this notebook: [Exploratory Data Analysis Notebook](modelling/rf.ipynb)
* Our U-Net training can be found in this notebook: [Exploratory Data Analysis Notebook](modelling/U_Net.ipynb)
* Our ms-convSTAR training can be found in this notebook: [Exploratory Data Analysis Notebook](modelling/modelling_ms_convstar_dlbs.ipynb)
    *Hint*: for this notebook to run you may need some additional source files from [CHX-AgroLuege Repository](https://gitlab.fhnw.ch/thomas.mandelz/AgroLuege).

## Getting Started

* Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
* Explorative Dataanalysis Scripts and Files are being kept [here](eda)
* Models Definitions are being kept [here](models)
* Source files for training are being kept [here](modelling)
* SSource Code for deep-learning training pipeline are being kept [here](src)

## Pipenv for Virtual Environment

### First install of Environment

* open `cmd`
* `cd /your/local/github/repofolder/`
* `pipenv install`
* Restart VS Code
* Choose the newly created "tierli_ahluege" Virtual Environment python Interpreter

### Environment already installed (Update dependecies)

* open `cmd`
* `cd /your/local/github/repofolder/`
* `pipenv sync`

## Contributing Members

* **[Thomas Mandelz](https://github.com/tmandelz)**
* **[Daniela Herzig](https://gitlab.fhnw.ch/daniela.herzig)**
