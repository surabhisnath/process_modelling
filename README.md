# A Compositional Model of Semantic Fluency

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-enabled-ee4c2c?logo=pytorch&logoColor=white)

This repository contains the data and scripts for the paper: A Compositional Model of Semantic Fluency.
Authors: [Surabhi S Nath](https://surabhisnath.github.io), [Alireza Modirshanechi](https://sites.google.com/view/modirsha), [Peter Dayan](https://www.mpg.de/12309370/biological-cybernetics-dayan)

## Abstract

Humans frequently demonstrate semantic fluency---for example while recalling names of summer fruits or cities in Italy. The semantic fluency task, of recalling examplars from a category, is commonly used to gain insight into the interaction of language and executive cognitive functioning. Previous research proposes several representations to capture the structure and organization of semantic memory, which are then useful for describing the mechanism underlying search and retrieval in fluency tasks. Two common examples include distributional representations and category norms. These representations however, are limited---distributional representations, acquired using pretrained (language) models, are uninterpretable, while category norms are hand-crafted, rigid and are rarely scalable. We propose a novel representation, which uses large language models to generate high-quality interpretable embeddings, composing the power of distributional representations and category norms. We operationalize measures based on our representation which relate to task dynamics. Using the measures, we develop a generative model for the semantic fluency task which outperforms existing models, and supports a deeper understanding of the underlying process.

## Repository Description

1. `csvs` contain the csv data files
2. `figures` contain the final figures used in the paper
3. `files` contain all config and auxillary files used in the code 
4. `fits` contain the model fit pickles
5. `models` contain the model classes and the main runner script
6. `scripts` contain code for all auxillary analysis and figure plotting
7. `simulations` contain the model simulation pickles

## Setup

We recommend setting up a python virtual environment and installing all the requirements. Please follow these steps:

```bash
git clone https://github.com/surabhisnath/process_modelling.git
cd process_modelling

python3 -m venv .env

# On macOS/Linux
source .env/bin/activate
# On Windows
.env\Scripts\activate

pip install -r requirements.txt
```

## Running the code

`models/runner.py` is the main runner script for the following analyses. All settings can be set using arguments of `runner.py`.
Before you being, ensure all models you wish to run are set to 1 in `files/model_to_run`.

1. Model Fitting and Simulation: `python models/runner.py` (can add `--nofit` or `--nosimulate` if either is not desired).
Plotting NLLs: `python scripts/model_NLLs.py`, plot saved as `plots/model_nlls.png`
Plotting BLEUs: `python scripts/model_BLEUs.py`, plot saved as `plots/model_bleus.png`

2. Model Recovery: `python -u runner.py --nofit --nosimulate --recovery`

3. Parameter Recovery: `python -u runner.py --nofit --nosimulate --parameterrecovery`

4. Ablations: `python -u runner.py --nofit --nosimulate --ablation`, prints top 8 most important features for HS and Activity and plots saved as `plots/ablation_HS.png`, `plots/ablation_Activity.png`, `plots/ablation_features.png`

5.  
