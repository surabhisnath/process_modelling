# A Compositional Model of Semantic Fluency

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-enabled-ee4c2c?logo=pytorch&logoColor=white)

This repository contains the data and scripts for the paper: A Compositional Model of Semantic Fluency.

Authors: [Surabhi S Nath](https://surabhisnath.github.io), [Alireza Modirshanechi](https://sites.google.com/view/modirsha), [Peter Dayan](https://www.mpg.de/12309370/biological-cybernetics-dayan)

## Abstract

The ability to recall semantically connected concepts---be it animals, summer fruits, or cities in Italy---is a remarkable capacity of the human mind. Such semantic fluency is thought to rely on traversing a mental space in which concepts are represented in terms of their meanings. However, the structure, properties, and navigability of this representational space remain enigmatic and highly debated. Existing approaches rely either on complex, uninterpretable distributional word-embeddings or on rigid, hand-crafted category norms. Here, we exploit the strengths of both, introducing Conceptome: a version of a compositional, interpretable, feature-based representation of semantic concepts, constructed by leveraging large language models. We use Conceptome to develop Conceptome-search, an auto-regressive model of how humans explore semantic spaces. We validate Conceptome and Conceptome-search using an animal fluency task, showing that they outperform state-of-the-art models in predicting human choices and capture key behavioral patterns such as interference. Our work, hence, offers new insights into the mechanisms underlying semantic fluency and memory retrieval. More broadly, our approach provides a general framework for constructing high-quality representations, with potential applications across cognition, including exploration, navigation, and creative thinking.

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

4. Ablations: `python -u runner.py --nofit --nosimulate --ablation`, prints top 8 most important features for HS and Activity and plots saved as `plots/ablation_HS.png`, `plots/ablation_Activity.png`, `plots/ablation_features.png`

5.  
