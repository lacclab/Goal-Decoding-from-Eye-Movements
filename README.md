# Decoding Reading Goals from Eye Movements

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

This project aims to decode reading goals (i.e. information-seeking versus ordinary reading) from eye movements using machine learning techniques.

## Getting Started

### Prerequisites

- [Mamba](https://github.com/conda-forge/miniforge#mambaforge) or Conda

### Setup

1. **Clone the Repository**

    Start by cloning the repository to your local machine:

    ```bash
    git clone https://github.com/lacclab/Goal-Decoding-from-Eye-Movements.git
    cd Goal-Decoding-from-Eye-Movements
    ```

2. **Create a Virtual Environment**

    Create a new virtual environment using Mamba (or Conda) and install the dependencies:

    ```bash
    mamba env create -f environment.yaml
    ```

## Reproducing the results

1. To train the models including the full hyperparameter sweep run `bash scripts/sweep_wrapper.sh`. This create sweep configuration files. Run of the created files in the terminal. 

2. Then, to get the predictions on the test sets run `bash scripts/eval_wrapper.sh`.

3. To aggregate and display the results run the `notebooks/display_results_task_decoding.ipynb` notebook.

4. For the error analysis plots run `notebooks/error_analysis.ipynb` and for the statistical tests `stats.ipynb`.

