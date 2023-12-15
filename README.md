# Encrypted LLM Inference

## Installation

This repository was developed and tested using Python 3.8.6 on Markov.

First, clone this repository and set up a virtual environment.

To install dependencies:
```bash
pip install git+https://github.com/sarojaerabelli/py-fhe.git
pip install numpy
```

## Usage
Example:
```bash
python main.py --algorithm row_parallel --matrix-size 128 --block-size 16
```

Note that both parallel algorithms will assume access to a number of threads equal to `NUM_WORKERS` (set to 32 in `main.py`). 
Be sure to run on a machine with at least this many threads available, or change this value.
