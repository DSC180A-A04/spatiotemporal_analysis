# Spatiotemporal Analysis

The analysis in this repo uses uncertainty quantification feature that we have developed in [torchTS](https://github.com/Rose-STL-Lab/torchTS).

Example output:

We make inference with `[0.05, 0.5, 0.95]` confidence levels.
![uncertainty_quantification](./static/uncertainty_quantification.png)

## Getting Started

1. Create a virtual environment

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Train models and make predictions

```bash
python run.py
```
