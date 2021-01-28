# Active Learning for Single Class Malignant Tumor Detection in Mammograms

## Dataset Preparation
- Obtain the AIIMS data from sumanyu in a particular format.
- Make a `data` directory in the root of this project.
- Put the AIIMS data in this `data` directory.

## Scripts Description
- `aiims_eval.py`: For finding **fROC** on AIIMS test data.
- `train.py`: For training on some percentage of training data.
- `AL.py`: For Active Learning iterations.

## Dependencies
- Anaconda3
- Python>=3.6

## Environment
- Create a conda env from `.yml` specified.
```python
conda create env -f env.yml
```