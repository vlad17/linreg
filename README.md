# linreg: Linear Regression with Vowpal Wabbit

Coming up with successful cluster configurations for VW.

Requirements: [Anaconda 3](https://www.anaconda.com/distribution/) in the path.

Make sure that `conda` is available in your path. Then run `conda activate`.

## Setup

Only needs to be done once:
```
# you may want to modify the "prefix:" line in
# environment.yaml to retarget where the python deps live
conda env create -f environment.yaml
```

Should be done once per session:
```
conda activate linreg-env
```

Use `conda env export --no-builds > environment.yaml` to save new dependencies.

To be done when dependencies update in the yaml file but your local environment is out of date, use `conda env update -f environment.yaml --prune`.

## Usage

All modules are expected to be run from the root directory of the repository

```
python -m linreg.main.gendata --help
python -m linreg.main.train --help
python -m linreg.main.eval --help

./scripts/run.sh  1000 100 100 1000 1000 # n p snr iters_sgd iters_full
```

