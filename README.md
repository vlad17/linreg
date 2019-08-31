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
python -m linreg.main.gendata --n 1000 --p 100 --snr 100 --out ./data/n1000p100snr100.npz

python -m linreg.main.train --help
python -m linreg.main.train --infile ./data/n1000p100snr100.npz --iters 100 --save_every_n 1 --noprecompute
python -m linreg.main.train --infile ./data/n1000p100snr100.npz --iters 10 --save_every_n 1 --precompute


python -m linreg.main.eval --help
```

