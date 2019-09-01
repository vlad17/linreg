n="$1"
p="$2"
snr="$3"
iters_sgd="$4"
iters_full="$5"
save_every_sgd="$6"
save_every_full="$7"

python -m linreg.main.gendata --n "$n" --p "$p" --snr "$snr" --out "./data/n${n}p${p}snr${snr}.npz"
python -m linreg.main.train --infile "./data/n${n}p${p}snr${snr}.npz" --iters $iters_sgd --save_every_n "$save_every_sgd" --noprecompute
python -m linreg.main.train --infile "./data/n${n}p${p}snr${snr}.npz" --iters $iters_full --save_every_n "$save_every_full" --precompute
python -m linreg.main.eval --infile "./data/n${n}p${p}snr${snr}.npz"
