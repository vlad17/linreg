"""
Generates a dataset given
parameters n and p. Uses <SOME DISTRIBUTION>

Writes it to output file specified by --out, by default
in ./data/generated-{n}-{p}.npyz.
"""

from absl import app, flags
import numpy as np

from .. import log

flags.DEFINE_string("out", None, "")


flags.DEFINE_integer("n", 1000, "number of data points to create")


flags.DEFINE_integer("p", 100, "number of (dense) features in each data point")


def _main(_argv):
    log.init()

    n = flags.FLAGS.n
    p = flags.FLAGS.p
    log.debug("creating samples with n = {} p = {}", n, p)

    X = np.random.randn(n, p)
    beta = np.random.randn(p)
    y = X.dot(beta) + np.random.randn(n)

    out = flags.FLAGS.out or f"./data/generated-{n}-{p}.npz"
    log.debug("writing out to {}", out)
    np.savez(out, X=X, beta=beta, y=y)


if __name__ == "__main__":
    app.run(_main)
