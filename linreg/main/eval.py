"""
Given a file --infile pointing to a npz archive with X and y vectors,
and assuming training has been run for both precomputed and computed
GD, plots stuff nicely.

assuming --infile is input.npz, writes plots

input.npz-time.pdf
input.npz-time-avg.pdf
input.npz-samples.pdf
input.npz-samples-avg.pdf
"""

from absl import app, flags
import os
from time import time
import numpy as np

from .. import log
from ..utils import import_matplotlib

flags.DEFINE_string(
    "infile", None, "the input X, true beta, and y arrays, as a npz file"
)
flags.mark_flag_as_required("infile")

flags.DEFINE_string("out", None, "output prefix")


def _main(_argv):
    log.init()

    infile = flags.FLAGS.infile
    out = flags.FLAGS.out or infile

    values = np.load(flags.FLAGS.infile)
    X, beta_true, y = values["X"], values["beta"], values["y"]
    n, p = X.shape

    plt = import_matplotlib()
    st = np.load(infile + "-trace/time.npy")
    sl = np.load(infile + "-trace/loss_avg.npy")
    ft = np.load(infile + "-tracep/time.npy")
    fl = np.load(infile + "-tracep/loss_avg.npy")

    plt.semilogy(st, sl, ls="--", color="blue", label="stochastic")
    plt.semilogy(ft, fl, color="red", label="precompute")
    plt.legend()
    plt.title("loss v time")
    f = out + "-time.pdf"
    log.debug("writing to {}", f)
    plt.savefig(f, format="pdf", bbox_inches="tight")
    plt.clf()

    st = np.load(infile + "-trace/samples.npy")
    st = st * (p * 3)
    sl = np.load(infile + "-trace/loss_avg.npy")
    ft = np.load(infile + "-tracep/samples.npy")
    ft = np.ones_like(ft)
    ft *= 2 * p ** 2
    ft = np.cumsum(ft)
    fl = np.load(infile + "-tracep/loss_avg.npy")

    plt.loglog(st, sl, ls="--", color="blue", label="stochastic")
    plt.loglog(ft, fl, color="red", label="precompute")
    plt.legend()
    plt.title("loss v flops")
    f = out + "-flop.pdf"
    log.debug("writing to {}", f)
    plt.savefig(f, format="pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    app.run(_main)
