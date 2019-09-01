"""
Given a file --infile pointing to a npz archive with X and y vectors,
this kicks off an iterative process to find an estimate for the
linear regression parameters.

Intermediate steps are saved to a directory in the same location
as --infile, suffixed with '-trace'. I.e., if --infile is

./data/input.npz

./data/input.npz-trace/time.npy
./data/input.npz-trace/samples.npy
./data/input.npz-trace/loss.npy
./data/input.npz-trace/loss_avg.npy

This procedure runs several "iterations" which are just gradient steps
(stochastic or full). For every saved iteration, this prints:

time.npy - wall clock time, in seconds, to reach that step
samples.npy - number of samples seen by this point
loss.npy - loss at that point
loss_avg.npy - loss of average iterate at that point

Finally, the flags used for the training process are written out into

./data/input.npz-trace/flags.txt

if "precompute" is selected, we write out the same files but to

*-tracep/*
"""

from absl import app, flags
import os
from time import time
import numpy as np

from .. import log

flags.DEFINE_string(
    "infile", None, "the input X, true beta, and y arrays, as a npz file"
)
flags.mark_flag_as_required("infile")

flags.DEFINE_boolean(
    "precompute", False, "use precomputed X^TX approach vs SGD"
)

flags.DEFINE_integer(
    "iters", 100, "maximum number of seconds to run (if set to 0, no limit)"
)

flags.DEFINE_integer("save_every_n", 1, "save every n")


def _main(_argv):
    log.init()

    suffix = "-tracep" if flags.FLAGS.precompute else "-trace"

    os.makedirs(flags.FLAGS.infile + suffix, exist_ok=False)

    values = np.load(flags.FLAGS.infile)
    X, beta_true, y = values["X"], values["beta"], values["y"]
    n, p = X.shape

    beta = np.zeros_like(beta_true)

    time_elapsed = 0
    num_writes = 0
    samples_seen = 0
    samples_seen_at_write = []
    time_at_write = []
    loss_at_write = []
    avg_loss_at_write = []
    beta_sum = np.zeros_like(beta)

    def save(i):
        nonlocal num_writes, time_at_write, beta, loss_at_write, samples_seen_at_write, beta_sum
        loss = np.linalg.norm(beta - beta_true)
        ave_loss = loss if i == 0 else np.linalg.norm(beta_sum / i - beta_true)
        log.debug(
            "{:8.0f} sec {:10.8f} loss {:10.8f} ave loss {:10d} samples {:4d}-th iterate",
            time_elapsed,
            loss,
            ave_loss,
            samples_seen,
            num_writes * flags.FLAGS.save_every_n,
        )
        loss_at_write.append(loss)
        time_at_write.append(time_elapsed)
        samples_seen_at_write.append(samples_seen)
        avg_loss_at_write.append(ave_loss)
        num_writes += 1

    if flags.FLAGS.precompute:
        t = time()
        XTX = X.T.dot(X)
        XTy = X.T.dot(y)
        time_elapsed += time() - t

    save(0)

    for i in range(flags.FLAGS.iters):

        t = time()

        if flags.FLAGS.precompute:
            grad = (XTX.dot(beta) - XTy) / n
            samples_seen += n
        else:
            sample = np.random.randint(n)
            grad = X[sample] * (X[sample].dot(beta) - y[sample])
            samples_seen += 1
        beta -= grad * 0.001 * (10 if flags.FLAGS.precompute else 1)

        time_elapsed += time() - t
        beta_sum += beta

        if not (i % flags.FLAGS.save_every_n):
            save(i + 1)
            if np.linalg.norm(grad) / len(grad) < 1e-8:
                log.debug("grad norm very small, stopping early")
                break

    save(i + 1)

    np.save(flags.FLAGS.infile + f"{suffix}/time.npy", np.array(time_at_write))
    np.save(
        flags.FLAGS.infile + f"{suffix}/samples.npy",
        np.array(samples_seen_at_write),
    )
    np.save(flags.FLAGS.infile + f"{suffix}/loss.npy", np.array(loss_at_write))
    np.save(
        flags.FLAGS.infile + f"{suffix}/loss_avg.npy", np.array(loss_at_write)
    )
    flags.FLAGS.append_flags_into_file(
        flags.FLAGS.infile + f"{suffix}/flags.txt"
    )


if __name__ == "__main__":
    app.run(_main)
