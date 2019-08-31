"""
Given a file --infile pointing to a npz archive with X and y vectors,
this kicks off an iterative process to find an estimate for the
linear regression parameters.

Intermediate steps are saved to a directory in the same location
as --infile, suffixed with '-trace'. I.e., if --infile is

./data/input.npz

Then the saved output vectors are:

./data/input.npz-trace/0.npy
./data/input.npz-trace/1.npy
...
etc.

as well as

./data/input.npz-trace/time.npy
./data/input.npz-trace/samples.npy


where each <digit>.npy file is a flat array of the parameters at
a given time along the optimization process

and time.npy is a flat array of the
wall clock time in seconds that it took to reach that step and
samples.npy is a flat array of the number of uniform random samples
seen at that step.

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

    def save():
        nonlocal num_writes, time_at_write, beta, loss_at_write, samples_seen_at_write
        loss = np.linalg.norm(beta - beta_true)
        log.debug(
            "{:8.0f} sec {:10.8f} loss {:10d} samples {:4d}-th iterate",
            time_elapsed,
            loss,
            samples_seen,
            num_writes,
        )
        width = str(len(str(flags.FLAGS.iters)))
        np.save(
            flags.FLAGS.infile
            + ("{}/{:0" + width + "d}.npy").format(suffix, num_writes),
            beta,
        )
        loss_at_write.append(loss)
        time_at_write.append(time_elapsed)
        samples_seen_at_write.append(samples_seen)
        num_writes += 1

    if flags.FLAGS.precompute:
        t = time()
        XTX = X.T.dot(X)
        XTy = X.T.dot(y)
        time_elapsed += time() - t

    save()

    for i in range(flags.FLAGS.iters):

        t = time()

        if flags.FLAGS.precompute:
            grad = (XTX.dot(beta) - XTy) / n
            samples_seen += n
        else:
            sample = np.random.randint(n)
            grad = X[sample] * (X[sample].dot(beta) - y[sample])
            samples_seen += 1
        beta -= grad * 0.01

        time_elapsed += time() - t

        if not (i % flags.FLAGS.save_every_n):
            save()
            if np.linalg.norm(grad) / len(grad) < 1e-8:
                log.debug("grad norm very small, stopping early")
                break

    save()

    np.save(flags.FLAGS.infile + f"{suffix}/time.npy", np.array(time_at_write))
    np.save(
        flags.FLAGS.infile + f"{suffix}/samples.npy",
        np.array(samples_seen_at_write),
    )
    np.save(flags.FLAGS.infile + f"{suffix}/loss.npy", np.array(loss_at_write))
    flags.FLAGS.append_flags_into_file(
        flags.FLAGS.infile + f"{suffix}/flags.txt"
    )


if __name__ == "__main__":
    app.run(_main)
