

import argparse
import numpy as np


def gen(input_path):
    logits = np.load(input_path)

    # you should theoretically softmax these logits for correctness
    # but since we're just taking the argmax it really doesn't matter

    preds = np.argmax(logits, axis=1)
    indices = np.arange(len(preds))
    out = np.column_stack((indices, preds))
    np.savetxt("predictions.csv", out, fmt="%d", delimiter=",", header="Id,Category", comments="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")

    args = parser.parse_args()

    gen(args.input_path)


if __name__ == "__main__":
    main()
