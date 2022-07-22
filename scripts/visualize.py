import matplotlib.pyplot as plt
import numpy as np
from typing import List, Text, Tuple


def line2matrix(line: Text, n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    converts alignemnt given in the format "0-1 3p4 5-6" to alignment matrices
    n, m: maximum length of the involved sentences (i.e., dimensions of the alignemnt matrices)
    '''
    def convert(i, j):
        i, j = int(i), int(j)
        if i >= n or j >= m:
            raise ValueError("Error in Gold Standard?")
        return i, j
    possibles = np.zeros((n, m))
    sures = np.zeros((n, m))
    for elem in line.split(" "):
        if "p" in elem:
            i, j = convert(*elem.split("p"))
            possibles[i, j] = 1
        elif "-" in elem:
            i, j = convert(*elem.split("-"))
            possibles[i, j] = 1
            sures[i, j] = 1
    return sures, possibles


def plot_alignments(e: List[Text],
                    f: List[Text],
                    sures: np.ndarray,
                    possibles: np.ndarray,
                    alignment1: np.ndarray,
                    alignment2: np.ndarray = None,
                    title: Text = None,
                    filename: Text = None,
                    dpi: int = 150):
    shorter = min(len(e), len(f))
    scalefactor = min((4 / shorter), 1)

    groundtruth = 0.75 * sures + 0.4 * possibles

    fig, ax = plt.subplots()
    im = ax.imshow(groundtruth, cmap="Greens", vmin=0, vmax=1.5)

    # show all ticks...
    ax.set_xticks(np.arange(len(f)))
    ax.set_yticks(np.arange(len(e)))
    # ... and label them
    ax.set_xticklabels(f, fontsize=25 * scalefactor)
    ax.set_yticklabels(e, fontsize=25 * scalefactor)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="left",
             rotation_mode="default")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")
    ax.set_xticks(np.arange(groundtruth.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(groundtruth.shape[0] + 1) - .5, minor=True)

    # set grid
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Loop over data dimensions and create text annotations.
    circle = dict(boxstyle="circle,pad=0.3", fc=(0, 0, 0, 0.0), ec="black", lw=3)
    roundthing = dict(boxstyle="square,pad=0.3", fc="black", ec=(0, 0, 0, 0.0), lw=2)

    # plot alignments
    for i in range(len(e)):
        for j in range(len(f)):
            if alignment1[i, j] > 0:
                t = ax.text(j, i, "x", ha="center", va="center",
                            size=25 * scalefactor,
                            bbox=circle, color=(0, 0, 0, 0.0))
            if alignment2 is not None and alignment2[i, j] > 0:
                t = ax.text(j, i, "x", ha="center", va="center",
                            size=12 * scalefactor,
                            bbox=roundthing, color=(0, 0, 0, 0.0))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if filename:
        plt.savefig(filename, dpi=dpi)
    else:
        plt.show()


if __name__ == '__main__':
    line2matrix("0-0 1p1 2-1", 3, 2)
    plot_alignments(["Testing", "this", "."],
                    ["Hier", "wird", "getestet", "."],
                    np.array([[0, 0, 1, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 1]]),
                    np.array([[0, 0, 0, 0],
                              [1, 0, 0, 0],
                              [0, 0, 0, 0]]),
                    np.array([[0, 1, 0, 0],
                              [0, 0, 0, 0],
                              [0, 1, 0, 0]]),
                    np.array([[0, 0, 0, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]),
                    "Example")
