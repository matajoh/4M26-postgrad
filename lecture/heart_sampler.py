import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def sample_from_heart():
    heart = np.array(Image.open("heart.png"))
    heart = heart[..., 0]

    samples = np.random.random((10000, 2))

    sample_pixels = (samples * 256).astype(np.int64)
    valid = heart[sample_pixels[:, 0], sample_pixels[:, 1]] == 0
    invalid = ~valid

    plt.figure(figsize=(8, 8))
    plt.scatter(samples[valid, 1], 1-samples[valid, 0], s=3, c="pink")
    plt.scatter(samples[invalid, 1], 1-samples[invalid, 0], s=3, c="black")
    x = y = 0.5
    xytext = (0, 40)
    plt.annotate("P(you, me)=1", xy=(x, y),
                 xytext=xytext, textcoords='offset points',
                 fontsize=24, ha="center",
                 arrowprops={'arrowstyle': '-|>', 'color': 'black'})
    plt.tight_layout()
    plt.savefig("heart_samples.png")


if __name__ == "__main__":
    sample_from_heart()
