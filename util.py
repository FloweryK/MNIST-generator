import os
import matplotlib.pyplot as plt


def savefig(epoch, samples):
    fig, ax = plt.subplots(1, 10, figsize=(10, 1))
    for r in range(10):
        ax[r].set_axis_off()
        ax[r].imshow(samples[r].data.cpu().numpy().reshape(28, 28))
    if not os.path.exists("samples/"):
        os.mkdir("samples/")
    plt.savefig("samples/{}.png".format(str(epoch).zfill(3)), bbox_inches="tight")
    plt.close(fig)
