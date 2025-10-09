#%% Imports
import numpy as np
import matplotlib.pyplot as plt



#%% Definitions
def plot_losses(train_losses, val_losses, save_path=None,  title="Loss vs. Epoch"):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Val")

    # plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs. Loss")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
