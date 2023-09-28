import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import os

torch.cuda.empty_cache()

def MNIST_loaders(train_batch_size=50000, test_batch_size=50000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        # Lambda(lambda x: torch.flatten(x))
        ])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
def save_model(model, optimizer, epoch, best_loss, loss, save_path):
  """Save the model and optimizer state dicts.

  Args:
    model: The model to save.
    optimizer: The optimizer to save.
    epoch: The current epoch.
    best_loss: The best loss achieved so far.
    loss: The loss achieved so far.
    save_path: The path to save the models to.
  """
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  model_state_dict = model.state_dict()
  optimizer_state_dict = optimizer.state_dict()

  # Save the best model.

  if epoch == 0 or best_loss > loss:
    best_loss = loss
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }, os.path.join(save_path, "best.pth"))

  # Save the latest model.

  torch.save({
    "epoch": epoch,
    "model_state_dict": model_state_dict,
    "optimizer_state_dict": optimizer_state_dict,
  }, os.path.join(save_path, "last.pth"))
  return best_loss
