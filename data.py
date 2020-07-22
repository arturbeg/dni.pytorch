import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

def get_mnist(root, n_labeled, bsz, download=True):
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            MNIST_MEAN, MNIST_STD)
    ])

    base_dataset = torchvision.datasets.MNIST(root, train=True, download=download, transform=transform)
    # a bit redundant TODO: make sure both datasets are identical
    base_unlabelled_dataset = torchvision.datasets.MNIST(root, train=True, download=download, transform=transform)
    base_unlabelled_dataset.train_labels = base_unlabelled_dataset.train_labels.new_full(
        base_unlabelled_dataset.train_labels.size(), -1)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.train_labels,
                                                                         int(n_labeled / 10))
    train_labeled = Subset(base_dataset, train_labeled_idxs)
    train_unlabeled = Subset(base_unlabelled_dataset, train_unlabeled_idxs)

    val = Subset(base_dataset, val_idxs)
    test_dataset = torchvision.datasets.MNIST(root, train=False, download=download, transform=transform)

    train_labeled_loader = DataLoader(train_labeled, batch_size=bsz, shuffle=True, drop_last=True)
    train_unlabeled_loader = DataLoader(train_unlabeled, batch_size=bsz, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=bsz, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)

    # print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")

    return train_labeled_loader, train_unlabeled_loader, val_loader, test_loader


def get_mnist_np(root, download=True):
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            MNIST_MEAN, MNIST_STD)
    ])

    train_dataset = torchvision.datasets.MNIST(root, train=True, download=download, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root, train=False, download=download, transform=transform)

    train_data_np = train_dataset.train_data.numpy()
    train_labels_np = train_dataset.train_labels.numpy()

    test_data_np = test_dataset.test_data.numpy()
    test_labels_np = test_dataset.test_labels.numpy()

    return train_data_np, train_labels_np, test_data_np, test_labels_np

def shuffle(x, y):
    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    return x, y

def preprocess(train_data_np, train_labels_np, test_data_np, test_labels_np, proportion_labeled=0.1, is_classification=True):
    # produces the same split every time (deterministic)

    split_at = int(len(train_data_np) * proportion_labeled)
    x_labeled, x_unlabelled = train_data_np[:split_at], train_data_np[split_at:]
    y_labeled, y_unlabelled = train_labels_np[:split_at], train_labels_np[split_at:]

    x_labeled, x_unlabelled, y_labeled, y_unlabelled = torch.from_numpy(x_labeled), \
                                                       torch.from_numpy(x_unlabelled), \
                                                       torch.from_numpy(y_labeled), torch.from_numpy(y_unlabelled)
    x_test, y_test = torch.from_numpy(test_data_np), torch.from_numpy(test_labels_np)
    w = torch.zeros(size=(y_unlabelled.shape[0], 10), requires_grad=True) # 10 refers to the number of classes (FROM LGA PAPER)

    if is_classification:
        x_labeled, x_unlabelled, x_test = x_labeled.view(-1, 28 * 28), x_unlabelled.view(-1, 28 * 28), x_test.view(-1, 28 * 28)

    return x_labeled.float(), x_unlabelled.float(), x_test.float(), y_labeled.float(), w, y_unlabelled.float(), y_test.float()


# TODO: turn two functions below into one
def sample_minibatch(x, y, w, batch_size):
    # y is w if dealing with the unlabelled dataset
    indices = torch.randperm(len(x))[:batch_size]

    if w is None:
        return x[indices], y[indices], None, indices
    else:
        return x[indices], y[indices], w[indices], indices

def sample_minibatch_deterministically(x, y, batch_i, batch_size):
    indicies = torch.arange((batch_i * batch_size), (batch_i + 1) * batch_size, 1)
    batch_x = x[indicies]
    batch_y = y[indicies]
    return batch_x, batch_y, indicies