import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import Dataset


def get_num_components(tag: str) -> int:
    if tag == "2-clusters":
        return 2
    elif tag == "3-clusters":
        return 3
    elif tag == "9-clusters":
        return 9
    elif tag == "2-moons":
        return 50
    elif tag == "circles":
        return 50
    elif tag == "aniso":
        return 30
    elif tag == "varied":
        return 3


@torch.no_grad()
def generate_data(tag: str, device: torch.device, n_samples: int = 1000):
    if tag == "2-clusters":
        centers = [[0.0, 0.0], [0.5, 0.5]]
        cluster_stds = 0.1
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )

    elif tag == "3-clusters":
        centers = [[0.0, 0.0], [0.5, 0.5], [0.5, 0.0]]
        cluster_stds = 0.05
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )
    elif tag == "9-clusters":
        centers = [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 1.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        cluster_stds = 0.1
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )
    elif tag == "2-moons":
        data, y = datasets.make_moons(n_samples=n_samples, noise=0.1, random_state=0)

    elif tag == "circles":
        data, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

    elif tag == "aniso":
        # Anisotropicly distributed data
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            cluster_std=0.2,
            random_state=0,
            centers=[[-1, -1], [-1, 0.5], [0.5, 0.5]],
        )
        transformation = [[0.5, -0.2], [-0.2, 0.4]]
        X_aniso = np.dot(X, transformation)
        data = X_aniso

    elif tag == "varied":
        # blobs with varied variances
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            cluster_std=[0.5, 0.1, 0.3],
            random_state=0,
            center_box=[-2, 2],
        )

    data_x = data
    data_x = torch.from_numpy(data_x).float().to(device)
    return data_x


# TEXTWIDTH=3.33740 # Double-Column ACM Paper

import matplotlib

matplotlib.rcParams["axes.unicode_minus"] = False


class SynthDataset(Dataset):
    def __init__(self, tag: str, num_samples: int):
        self.tag = tag
        self.num_samples = num_samples
        self.data = generate_data(
            tag, n_samples=num_samples, device=torch.device("cpu")
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        return sample