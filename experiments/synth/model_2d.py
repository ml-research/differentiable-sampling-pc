import torch
from torch import nn
from torch.nn import functional as F

from experiments.synth.utils_2d import get_num_components


def project(value, vmin, vmax):
    return torch.sigmoid(value) * (vmax - vmin) + vmin


class FlatSpn2D(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_components = get_num_components(args.data_synth)
        self.weights = nn.Parameter(torch.rand(n_components))
        self.n_components = n_components

        # Constrain the parameter to the valid ranges
        self._loc = nn.Parameter(torch.randn(2, n_components))
        self._scale = nn.Parameter(torch.randn(2, n_components))
        self.min_scale = 0.05
        self.max_scale = 0.5
        self.min_loc = -3
        self.max_loc = 3
        self.args = args

        if args.spn_learn_tau:
            self.tau = nn.Parameter(torch.tensor(0.0))

    @property
    def scale(self):
        return project(self._scale, self.min_scale, self.max_scale)

    @property
    def loc(self):
        return project(self._loc, self.min_loc, self.max_loc)

    def sample(self, n_samples):
        return self.sample_diff(n_samples, tau=1.0, hard=True)

    def sample_diff(self, n_samples, tau=None, hard=None):

        if tau is None:
            tau = self.args.spn_tau
        if hard is None:
            hard = self.args.spn_hard

        if self.args.spn_learn_tau:
            tau = torch.sigmoid(self.tau)

        # Leaf layer
        eps = torch.normal(0.0, 1.0, (n_samples,2, self.n_components), device=self.loc.device)
        loc = self.loc.unsqueeze(0)
        scale = self.scale.unsqueeze(0)
        samples = loc + scale * eps

        # Sum layer
        logw = torch.log_softmax(self.weights, dim=0)
        logw = logw.squeeze(0).expand(n_samples, -1)
        y = F.gumbel_softmax(logits=logw, tau=tau, hard=hard)

        # Interpolate child samples
        x = torch.sum(y.view(n_samples, 1, self.n_components) * samples, dim=2)
        assert x.shape == (n_samples, 2)
        return x

    def forward(self, data: torch.Tensor):
        return self.log_prob(data)

    def log_prob(self, data: torch.Tensor):
        assert data.size(1) == 2

        # Leaf layer
        loc = self.loc.unsqueeze(0)
        scale = self.scale.unsqueeze(0)
        data = data.unsqueeze(-1)
        lls = torch.distributions.Normal(loc=loc, scale=scale).log_prob(data)
        assert lls.shape == (data.shape[0], 2, self.n_components)

        # Product layer
        lls = lls.sum(dim=1)
        assert lls.shape == (data.shape[0], self.n_components)

        # Sum layer
        logweights = torch.log_softmax(self.weights, dim=0)
        lls = torch.logsumexp(lls + logweights, dim=1)

        return lls