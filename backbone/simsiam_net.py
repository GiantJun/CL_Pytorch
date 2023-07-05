import  torch.nn as nn
import torch.nn.functional as F
import torch

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x


class SimSiamNet(nn.Module):
    def __init__(self, network_type, network, projection_dim=2048, prediction_dim=512):
        super().__init__()
        if 'resnet' in network_type:
            self.feature_dim = network.fc.in_features
            network.fc = ProjectionMLP(self.feature_dim, projection_dim, projection_dim)
        else:
            raise ValueError('{} did not support yet!'.format(network_type))
        self.encoder = network
        self.prediction = PredictionMLP(projection_dim, prediction_dim, projection_dim)

    def forward(self, x1, x2):
        z1=self.encoder(x1)
        p1=self.prediction(z1)

        z2=self.encoder(x2)
        p2=self.prediction(z2)

        return -(F.cosine_similarity(p1, z2.detach()).mean() + F.cosine_similarity(p2, z1.detach()).mean()) * 0.5
