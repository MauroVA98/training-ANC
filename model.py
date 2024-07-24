import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset

Model = namedtuple('Model', 'Phi H cfg')


class Phi(nn.Module):
    def __init__(self, config: dict):
        super(Phi, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(self.config['ni'], self.config['Phi']['features'][0])
        for i, feature in enumerate(self.config['Phi']['features']):
            if i != len(self.config['Phi']['features'])-1:
                setattr(self, f'fc{i + 2}', nn.Linear(feature, self.config['Phi']['features'][i + 1]))
            else:
                setattr(self, f'fc{i + 2}', nn.Linear(feature, self.config['not'] - 1))

    def forward(self, x):
        for i, activation in enumerate(self.config['Phi']['activations']):
            if activation == 'relu':
                fc = getattr(self, f'fc{i + 1}')
                x = F.relu(fc(x))
        fc = getattr(self, f'fc{i + 2}')
        x = fc(x)
        if len(x.shape) == 1:
            # single input
            return torch.cat([x, torch.ones(1)])
        else:
            # batch input for train
            return torch.cat([x, torch.ones([x.shape[0], 1])], dim=-1)


# Cross-Entropy Loss
class H(nn.Module):
    def __init__(self, config: dict):
        super(H, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(self.config['not'], self.config['H']['features'][0])
        for i, feature in enumerate(self.config['H']['features']):
            if i != len(self.config['H']['features']) - 1:
                setattr(self, f'fc{i + 2}', nn.Linear(feature, self.config['H']['features'][i + 1]))
            else:
                setattr(self, f'fc{i + 2}', nn.Linear(feature, self.config['nc']))

    def forward(self, x):
        for i, activation in enumerate(self.config['H']['activations']):
            if activation == 'relu':
                fc = getattr(self, f'fc{i + 1}')
                x = F.relu(fc(x))
        fc = getattr(self, f'fc{i + 2}')
        x = fc(x)
        return x


class Dataset(Dataset):
    def __init__(self, inputs, outputs, c):
        self.inputs = inputs
        self.outputs = outputs
        self.c = c

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx,]
        output = self.outputs[idx,]
        return {'input': input, 'output': output, 'c': self.c}


def save2torch(*, Phi: nn.Module, H: nn.Module, name: str, cfg: dict):
    path = os.path.join(os.getcwd(), 'models', cfg["save_path"], cfg["model_name"])
    if not os.path.isdir(path):
        os.makedirs(path)
    if H is not None:
        torch.save({
            'Phi': Phi.state_dict(),
            'H': H.state_dict(),
            'cfg': cfg,
        }, os.path.join(path, name + '.pth'))
    else:
        torch.save({
            'Phi': Phi.state_dict(),
            'H': None,
            'cfg': cfg,
        }, os.path.join(path, name + '.pth'))


def save2txt(model: nn.Module, name: str, cfg: dict):
    path = os.path.join(os.getcwd(), 'models', cfg["save_path"], cfg["model_name"], name + ".txt")
    with open(path, "w") as f:
        for key in model.state_dict().keys():
            if "weight" in key:
                np.savetxt(f, model.state_dict()[key].detach().numpy(), delimiter=",", fmt="%.7e")
                f.write('\n')
            if "bias" in key:
                np.savetxt(f, model.state_dict()[key].detach().numpy()[None], delimiter=",", fmt="%.7e")
                f.write('\n')


def load_model(path: str):
    model = torch.load(path)
    cfg = model['cfg']

    phi = Phi(config=cfg)
    h = H(config=cfg)

    phi.load_state_dict(model['Phi'])
    h.load_state_dict(model['H'])

    phi.eval()
    h.eval()
    return Model(phi, h, cfg)