import torch
import torch.nn as nn


mapping = {}


def put_on_device(device):
    def _thunk(cls):
        def add_device(*words, **keywords):
            object_ = cls(*words, **keywords)
            object_.to(device)
            return object_
        return add_device
    return _thunk


def register(name):
    def _thunk(cls):
        mapping[name] = cls
        return cls
    return _thunk


@register("impala_cnn")
class ImpalaCNN(nn.Module):
    def __init__(self, depths=None):
        super().__init__()
        if depths is None:
            depths = [16, 32, 32, 32, 32]
        cnn_modules = []
        for i in range(len(depths) - 1):
            conv_sequence = nn.Sequential(
                nn.Conv2d(depths[i], depths[i+1], 3, stride=1),
                nn.MaxPool2d(3, stride=2),
                ResidualBlock(depths[i+1]),
                ResidualBlock(depths[i+1])
            )
            self.__setattr__("conv_sequence_{}".format(i), conv_sequence)
            cnn_modules.append(conv_sequence)
        self.cnn_modules = nn.ModuleList(cnn_modules)

    def forward(self, input_):
        output = input_.float() / 255.
        for cnn_module in self.cnn_modules:
            output = cnn_module(output)
        output = torch.relu(output)
        output = output.flatten(1)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, depth):
        super(ResidualBlock, self).__init__()
        self.bypass = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input_):
        return input_ + self.bypass(input_)


def get_network_builder(name):
    """copy from google-research-football"""
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))


if __name__ == "__main__":
    cnn = ImpalaCNN()
    image = torch.rand((32, 16, 72, 96))
    out = cnn(image)
    print(out.shape)
