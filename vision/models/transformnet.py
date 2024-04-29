import torch

from collections import OrderedDict

class TransformNet(torch.nn.Module):
    def __init__(self, dnn, dnn_name, dataset_name):
        super(TransformNet, self).__init__()

        self.dnn_name = dnn_name
        self.dataset_name = dataset_name
        ft, classifier = self._split(dnn)

        self.dnn = torch.nn.Sequential(OrderedDict([
            ('features', torch.nn.Sequential(*ft)),
            ('classifier', torch.nn.Sequential(classifier))
        ]))

    def forward(self, x):
        z = self.dnn(x)
        return z

    def _split(self, dnn):
        layer_cake = list(dnn.children())
        last_layer = torch.nn.Sequential(*layer_cake[-1:])
        head_model = torch.nn.Sequential(*layer_cake[:-1])

        classifier = last_layer
        ft = [head_model, torch.nn.Flatten(1)]
        return ft, classifier