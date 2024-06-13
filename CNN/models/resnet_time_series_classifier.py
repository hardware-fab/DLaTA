"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import torch.nn as nn

from .resnet import ResNet


class ResNetTimeSeriesClassifier(nn.Module):
    def __init__(self, classifier_params, encoder_params, dropout=0.):
        super(ResNetTimeSeriesClassifier, self).__init__()

        self.encoder = ResNet(**encoder_params)

        self.classifier = nn.Linear(self.encoder.encoding_size,
                                    classifier_params['out_channels'])

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        x = self.encoder(x)
        x = self.classifier(x)
        return x
