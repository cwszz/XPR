import torch.nn as nn
import torch
from transformers import XLMRobertaModel

from DictMatching.model import CrossLingualRobertaForSequenceClassification


def _get_simclr_projection_head(num_ftrs: int, out_dim: int):
    """Returns a 2-layer projection head.

    Reference (07.12.2020):
    https://github.com/google-research/simclr/blob/master/model_util.py#L141

    """
    modules = [
        nn.Linear(num_ftrs, num_ftrs*4),
        # nn.BatchNorm1d(num_ftrs),
        nn.ReLU(),
        nn.Linear(num_ftrs*4, out_dim)
    ]
    return nn.Sequential(*modules)


class SimCLR(nn.Module):
    def __init__(self, config):
        super(SimCLR, self).__init__()
        self.model = XLMRobertaModel.from_pretrained("xlm-roberta-base", config=config, add_pooling_layer=False)
        self.projection_head = _get_simclr_projection_head(768, 768)

    def forward(self,
                x0,
                x1
                ):
        # forward pass of first input x0
        f0 = self.model(**x0).last_hidden_state
        f0 = f0[:, 0, :]
        f0 = self.projection_head(f0)

        f1 = self.model(**x1).last_hidden_state
        f1 = f1[:, 0, :]
        f1 = self.projection_head(f1)

        return f0, f1
