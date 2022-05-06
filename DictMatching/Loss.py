import torch
from torch.nn import CrossEntropyLoss

class MoCoLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lossFunc = CrossEntropyLoss()

    def forward(self, logits, labels):
        batch_size, _ = logits.shape
        loss = self.lossFunc(logits, labels)
        acc = ((logits.clone().detach().argmax(dim=-1) == labels).sum() / batch_size).cpu().item()
        return loss, acc
