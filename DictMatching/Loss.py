import torch
from torch.nn import CrossEntropyLoss


class NTXentLoss(torch.nn.Module):
    """Implementation of the Contrastive Cross Entropy Loss.
    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like
    the one described in the MoCo[1] paper.
    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    [1] MoCo, 2020, https://arxiv.org/abs/1911.05722

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank.
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.
    Raises:
        ValueError if abs(temperature) < 1e-8 to prevent divide by zero.
    Examples:
        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        """Forward pass through Contrastive Cross-Entropy Loss.
        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.
            Args:
                out0:
                    Output projections of the first set of transformed images.
                    Shape: (batch_size, embedding_size)
                out1:
                    Output projections of the second set of transformed images.
                    Shape: (batch_size, embedding_size)
            Returns:
                Contrastive Cross Entropy Loss value.
        """

        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        # print(out0.shape)
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)

        # use other samples from batch as negatives
        output = torch.cat((out0, out1), axis=0)

        # the logits are the similarity matrix divided by the temperature
        logits = torch.einsum('nc,mc->nm', output, output) / self.temperature
        # We need to removed the similarities of samples to themselves
        logits = logits[~torch.eye(2 * batch_size, dtype=torch.bool, device=out0.device)].view(2 * batch_size, -1)

        # The labels point from a sample in out_i to its equivalent in out_(1-i)
        labels = torch.arange(batch_size, device=device, dtype=torch.long)

        # -1 因为eye没了
        labels = torch.cat([labels + batch_size - 1, labels])
        loss = self.cross_entropy(logits, labels)

        acc = ((logits.clone().detach().argmax(dim=-1) == labels).sum() / (batch_size * 2)).cpu().item()
        return loss, acc


class MoCoLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lossFunc = CrossEntropyLoss()

    def forward(self, logits, labels):
        batch_size, _ = logits.shape
        loss = self.lossFunc(logits, labels)
        acc = ((logits.clone().detach().argmax(dim=-1) == labels).sum() / batch_size).cpu().item()
        return loss, acc
