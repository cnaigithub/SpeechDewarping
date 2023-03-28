from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Tacotron2Loss, self).__init__()
        self.reduction = reduction
        
    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.contiguous().view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.contiguous().view(-1, 1)
        mel_loss = nn.MSELoss(reduction=self.reduction)(mel_out, mel_target) + \
            nn.MSELoss(reduction=self.reduction)(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)(gate_out, gate_target)
        return mel_loss, gate_loss