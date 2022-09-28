import torch
import torch.nn as nn
import torch.nn.functional as F
from .pskd_loss import Custom_CrossEntropy_PSKD
class TFKD_regularization(nn.Module):
    def __init__(self, model, local_rank):
        super().__init__()
        self.local_rank = local_rank
        if model == "resnet18": self.alpha, self.T, self.multiplier = 0.1, 20, 100
        elif model == "resnext": self.alpha, self.T, self.multiplier = 0.5, 40, 1
        elif model == "densenet121": self.alpha, self.T, self.multiplier = 0.1, 40, 1
        else: self.alpha, self.T, self.multiplier = 0.1, 40, 1
        self.correct_prob = 0.99
        self.criterion_CE = nn.CrossEntropyLoss()
        self.KLDiv = nn.KLDivLoss()
    def forward(self, output, label):
        '''

        :param output: logit
        :param label: 클래스 나타내는 숫자. 1,2,3,4,5,6...
        :return:
        '''
        K = output.size(1)
        teacher_soft = torch.ones_like(output).cuda(self.local_rank)
        teacher_soft = teacher_soft * (1 - self.correct_prob) / (K-1)
        for i in range(output.shape[0]):
            teacher_soft[i, label[i]] = self.correct_prob
        loss_CE = self.criterion_CE(output, label)
        loss_soft_reg = self.KLDiv(F.log_softmax(output, dim=1), F.softmax(teacher_soft/self.T, dim=1)) * self.multiplier
        loss_KD = (1. - self.alpha) * loss_CE + self.alpha * loss_soft_reg
        return loss_KD

class TFKD_regularization_version9(nn.Module):
    def __init__(self, model, local_rank, alpha):
        super().__init__()
        self.local_rank = local_rank
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda(self.local_rank)
        self.tfkd = TFKD_regularization(model, self.local_rank)
        self.kldiv = Custom_CrossEntropy_PSKD(self.local_rank)
        self.correct_prob = 0.99
    def make_soft_target(self, output):
        K = output.size(1)
        soft_targets = torch.ones_like(output).cuda(self.local_rank)
        soft_targets *= (1- self.correct_prob) / (K-1)
        return soft_targets
    def forward(self, output, label):
        '''
        
        :param output:  logit
        :param label: 클래스 나타내는 숫자. 1,2,3,4,5,6...
        :return: 
        '''
        soft_label = self.make_soft_target(output)  # LS된 soft one hot
        loss = self.tfkd(output, label)
        argsorted = torch.argsort(output, dim=1, descending=True)
        for idx, i in enumerate(range(0, 90 + 1, 5)):
            sub_out = torch.gather(output, 1, argsorted[:, i: i + 10])
            sub_target = torch.gather(soft_label, 1, argsorted[:, i: i + 10])
            loss += self.kldiv(sub_out, sub_target) * self.alpha
        return loss

if __name__ == "__main__":
    criterion_KD = TFKD_regularization_version9("resnet18", 0)
    criterion_KD2 = TFKD_regularization("resnet18", 0)
    output = torch.randn((64, 100)).cuda()
    label = torch.arange(64).cuda()
    loss = criterion_KD(output, label)
    loss2 = criterion_KD2(output, label)
    print(loss, loss2)
