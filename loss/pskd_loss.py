import torch
import torch.nn as nn
from torch.nn import functional as F


class Custom_CrossEntropy_PSKD(nn.Module):
	def __init__(self, local_rank):
		super().__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda(local_rank)

	def forward(self, output, targets):
		"""
		Args:
			output: logit
			targets: softmax 된 값. (정확히는 이전 에폭 모델의 output에 소프트맥스 한 것과 하드레이블의 조합)
		"""
		log_probs = self.logsoftmax(output)
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class Custom_CrossEntropy_PSKD_Temperature(nn.Module):
    def __init__(self, local_rank, T):
        print("temperature loss\n\n")
        super().__init__()
        self.T = T
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda(local_rank)
        self.pskd = Custom_CrossEntropy_PSKD(local_rank)
    def forward(self, output, targets):
        loss = self.pskd(output, targets)
        output = F.softmax(output, dim=1)
        output = output.log()
        log_probs = self.logsoftmax(output/self.T)
        target_logit = targets.log()
        targets = F.softmax(target_logit/self.T, dim=1)
        loss = (-targets * log_probs).mean(0).sum() * (self.T**2)
        return loss
class Custom_CrossEntropy_PSKD_version9(nn.Module):
    def __init__(self, local_rank, alpha):
        super().__init__()
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda(local_rank)
        self.pskd = Custom_CrossEntropy_PSKD(local_rank)

    def forward(self, output, targets):
        '''
        :param output: logit
        :param targets: softmax 된 값. (정확히는 이전 에폭 모델의 output에 소프트맥스 한 것과 하드레이블의 조합)
        :return:
        '''
        loss = self.pskd(output, targets)
        argsorted = torch.argsort(targets, dim=1, descending=True)
        for idx, i in enumerate(range(0, 90+1, 5)):
            sub_out = torch.gather(output, 1, argsorted[:,i : i + 10])
            sub_target = F.softmax(torch.gather(targets, 1, argsorted[:,i : i + 10]))
            loss += self.pskd(sub_out, sub_target) * self.alpha
        return loss
class Custom_CrossEntropy_PSKD_version10(nn.Module):
    def __init__(self, local_rank, alpha):
        super().__init__()
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda(local_rank)
        self.pskd = Custom_CrossEntropy_PSKD(local_rank)

    def forward(self, output, targets):
        '''
        :param output: logit
        :param targets: softmax 된 값. (정확히는 이전 에폭 모델의 output에 소프트맥스 한 것과 하드레이블의 조합)
        :return:
        '''
        loss = self.pskd(output, targets)
        argsorted = torch.argsort(targets, dim=1, descending=True)
        for idx, i in enumerate(range(0, 30+1, 5)):
            sub_out = torch.gather(output, 1, argsorted[:,i : i + 10])
            sub_target = F.softmax(torch.gather(targets, 1, argsorted[:,i : i + 10]))
            loss += self.pskd(sub_out, sub_target) * self.alpha
        return loss
class Custom_CrossEntropy_PSKD_version9_imagenet(nn.Module):
    def __init__(self, local_rank, alpha):
        super().__init__()
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda(local_rank)
        self.pskd = Custom_CrossEntropy_PSKD(local_rank)

    def forward(self, output, targets):
        '''
        :param output: logit
        :param targets: softmax 된 값. (정확히는 이전 에폭 모델의 output에 소프트맥스 한 것과 하드레이블의 조합)
        :return:
        '''
        loss = self.pskd(output, targets)
        argsorted = torch.argsort(targets, dim=1, descending=True)
        for idx, i in enumerate(range(0, 90+1, 10)):
            sub_out = torch.gather(output, 1, argsorted[:,i : i + 10])
            sub_target = F.softmax(torch.gather(targets, 1, argsorted[:,i : i + 10]))
            loss += self.pskd(sub_out, sub_target) * self.alpha
        return loss
class Custom_CrossEntropy_PSKD_version13(nn.Module):
    def __init__(self, local_rank, alpha):
        super().__init__()
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda(local_rank)
        self.pskd = Custom_CrossEntropy_PSKD(local_rank)

    def forward(self, output, targets):
        '''
        :param output: logit
        :param targets: softmax 된 값. (정확히는 이전 에폭 모델의 output에 소프트맥스 한 것과 하드레이블의 조합)
        :return:
        '''
        loss = self.pskd(output, targets)
        argsorted = torch.argsort(targets, dim=1, descending=True)
        argsorted2 = torch.argsort(output, dim=1, descending=True)
        for idx, i in enumerate(range(0, 90+1, 5)):
            sub_out = torch.gather(output, 1, argsorted[:,i : i + 10])
            sub_target = F.softmax(torch.gather(targets, 1, argsorted[:,i : i + 10]))
            loss += self.pskd(sub_out, sub_target) * self.alpha
        return loss
class Custom_CrossEntropy_PSKD_version14(nn.Module):
    def __init__(self, alpha, local_rank):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.pskd = Custom_CrossEntropy_PSKD(local_rank)
        self.alpha = alpha

    def forward(self, output, soft_targets, added_soft_target, targets_onehot, alpha_T):
        """
        Args:
            output: Model output (before softmax)
            soft_targets: t-1 model output (before softmax)
            added_soft_target: same as pskd (after softmax)
            targets_onehot, : one hot encoded hard target
            alpha_T: same as pskd
            epoch: (ignored)
        """
        loss = self.pskd(output, added_soft_target)
        argsorted = torch.argsort(soft_targets, dim=1, descending=True)

        for idx, i in enumerate(range(0, 90+1, 5)):
            sub_out = torch.gather(output, 1, argsorted[:,i : i + 10])
            sub_target = F.softmax(torch.gather(soft_targets, 1, argsorted[:,i : i + 10]))
            sub_hard_target = torch.gather(targets_onehot, 1, argsorted[:, i : i + 10])
            indice = torch.sum(sub_hard_target, dim=1)
            not_exist_target = torch.logical_not(indice) == True
            exist_target = torch.logical_not(not_exist_target) == True
            sub_target[exist_target, :] = (1 - alpha_T) * sub_hard_target[exist_target,: ] + alpha_T * sub_target[exist_target, :]
            loss += self.pskd(sub_out, sub_target) * self.alpha
        return loss
class Custom_CrossEntropy_PSKD_version15(nn.Module):  # output에 softmax 걸고 argsort
    def __init__(self, local_rank, alpha):
        super().__init__()
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda(local_rank)
        self.pskd = Custom_CrossEntropy_PSKD(local_rank)

    def forward(self, output, targets):
        '''
        :param output: logit
        :param targets: softmax 된 값. (정확히는 이전 에폭 모델의 output에 소프트맥스 한 것과 하드레이블의 조합)
        :return:
        '''
        loss = self.pskd(output, targets)
        argsorted = torch.argsort(targets, dim=1, descending=True)
        output = F.softmax(output, dim=1)
        for idx, i in enumerate(range(0, 90+1, 5)):
            sub_out = torch.gather(output, 1, argsorted[:,i : i + 10])
            sub_target = F.softmax(torch.gather(targets, 1, argsorted[:,i : i + 10]))
            loss += self.pskd(sub_out, sub_target) * self.alpha
        return loss

if __name__ == "__main__":
    asd = Custom_CrossEntropy_PSKD(0)