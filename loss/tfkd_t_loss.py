import torch
import torch.nn as nn
import torch.nn.functional as F
from .pskd_loss import Custom_CrossEntropy_PSKD
class Custom_CrossEntropy_PSKD(nn.Module):
	def __init__(self):
		super().__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, output, targets):
		"""
		Args:
			output: logit
			targets: softmax 된 값. (정확히는 이전 에폭 모델의 output에 소프트맥스 한 것과 하드레이블의 조합)
		"""
		log_probs = self.logsoftmax(output)
		loss = (- targets * log_probs).mean(0).sum()
		return loss
class TFKD_t_regularization(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model == "resnet18": self.alpha, self.T, self.multiplier = 0.95, 6, 50
        elif model == "resnext": self.alpha, self.T, self.multiplier = 0.9, 20, 100
        elif model == "densenet121": self.alpha, self.T, self.multiplier = 0.9, 20, 50
        else: self.alpha, self.T, self.multiplier = 0.9, 20, 50
    def forward(self, output, t_output, label):
        '''

        :param output: 현재 student logit
        :param t_output: CE로 훈련된 student logit
        :param label: 클래스 1,2,3,...
        :return:
        '''
        loss_CE = F.cross_entropy(output, label)
        D_KL = nn.KLDivLoss()(F.log_softmax(output/self.T, dim=1), F.softmax(t_output/self.T, dim=1)) * \
               (self.T * self.T) * self.multiplier
        KD_loss = (1 - self.alpha) * loss_CE + self.alpha * D_KL
        return KD_loss
class TFKD_t_regularization_bin(nn.Module):
    def __init__(self, model, alpha, n_cls):
        super().__init__()
        self.alpha = alpha
        self.KLDiv = Custom_CrossEntropy_PSKD()
        self.id_mat = torch.eye(n_cls).cuda()
    def forward(self, output, t_output, label, alpha_T):
        '''
        :param output: 현재 student logit
        :param t_output: CE로 훈련된 student logit
        :param label: 클래스 1,2,3,...
        :return:
        '''
        label_one_hot = self.id_mat[label.cpu().detach().numpy()]
        soft_label = (1-alpha_T) * label_one_hot + alpha_T * F.softmax(t_output, dim=1).detach()
        loss = self.KLDiv(output, soft_label)
        argsorted = torch.argsort(t_output, dim=1, descending=True)
        for idx, i in enumerate(range(0, 90 + 1, 5)):
            sub_out = torch.gather(output, 1, argsorted[:, i:i+10])
            sub_soft_label = F.softmax(torch.gather(soft_label, 1, argsorted[:, i:i+10]))
            loss += self.KLDiv(sub_out, sub_soft_label) * self.alpha
        return loss



# class TFKD_t_regularization_CE(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         if model == "resnet18": self.alpha, self.T, self.multiplier = 0.95, 6, 1
#         elif model == "resnext": self.alpha, self.T, self.multiplier = 0.9, 20, 100
#         elif model == "densenet121": self.alpha, self.T, self.multiplier = 0.9, 20, 50
#         else: self.alpha, self.T, self.multiplier = 0.9, 20, 50
#     def forward(self, output, t_output, label):
#         '''
#         :param output: 현재 student logit
#         :param t_output: CE로 훈련된 student logit
#         :param label: 0또는 1
#         :return:
#         '''
#         loss_CE = nn.KLDivLoss()(F.log_softmax(output, dim=1), label)
#         D_KL = nn.KLDivLoss()(F.log_softmax(output/self.T, dim=1), F.softmax(t_output/self.T, dim=1)) * \
#                (self.T * self.T) * self.multiplier
#         KD_loss = (1 - self.alpha) * loss_CE + self.alpha * D_KL
#         return KD_loss
# class TFKD_t_regularization_version9_CE(nn.Module):
#     def __init__(self, model, alpha, local_rank, n_cls):
#         super().__init__()
#         self.alpha = alpha
#         self.local_rank = local_rank
#         self.n_cls = n_cls
#         self.tfkd = TFKD_t_regularization(model)
#         self.tfkd_CE = TFKD_t_regularization_CE(model)
#     def forward(self, output, t_output, label):
#         loss = self.tfkd(output, t_output, label)
#         one_hot = torch.eye(self.n_cls).cuda(self.local_rank)[label]
#         argsorted = torch.argsort(t_output, dim=1, descending=True)
#         for idx, i in enumerate(range(0, 90 + 1, 5)):
#             sub_out = torch.gather(output, 1, argsorted[:, i:i+10])
#             sub_t_out = torch.gather(t_output, 1, argsorted[:, i:i+10])
#             sub_one_hot = torch.gather(one_hot, 1, argsorted[:, i:i+10])
#             loss += self.tfkd_CE(sub_out, sub_t_out, sub_one_hot) * self.alpha
#         return loss
# class TFKD_t_regularization_no_CE(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         if model == "resnet18": self.alpha, self.T, self.multiplier = 0.95, 6, 1
#         elif model == "resnext": self.alpha, self.T, self.multiplier = 0.9, 20, 2
#         elif model == "densenet121": self.alpha, self.T, self.multiplier = 0.9, 20, 1
#         else: self.alpha, self.T, self.multiplier = 0.9, 20, 1
#     def forward(self, output, t_output):
#         '''
#         :param output: 현재 student logit
#         :param t_output: CE로 훈련된 student logit
#         :param label: 0또는 1
#         :return:
#         '''
#         D_KL = nn.KLDivLoss()(F.log_softmax(output/self.T, dim=1), F.softmax(t_output/self.T, dim=1)) * \
#                (self.T * self.T) * self.multiplier
#         return D_KL
# class TFKD_t_regularization_version9_no_CE(nn.Module):
#     def __init__(self, model, alpha, local_rank, n_cls):
#         super().__init__()
#         self.alpha = alpha
#         self.local_rank = local_rank
#         self.n_cls = n_cls
#         self.tfkd = TFKD_t_regularization(model)
#         self.tfkd_no_CE = TFKD_t_regularization_no_CE(model)
#     def forward(self, output, t_output, label):
#         loss = self.tfkd(output, t_output, label)
#         argsorted = torch.argsort(t_output, dim=1, descending=True)
#         for idx, i in enumerate(range(0, 90 + 1, 5)):
#             sub_out = torch.gather(output, 1, argsorted[:, i:i+10])
#             sub_t_out = torch.gather(t_output, 1, argsorted[:, i:i+10])
#             loss += self.tfkd_no_CE(sub_out, sub_t_out) * self.alpha
#         return loss

if __name__ == "__main__":
    pass
