import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class CarliniL0:
    def __init__(self, model, args, targeted=False, CONST=0.001, CONST_LIt=100):
        self.model = model
        self.targeted = targeted

        self.device = args.device
        self.model.to(self.device)
        self.CONST_limit = CONST_LIt
        self.CONST = CONST

    def attack(self, images, labels, c=1, kappa=0, max_iter=100, learning_rate=0.01):
        self.CONST = c
        active_pixels = torch.ones_like(images, dtype=torch.bool, device=self.device)

        # Define f-function
        def f(x):
            outputs = self.model(x)
            one_hot_labels = torch.eye(len(outputs[0]), device=self.device)[labels]

            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())

            # If targeted, optimize for making the other class most likely
            if self.targeted:
                return torch.clamp(i - j, min=-kappa)

            # If untargeted, optimize for making the other class most likely
            else:
                return torch.clamp(j - i, min=-kappa)

        V = torch.ones_like(images, requires_grad=False).to(self.device)
        w = torch.zeros_like(images, requires_grad=True).to(self.device)  # 优化的参数
        optimizer = optim.Adam([w], lr=learning_rate)

        # 准备初始预测和原始图像的副本
        pred = 1 / 2 * (nn.Tanh()(w + 1))  # 需要detach()以避免梯度追踪
        original = images.clone().detach()

        def l2_attack():
            while self.CONST < self.CONST_limit:
                pred_loss = 1e6
                for step in range(max_iter):
                    a = 1 / 2 * (nn.Tanh()(w + 1)) * V + (1 - V) * original
                    loss1 = nn.MSELoss(reduction='mean')(a, images)
                    loss2 = torch.sum(f(a))

                    cost = loss1 + self.CONST * loss2

                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
                    # 检查是否攻击成功
                    if step % (max_iter // 5) == 0:
                        if loss2 < 0.01/V.shape[0]:
                            return True, a, w.grad.data

                # 更新CONST
                self.CONST *= 2
            return False, a, w.grad.data

        num = 0
        while True:
            flag, nimg, w_grad = l2_attack()
            if flag:
                pred=nimg
                grad_abs = torch.abs(w_grad)
                diff_abs = torch.abs(nimg - original)
                importance = grad_abs * diff_abs
                importance = torch.sum(importance, dim=1, keepdim=True)
                num += 1

                active_pixels_0 = active_pixels[:, 0, :, :].unsqueeze(1)
                importance[~active_pixels_0] = float('inf')
                flatten_importance = importance.view(importance.size(0), -1)
                min_indices = torch.argmin(flatten_importance, dim=1)
                print("V", V.sum())

                for i in range(3):
                    batch_indices = torch.arange(V.size(0)).to(V.device)
                    V[batch_indices, i, min_indices // 32, min_indices % 32] = 0
                    active_pixels[batch_indices, i, min_indices // 32, min_indices % 32] = False

            else:
                return pred

            if num > 1010:
                return 1 / 2 * (nn.Tanh()(w + 1)) * V + (1 - V) * original
