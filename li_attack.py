import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class CarliniLi:
    def __init__(self, model, args, targeted=False):
        self.model = model
        self.targeted = targeted

        self.device = args.device
        self.model.to(self.device)

    def attack(self, images, labels, c=1.5, tau=1, kappa=0, max_iter=1000, learning_rate=0.01):

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

        w = torch.zeros_like(images, requires_grad=True).to(self.device)


        optimizer = optim.Adam([w], lr=learning_rate)

        prev = 1e10

        for step in range(1,max_iter):

            a = 1 / 2 * (nn.Tanh()(w) + 1)
            loss1 = torch.sum(F.relu(torch.abs(images-a) - tau))
            # loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(c * f(a))

            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (max_iter // 5) == 0:
                if cost > prev:
                    return a
                prev = cost
            if torch.all(w < tau).item():
                tau *= 0.9  # Reduce tau

        attack_images = 1 / 2 * (nn.Tanh()(w) + 1)

        return attack_images
