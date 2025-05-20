import torch
from torch.optim import SGD
from torch.optim import lr_scheduler
import warnings

class LinearDecayLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay = start_decay
        self.n_epoch = n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch = self.n_epoch
        b_lr = self.base_lrs[0]
        start_decay = self.start_decay
        if last_epoch > start_decay:
            lr = b_lr - b_lr / (n_epoch - start_decay) * (last_epoch - start_decay)
        else:
            lr = b_lr
        return [lr]
class DelayedStepLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, start_decay=0):
        self.step_size = step_size
        self.gamma = gamma
        self.start_decay = start_decay
        super(DelayedStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.start_decay:
            return [group['lr'] for group in self.optimizer.param_groups]

        if self.last_epoch - self.start_decay == 0:
            return [group['lr']*0.2 for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.start_decay) % self.step_size == 0:
            return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]

class LinearDecayLRBooster(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1, booster=2):
        self.start_decay = start_decay
        self.n_epoch = n_epoch
        self.booster = booster
        super(LinearDecayLRBooster, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch = self.n_epoch
        b_lr = self.base_lrs[-1]

        if last_epoch > 0:
            try:
                cur_lr = self.get_last_lr()
            except:
                cur_lr = b_lr * self.booster
        start_decay = self.start_decay

        if last_epoch >= start_decay:
            lr = b_lr * self.booster - (b_lr * self.booster) / (n_epoch - start_decay) * (last_epoch - start_decay)
        else:
            if last_epoch < start_decay:
                lr = b_lr + (b_lr * self.booster - b_lr) / start_decay * last_epoch
            else:
                lr = cur_lr

        self._last_lr = lr
        print(f'Active Learning Rate --- {lr}')
        return [lr]


if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, 0.001)
    s = LinearDecayLR(optimizer, 100, 75)
    ss = []
    for epoch in range(100):
        optimizer.step()
        s.step()
        ss.append(s.get_lr()[0])

    print(ss)
