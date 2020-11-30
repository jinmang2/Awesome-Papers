import torch
import torch.nn as nn


class Father(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc  = nn.Linear(10, 10)
        setattr(self.fc.__class__, 'do', lambda f: f())
        self.fc.do(self.Do)

    @staticmethod
    def Do():
        print(self)


a = Father()

print(a)
print(a.fc)
    