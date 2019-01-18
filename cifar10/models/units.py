import torch.nn as nn

class MyReLU(nn.Threshold):
    """
     This is just an alias of ReLU
    """
    def __init__(self, inplace=False):
        super(MyReLU, self).__init__(0, 0, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str