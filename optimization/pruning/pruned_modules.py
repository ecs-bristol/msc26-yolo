
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class C2fPruningFriendly(nn.Module):

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()

        self.c = int(c2 * e)

        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)

        self.cv2 = Conv(
            (2 + n) * self.c,
            c2,
            1,
            1
        )

        self.m = nn.ModuleList(
            [nn.Identity() for _ in range(n)]
        )

    def forward(self, x):

        y = [
            self.cv0(x),
            self.cv1(x)
        ]

        y.extend(
            m(y[-1])
            for m in self.m
        )

        return self.cv2(
            torch.cat(y, dim=1)
        )
