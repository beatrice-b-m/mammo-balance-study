import torch
from torch import nn
import timm
# from torchvision.models import resnet34
# from torchvision.models.resnet import Bottleneck, BasicBlock
from torchinfo import summary
from argparse import ArgumentParser
import os


class ConvNext(nn.Module):
    def __init__(self, dropout_prop: float = 0.5, num_classes: int = 1):
        super(ConvNext, self).__init__()
        self.model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, in_chans=3, num_classes=1)

    def forward(self, xi):
        x = self.model(xi)
        return x

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    
#     parser = ArgumentParser("ConvNext Base")
#     parser.add_argument(
#         "--arch",
#         nargs='?',
#         help="Architecture version to test", 
#         type=int,
#         default=1
#     )
#     args = parser.parse_args()
    
    model = ConvNext()
    print(summary(model, input_size=(1, 3, 800, 600)))