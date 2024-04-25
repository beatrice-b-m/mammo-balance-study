# import torch
# from torch import nn
# from torchvision.models import resnet18, resnet34, resnet50
# from torchvision.models.resnet import Bottleneck, BasicBlock
# from torchinfo import summary
# from argparse import ArgumentParser
# import os


# # class ResNet50(nn.Module):
# #     def __init__(self, arch_num: str = '1', dropout_prop: float = 0.5, num_classes: int = 1):
# #         super(ResNet50, self).__init__()
# #         arch_dict = {
# #             "1": [1000, 512, 32, num_classes], 
# #         }

# #         self.base_model = resnet50(weights='IMAGENET1K_V1')
        
# #         # define classifier
# #         arch_layer_list = arch_dict[arch_num]
        
# #         # build the list of activation functions
# #         # should be approx [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Sigmoid()]
# #         # the last should be either softmax or sigmoid depending on the number of
# #         # predicted classes
# #         activation_func_list = [nn.ReLU()] * (len(arch_layer_list) - 2)
# #         activation_func_list.append(None) # using logits so we'll just append a None to skip instead of a sigmoid
        
# #         # get a list to store classifier layers in and add a dropout
# #         clf_layer_list = []
# #         clf_layer_list.append(nn.Dropout(dropout_prop))

# #         # attach the dense layers to the classifier
# #         for idx in range(len(arch_layer_list) - 1):
# # #             print(f"linear: {sel_arch_layer_list[layer_num]}>{sel_arch_layer_list[layer_num + 1]}")
# #             clf_layer_list += [nn.Linear(arch_layer_list[idx], arch_layer_list[idx + 1])]
    
# #             if activation_func_list[idx] is not None:
# #                 clf_layer_list += [activation_func_list[idx]]
        
# #         # get a sequential model by unpacking the list into the nn.Sequential wrapper
# #         self.classifier = nn.Sequential(*clf_layer_list)

# #     def forward(self, xi):
# #         x = self.base_model(xi)
# #         x = self.classifier(x)
# #         return x

# # class ResNet34(nn.Module):
# #     def __init__(self, arch_num: str = '1', dropout_prop: float = 0.5, num_classes: int = 1):
# #         super(ResNet34, self).__init__()
# #         arch_dict = {
# #             "1": [1000, 512, 32, num_classes], 
# #         }

# #         self.base_model = resnet34(weights='IMAGENET1K_V1')
        
# #         # define classifier
# #         arch_layer_list = arch_dict[arch_num]
        
# #         # build the list of activation functions
# #         # should be approx [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Sigmoid()]
# #         # the last should be either softmax or sigmoid depending on the number of
# #         # predicted classes
# #         activation_func_list = [nn.ReLU()] * (len(arch_layer_list) - 2)
# #         activation_func_list.append(None) # using logits so we'll just append a None to skip instead of a sigmoid
        
# #         # get a list to store classifier layers in and add a dropout
# #         clf_layer_list = []
# #         clf_layer_list.append(nn.Dropout(dropout_prop))

# #         # attach the dense layers to the classifier
# #         for idx in range(len(arch_layer_list) - 1):
# # #             print(f"linear: {sel_arch_layer_list[layer_num]}>{sel_arch_layer_list[layer_num + 1]}")
# #             clf_layer_list += [nn.Linear(arch_layer_list[idx], arch_layer_list[idx + 1])]
    
# #             if activation_func_list[idx] is not None:
# #                 clf_layer_list += [activation_func_list[idx]]
        
# #         # get a sequential model by unpacking the list into the nn.Sequential wrapper
# #         self.classifier = nn.Sequential(*clf_layer_list)

# #     def forward(self, xi):
# #         x = self.base_model(xi)
# #         x = self.classifier(x)
# #         return x
    
# # class ResNet18(nn.Module):
# #     def __init__(self, arch_num: str = '1', dropout_prop: float = 0.5, num_classes: int = 1):
# #         super(ResNet18, self).__init__()
# #         arch_dict = {
# #             "1": [1000, 512, 32, num_classes], 
# #         }

# #         self.base_model = resnet18(weights='IMAGENET1K_V1')
        
# #         # define classifier
# #         arch_layer_list = arch_dict[arch_num]
        
# #         # build the list of activation functions
# #         # should be approx [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Sigmoid()]
# #         # the last should be either softmax or sigmoid depending on the number of
# #         # predicted classes
# #         activation_func_list = [nn.ReLU()] * (len(arch_layer_list) - 2)
# #         activation_func_list.append(None) # using logits so we'll just append a None to skip instead of a sigmoid
        
# #         # get a list to store classifier layers in and add a dropout
# #         clf_layer_list = []
# #         clf_layer_list.append(nn.Dropout(dropout_prop))

# #         # attach the dense layers to the classifier
# #         for idx in range(len(arch_layer_list) - 1):
# # #             print(f"linear: {sel_arch_layer_list[layer_num]}>{sel_arch_layer_list[layer_num + 1]}")
# #             clf_layer_list += [nn.Linear(arch_layer_list[idx], arch_layer_list[idx + 1])]
    
# #             if activation_func_list[idx] is not None:
# #                 clf_layer_list += [activation_func_list[idx]]
        
# #         # get a sequential model by unpacking the list into the nn.Sequential wrapper
# #         self.classifier = nn.Sequential(*clf_layer_list)

# #     def forward(self, xi):
# #         x = self.base_model(xi)
# #         x = self.classifier(x)
# #         return x

# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"]=""
    
#     parser = ArgumentParser("ResNet50")
#     parser.add_argument(
#         "--arch",
#         nargs='?',
#         help="Architecture version to test", 
#         type=int,
#         default=1
#     )
#     args = parser.parse_args()
    
#     model = ResNet(str(args.arch))
#     print(summary(model, input_size=(1, 3, 800, 600)))