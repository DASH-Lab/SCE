from models import *
import torch
from .vit import My_Vit
if torch.__version__ == "1.10.1+cu113":
    from torchvision.models import efficientnet_b0
    from torchvision.models import efficientnet_b1
    from torchvision.models import efficientnet_b2

def get_network(args):
    if args.data_type == 'cifar100':
        if args.model.lower() == 'pyramidnet':
            net = PyramidNet(dataset = 'cifar100', depth=200, alpha=240, num_classes=100,bottleneck=True)
        elif args.model.lower() == 'pyramidnet_sd':
            net = PyramidNet_ShakeDrop(dataset = 'cifar100', depth=200, alpha=240, num_classes=100,bottleneck=True)
        elif args.model.lower() == 'resnet18':
            net = CIFAR_ResNet18_preActBasic(num_classes=100)
        elif args.model.lower() == "resnet50":
            net = CIFAR_ResNet50_Bottle(num_classes=100)
        elif args.model.lower() == 'resnet101':
            net = CIFAR_ResNet101_Bottle(num_classes=100)
        elif args.model.lower() == "resnet152":
            net = CIFAR_ResNet152_Bottle(num_classes=100)
        elif args.model.lower() == 'densenet121':
            net = CIFAR_DenseNet121(num_classes=100, bias=True)
        elif args.model.lower() == 'resnext':
            net = CifarResNeXt(cardinality=8, depth=29, nlabels=100, base_width=64, widen_factor=4)

    if args.data_type == 'imagenet':
        if args.model.lower() == 'resnet152':
            net = ResNet(dataset = 'imagenet', depth=152, num_classes=1000, bottleneck=True)                
        elif args.model.lower() == "resnet18":
            net = ResNet(dataset='imagenet', depth=18, num_classes=1000, bottleneck=True)
        elif args.model.lower() == "resnet50":
            net = ResNet(dataset='imagenet', depth=50, num_classes=1000, bottleneck=True)
        elif args.model.lower() == "resnet101":
            net = ResNet(dataset='imagenet', depth=101, num_classes=1000, bottleneck=True)
        elif args.model.lower() == "densenet121":
            net = torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=False, force_reload=True)
        elif args.model.lower() == "efficientnet_b0":
            net = efficientnet_b0(pretrained=False)
        elif args.model.lower() == "efficientnet_b1":
            net = efficientnet_b1(pretrained=False)
        elif args.model.lower() == "efficientnet_b2":
            net = efficientnet_b2(pretrained=False)
        elif args.model.lower() == "pyramidnet":
            net = PyramidNet(dataset = 'imagenet', depth=164, alpha=48, num_classes=1000, bottleneck=True)
        elif args.model.lower() == "vit":
            net = My_Vit()
        # elif args.model.lower() == "vit":
        #     net = ViT(image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1, emb_dropout=0.1)
        else: raise NotImplementedError(f"Model {args.model.lower()} is not implemented!!!!")
    return net
