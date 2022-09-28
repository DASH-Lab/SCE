import random
import os

from PIL import Image

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        # gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[0].mul_(0.299).add_(gs[1], alpha=0.587).add_(gs[2], alpha=0.114)
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs
class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)
class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)
class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)
class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)

        return transform(img)
class Custom_CIFAR10(CIFAR10):
    # ------------------------
    # Custom CIFAR-10 dataset which returns returns 1 images, 1 target, image index
    # ------------------------
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
class Custom_CIFAR100(CIFAR100):
    # ------------------------
    # Custom CIFAR-100 dataset which returns returns 1 images, 1 target, image index
    # ------------------------
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
class Custom_ImageFolder(ImageFolder):
    # ------------------------
    # Custom ImageFolder dataset which returns 1 images, 1 target, image index
    # ------------------------
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
def get_dataloader(args):
    if args.data_type == 'cifar10':
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        stdv = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        trainset = Custom_CIFAR10(root=args.data_root_dir, train=True, download=True,
                                                  transform=transform_train)
        validset = Custom_CIFAR10(root=args.data_root_dir, train=False, download=True,
                                                  transform=transform_val)

        if args.DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(trainset, pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.n_workers)

        valid_loader = torch.utils.data.DataLoader(validset, pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=None,
                                                   shuffle=False,
                                                   num_workers=args.n_workers)
    elif args.data_type == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        stdv = [0.2675, 0.2565, 0.2761]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        trainset = Custom_CIFAR100(root=args.data_root_dir, train=True, download=True,
                                                   transform=transform_train)
        validset = Custom_CIFAR100(root=args.data_root_dir, train=False, download=True,
                                                   transform=transform_val)

        if args.DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            print("[!] [Rank {}] Distributed Sampler Data Loading Done".format(args.local_rank))
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(trainset, pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.n_workers)

        valid_loader = torch.utils.data.DataLoader(validset, pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=None, shuffle=False,
                                                   num_workers=args.n_workers)
    elif args.data_type == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]
        jittering = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = Lighting(alphastd=0.1,
                            eigval=[0.2175, 0.0188, 0.0045],
                            eigvec=[[-0.5675, 0.7192, 0.4009],
                                   [-0.5808, -0.0045, -0.8140],
                                   [-0.5836, -0.6948, 0.4203]])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            jittering,
            lighting,
            transforms.Normalize(mean=mean, std=stdv),
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        trainset = Custom_ImageFolder(os.path.join(args.data_root_dir, "Imagenet", 'train'), transform=transform_train)
        validset = Custom_ImageFolder(os.path.join(args.data_root_dir, "Imagenet", 'val'), transform=transform_val)

        if args.DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            print("[!] [Rank {}] Distributed Sampler Data Loading Done".format(args.local_rank))
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(trainset,
                                                   pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.n_workers)

        valid_loader = torch.utils.data.DataLoader(validset,
                                                   pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=None,
                                                   shuffle=False,
                                                   num_workers=args.n_workers)
    else:
        raise Exception("[!] There is no option for Datatype")
    return train_loader, valid_loader, train_sampler

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jeonghokim/ICML_2022/src")
    from main import build_args
    args = build_args()
    if args.data_type == "imagenet":
        import torch.distributed as dist
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.local_rank)
    train_loader, valid_loader, train_sampler = get_dataloader(args)
    img, label, _ = next(iter(train_loader))
    print(img.shape)
