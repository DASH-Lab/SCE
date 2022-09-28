import argparse
import time
import os
import wandb
import copy

from torch.cuda.amp import autocast, GradScaler

from utils import *
from loss.pskd_loss import *
from loss.tfkd_loss import *
from loss.tfkd_t_loss import TFKD_t_regularization, TFKD_t_regularization_bin
from models.network import get_network
from dataset.dataloader import get_dataloader

def build_args():
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument('--data_type', type=str, default="cifar100", choices=["cifar10", "cifar100", "imagenet"])  ####
    parser.add_argument('--data_root_dir', type=str, default="/home/data")
    parser.add_argument("--n_workers", type=int, default=4)  ####

    #### train & test ####
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--model', type=str, default='resnet18', help='Select classifier',  ####
                        choices=["resnet18", "resnet101", "densenet121", "resnext", "pyramidnet", "pyramidnet_sd",
                                 "resnet152", "resnet50", "efficientnet_b0", "efficientnet_b0", "efficientnet_b0", "efficientnet_b0", "vit"])
    parser.add_argument("--model1", type=str, default="resnet18", help="For DML")
    parser.add_argument("--model2", type=str, default="resnet18", help="For DML")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', type=int, default=[150, 180, 210], nargs='+', help='when to drop lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
    parser.add_argument("--n_epochs", default=240, type=int)
    parser.add_argument("--KD_method", type=str, default="PSKD_bin",  ####
                        choices=["PSKD", "PSKD_bin", "CSKD", "CE", "TFKD", "TFKD_bin", "TFKD_t", "TFKD_t_bin", "DML", "DML_bin"])
    parser.add_argument("--alpha_T", type=float, default=0.8)
    parser.add_argument("--T", type=float, default=1.5)

    #### save & load ####
    parser.add_argument("--save_root_dir", default="/media/data1/jeonghokim/ICML_2022/save")
    parser.add_argument("--model_load_path", default=None)
    parser.add_argument("--save_note", default="test")
    parser.add_argument("--t_model_load_dir", type=str, default="/home/jeonghokim/ICML_2022/src/models/teacher_models")

    #### config ####
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument("--DDP", action="store_true")
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_notes", type=str, default="test")

    args = parser.parse_args()
    if args.data_type=="cifar10": args.n_cls = 10
    elif args.data_type=="cifar100": args.n_cls = 100
    elif args.data_type=="imagenet": args.n_cls = 1000
    if args.model == "resnext" or args.model == "resnet101" or args.model == "densenet121": args.alpha = 0.01
    elif args.model == "resnet50" or args.model == "resnet152" or args.model == "vit": args.alpha = 0.01
    elif args.model == "pyramidnet": args.alpha = 0.005
    else: args.alpha = 0.1
    if args.KD_method == "DML" or args.KD_method == "DML_bin":
        if args.model1 == "resnet18" and args.model2 == "resnet18": args.alpha = 0.1
        elif args.model1 == "resnet50" and args.model2 == "resnet50": args.alpha = 0.01
    if args.data_type == "imagenet": imagenet_args(args)
    if args.DDP: args.n_workers = args.n_workers // torch.cuda.device_count()
    args.save_name = f"[data_type-{args.data_type}]_[model-{args.model}]_[KD method-{args.KD_method}]_" \
                     f"[save_note-{args.save_note}]_[batch_size-{args.batch_size}]_[optimizer-{args.optimizer}]_" \
                     f"[alpha-{args.alpha}]"
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.save_model_dir = opj(args.save_dir, "save_models")
    args.logger_path = opj(args.save_dir, "log.txt")
    if args.KD_method == "TFKD_t" or args.KD_method == "TFKD_t_bin":
        args.t_model_load_path = opj(args.t_model_load_dir, f"{args.model}.pth")
    args.wandb_name=args.save_name
    os.makedirs(args.save_model_dir, exist_ok=True)
    return args
def imagenet_args(args):
    args.batch_size = 256 // torch.cuda.device_count()
    args.weight_decay = 1e-4
    args.lr_decay_epochs = [30, 60]
    args.n_epochs = 90
    args.alpha_T = 0.3
    args.DDP = True
    if args.model.lower() == "vit":
        args.batch_size = 512 // torch.cuda.device_count()
        args.weight_decay = 0.03
        args.lr = 1e-2

def train(args,
          model,
          train_loader,
          criterion_CE,
          optimizer,
          epoch,
          all_predictions,
          alpha_t,
          criterion_KD=None,
          teacher_model=None):
    train_loss = AverageMeter()
    train_top1_acc = AverageMeter()
    train_top5_acc = AverageMeter()
    model.train()
    scaler = GradScaler()
    for img, label, input_indices in train_loader:
        img = img.cuda(args.local_rank)
        label = label.cuda(args.local_rank)

        if args.KD_method == "PSKD" or args.KD_method == "PSKD_bin":
            labels_numpy = label.cpu().detach().numpy()
            id_mat = torch.eye(len(train_loader.dataset.classes)).cuda(args.local_rank)
            labels_one_hot = id_mat[labels_numpy]

            if epoch == 1:
                all_predictions[input_indices] = labels_one_hot
            soft_labels = ((1 - alpha_t) * labels_one_hot) + (alpha_t * all_predictions[input_indices])
            soft_labels = torch.autograd.Variable(soft_labels).cuda(args.local_rank)
            img = torch.autograd.Variable(img, requires_grad=True).cuda(args.local_rank)
            with autocast():
                output = model(img)
                softmax_output = F.softmax(output, dim=1)
                loss_ = criterion_KD(output, soft_labels)
            if args.DDP:
                gathered_prediction = [torch.ones_like(softmax_output) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_prediction, softmax_output)
                gathered_prediction = torch.cat(gathered_prediction, dim=0)

                gathered_indices = [torch.ones_like(input_indices.cuda(args.local_rank)) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indices, input_indices.cuda(args.local_rank))
                gathered_indices = torch.cat(gathered_indices, dim=0)
        elif args.KD_method == "TFKD" or args.KD_method == "TFKD_bin":
            with autocast():
                output = model(img)
                loss_ = criterion_KD(output, label)
        elif args.KD_method == "CE":
            with autocast():
                output = model(img)
                loss_ = criterion_CE(output, label)
        elif args.KD_method == "TFKD_t" or args.KD_method == "TFKD_t_bin":
            with autocast():
                t_output = teacher_model(img)
                t_output = torch.autograd.Variable(t_output, requires_grad=False)
                output = model(img)
                loss_ = criterion_KD(output, t_output, label, alpha_t)
        else:
            raise NotImplementedError(f"KD method {args.KD_method} is not implemented!!!!")

        optimizer.zero_grad()
        scaler.scale(loss_).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = Accuracy(output, label, topk=(1,5))
        train_loss.update(loss_.item(), img.shape[0])
        train_top1_acc.update(acc1.item(), img.shape[0])
        train_top5_acc.update(acc5.item(), img.shape[0])

        if args.KD_method == "PSKD" or args.KD_method == "PSKD_bin":
            if args.DDP:
                for jdx in range(len(gathered_prediction)):
                    all_predictions[gathered_indices[jdx]] = gathered_prediction[jdx]
            else:
                all_predictions[input_indices] = softmax_output
    if all_predictions is not None:
        return train_loss.avg, train_top1_acc.avg, train_top5_acc.avg, all_predictions.clone().detach()
    else:
        return train_loss.avg, train_top1_acc.avg, train_top5_acc.avg, all_predictions
def validation(args,
               model,
               valid_loader,
               criterion):
    valid_loss = AverageMeter()
    valid_top1_acc = AverageMeter()
    valid_top5_acc = AverageMeter()
    label_lst = []
    confidence = []
    model.eval()
    with torch.no_grad():
        for img, label, _ in valid_loader:
            img = img.cuda(args.local_rank)
            label = label.cuda(args.local_rank)
            label_np = label.cpu().detach().numpy()
            label_lst.extend(label_np.tolist())
            with autocast():
                output = model(img)
                loss_ = criterion(output, label)
            softmax_prediction = F.softmax(output, dim=1)
            softmax_prediction = softmax_prediction.cpu().detach().numpy()
            for v in softmax_prediction:
                confidence.append(v.tolist())

            acc1, acc5 = Accuracy(output, label, topk=(1,5))
            valid_loss.update(loss_.item(), img.shape[0])
            valid_top1_acc.update(acc1.item(), img.shape[0])
            valid_top5_acc.update(acc5.item(), img.shape[0])
    ece, aurc, eaurc = metric_ece_aurc_eaurc(confidence, label_lst, bin_size=0.1)
    metrics = {"valid_loss": valid_loss.avg,
               "valid_top1_acc": valid_top1_acc.avg,
               "valid_top5_acc": valid_top5_acc.avg,
               "ece": ece,
               "aurc": aurc,
               "eaurc": eaurc}
    return metrics
def train_imagenet_PSKD(args,
                   model,
                   prev_model,
                   train_loader,
                   criterion_KD,
                   optimizer,
                   epoch,
                   alpha_t):
    train_loss = AverageMeter()
    train_top1_acc = AverageMeter()
    train_top5_acc = AverageMeter()
    model.train()
    if prev_model is not None:
        prev_model.eval()
    scaler = GradScaler()
    for img, label, input_indices in train_loader:
        img = img.cuda(args.local_rank)
        label = label.cuda(args.local_rank)
        labels_numpy = label.cpu().detach().numpy()
        id_mat = torch.eye(len(train_loader.dataset.classes)).cuda(args.local_rank)
        labels_one_hot = id_mat[labels_numpy]
        if epoch == 1:  prev_pred = labels_one_hot
        else:
            with torch.no_grad():
                with autocast():
                    prev_pred = F.softmax(prev_model(img), dim=1)
        soft_labels = (1 - alpha_t) * labels_one_hot + alpha_t * prev_pred
        soft_labels = torch.autograd.Variable(soft_labels).cuda(args.local_rank)
        img = torch.autograd.Variable(img, requires_grad=True).cuda(args.local_rank)
        with autocast():
            output = model(img)
            loss_ = criterion_KD(output, soft_labels)
        optimizer.zero_grad()
        scaler.scale(loss_).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = Accuracy(output, label, topk=(1,5))
        train_loss.update(loss_.item(), img.shape[0])
        train_top1_acc.update(acc1.item(), img.shape[0])
        train_top5_acc.update(acc5.item(), img.shape[0])
    del prev_model
    return train_loss.avg, train_top1_acc.avg, train_top5_acc.avg, copy.deepcopy(model)
def train_imagenet_DML(args,
                       model1,
                       model2,
                       train_loader,
                       criterion_CE,
                       optimizer1,
                       optimizer2,
                       epoch,
                       alpha_t,
                       criterion_KD):
    train_loss1 = AverageMeter()
    train_loss2 = AverageMeter()
    train_top1_acc1 = AverageMeter()
    train_top5_acc1 = AverageMeter()
    train_top1_acc2 = AverageMeter()
    train_top5_acc2 = AverageMeter()
    model1.train()
    model2.train()
    scaler = GradScaler()
    for img, label, input_indices in train_loader:
        img = img.cuda(args.local_rank)
        label = label.cuda(args.local_rank)
        identity_mat = torch.eye(1000)
        if args.KD_method == "DML":
            with autocast():
                output1 = model1(img)
                output2 = model2(img)

                loss1 = criterion_CE(output1, label)
                loss1 += criterion_KD(F.log_softmax(output1, dim=1), F.softmax(output2.detach(), dim=1))
            optimizer1.zero_grad()
            scaler.scale(loss1).backward()
            scaler.step(optimizer1)
            scaler.update()
            with autocast():
                loss2 = criterion_CE(output2, label)
                loss2 += criterion_KD(F.log_softmax(output2, dim=1), F.softmax(output1.detach(), dim=1))

            optimizer2.zero_grad()
            scaler.scale(loss2).backward()
            scaler.step(optimizer2)
            scaler.update()
        elif args.KD_method == "DML_bin":
            label_np = label.cpu().detach().numpy()
            label_one_hot = identity_mat[label_np].cuda()
            with autocast():
                output1 = model1(img)
                output2 = model2(img)

                soft_label1 = ((1-alpha_t)*label_one_hot) + (alpha_t * F.softmax(output2.detach(), dim=1))
                loss1 = criterion_KD(output1, soft_label1)
            optimizer1.zero_grad()
            scaler.scale(loss1).backward()
            scaler.step(optimizer1)
            scaler.update()
            with autocast():
                soft_label2 = ((1 - alpha_t) * label_one_hot) + (alpha_t * F.softmax(output1.detach(), dim=1))
                loss2 = criterion_KD(output2, soft_label2)
            optimizer2.zero_grad()
            scaler.scale(loss2).backward()
            scaler.step(optimizer2)
            scaler.update()
        else: raise NotImplementedError(f"KD method {args.KD_method} is not implemented!!!")
        acc1, acc5 = Accuracy(output1, label, topk=(1, 5))
        train_loss1.update(loss1.item(), img.shape[0])
        train_top1_acc1.update(acc1.item(), img.shape[0])
        train_top5_acc1.update(acc5.item(), img.shape[0])

        acc1, acc5 = Accuracy(output2, label, topk=(1, 5))
        train_loss2.update(loss2.item(), img.shape[0])
        train_top1_acc2.update(acc1.item(), img.shape[0])
        train_top5_acc2.update(acc5.item(), img.shape[0])
    return train_loss1.avg, train_top1_acc1.avg, train_top5_acc1.avg, \
           train_loss2.avg, train_top1_acc2.avg, train_top5_acc2.avg
def main_imagenet_PSKD(args, logger):  # DDP 전제로 한다.
    torch.cuda.set_device(args.local_rank)
    model = get_network(args).cuda(args.local_rank)
    prev_model = None
    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.local_rank)
    logger.write(f"DDP Using {torch.cuda.device_count()} GPUS\n")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    train_loader, valid_loader, train_sampler = get_dataloader(args)
    criterion_CE = nn.CrossEntropyLoss()
    if args.KD_method == "PSKD": criterion_KD = Custom_CrossEntropy_PSKD(args.local_rank)
    elif args.KD_method == "PSKD_bin": criterion_KD = Custom_CrossEntropy_PSKD_version9_imagenet(args.local_rank, args.alpha)
    else: raise NotImplementedError(f"KD method {args.KD_method} is not implemented!!!!")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    else: raise NotImplementedError(f"Optimizer {args.optimizer} is not implemented!!!!")
    start_epoch  = 1
    best_top1_acc = -1
    best_top5_acc = -1
    best_ece = -1
    best_aurc = -1
    best_eaurc = -1
    if args.model_load_path:
        logger.write(f"model load from {args.model_load_path}\n")
        if not args.DDP:
            checkpoint = torch.load(args.model_load_path)
        else:
            dist.barrier()
            checkpoint = torch.load(args.model_load_path, map_location={"cuda:0": f"cuda:{args.local_rank}"})
        start_epoch = checkpoint["epoch"] + 1
        best_top1_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.write(f"model is successfully loaded!!!!\n"
                     f"start epoch: {args.start_epoch}, alpha_T: {args.alpha_T}, best acc: {best_top1_acc}\n")
        del checkpoint
    for epoch in range(start_epoch, args.n_epochs):
        start_t = time.time()
        adjust_learning_rate(args, epoch, optimizer)
        train_sampler.set_epoch(epoch)
        alpha_t = args.alpha_T * (epoch / args.n_epochs)
        alpha_t = max(0, alpha_t)

        train_loss, train_top1_acc, train_top5_acc, prev_model = train_imagenet_PSKD(args=args,
                                                                                     model=model,
                                                                                     prev_model=prev_model,
                                                                                     train_loader=train_loader,
                                                                                     criterion_KD=criterion_KD,
                                                                                     optimizer=optimizer,
                                                                                     epoch=epoch,
                                                                                     alpha_t=alpha_t)
        valid_metrics = validation(args=args,
                                   model=model,
                                   valid_loader=valid_loader,
                                   criterion=criterion_CE)
        valid_loss = valid_metrics["valid_loss"]
        valid_top1_acc = valid_metrics["valid_top1_acc"]
        valid_top5_acc = valid_metrics["valid_top5_acc"]
        valid_ece = valid_metrics["ece"]
        valid_aurc = valid_metrics["aurc"]
        valid_eaurc = valid_metrics["eaurc"]

        best_top5_acc = max(best_top5_acc, valid_top5_acc)
        best_ece = max(best_ece, valid_ece)
        best_aurc = max(best_aurc, valid_aurc)
        best_eaurc = max(best_eaurc, valid_eaurc)

        if valid_top1_acc > best_top1_acc and args.local_rank == 0:
            best_top1_acc = valid_top1_acc
            save_path = opj(args.save_model_dir, f"[epoch-{epoch}]_[best_top1_acc-{best_top1_acc}].pth")
            model_save(args=args,
                       logger=logger,
                       model=model,
                       optimizer=optimizer,
                       epoch=epoch,
                       best_acc=best_top1_acc,
                       save_path=save_path)
        logger.write(f"[Epoch-{epoch}]_[Time-{time.time() - start_t}]_[valid top1 acc-{valid_top1_acc:.4f}]_[valid top5 acc-{valid_top5_acc:.4f}]_"
                     f"[valid ece-{valid_ece:.4f}]_[valid aurc-{valid_aurc:.4f}]_[valid eaurc-{valid_eaurc:.4f}]\n")
        dist_model_save_load(args, model)
        if args.use_wandb and args.local_rank == 0:
            wandb_msg = {"train loss": train_loss,
                         "train acc1": train_top1_acc,
                         "train acc5": train_top5_acc,
                         "valid loss": valid_loss,
                         "valid acc1": valid_top1_acc,
                         "valid acc5": valid_top5_acc,
                         "valid ece": valid_ece,
                         "valid aurc": valid_aurc,
                         "valid eaurc": valid_eaurc}
            wandb.log(wandb_msg)
    logger.write(f"[Best]_[top1 acc-{best_top1_acc:.4f}]_[top5 acc-{best_top5_acc:.4f}]_[ece-{best_ece:.4f}]_"
                 f"[aurc-{best_aurc:.4f}]_[eaurc-{best_eaurc:.4f}]\n")
def main_imagenet_DML(args, logger):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.local_rank)
    logger.write(f"DDP Using {torch.cuda.device_count()} GPUS\n")
    model1 = get_network(args).cuda(args.local_rank)
    model2 = get_network(args).cuda(args.local_rank)
    model1 = torch.nn.parallel.DistributedDataParallel(model1, device_ids=[args.local_rank])
    model2 = torch.nn.parallel.DistributedDataParallel(model2, device_ids=[args.local_rank])
    train_loader, valid_loader, train_sampler = get_dataloader(args)
    criterion_CE = nn.CrossEntropyLoss()
    if args.KD_method == "DML": criterion_KD = nn.KLDivLoss(reduction="batchmean").cuda(args.local_rank)
    elif args.KD_method == "DML_bin": criterion_KD = Custom_CrossEntropy_PSKD_version9_imagenet(args.local_rank, args.alpha)
    else: raise NotImplementedError(f"KD method {args.KD_method} is not implemented!!!!")
    if args.optimizer == "SGD":
        optimizer1 = torch.optim.SGD(model1.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
        optimizer2 = torch.optim.SGD(model2.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optimizer == "Adam":
        optimizer1 = torch.optim.Adam(model1.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model2.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    else: raise NotImplementedError(f"Optimizer {args.optimizer} is not implemented!!!!")
    start_epoch = 1
    best_top1_acc1 = -1
    best_top5_acc1 = -1
    best_ece1 = -1
    best_aurc1 = -1
    best_eaurc1 = -1

    best_top1_acc2 = -1
    best_top5_acc2 = -1
    best_ece2 = -1
    best_aurc2 = -1
    best_eaurc2 = -1
    if args.model_load_path:
        model_load_path1 = args.model_load_path.replace(".pth", "_1.pth")
        model_load_path2 = args.model_load_path.replace(".pth", "_2.pth")
        logger.write(f"model load from {model_load_path1}, {model_load_path2}\n")
        if not args.DDP:
            checkpoint1 = torch.load(model_load_path1)
            checkpoint2 = torch.load(model_load_path2)
        else:
            dist.barrier()
            checkpoint1 = torch.load(model_load_path1, map_location={"cuda:0": f"cuda:{args.local_rank}"})
            checkpoint2 = torch.load(model_load_path2, map_location={"cuda:0": f"cuda:{args.local_rank}"})
        start_epoch = checkpoint1["epoch"] + 1
        best_top1_acc1 = checkpoint1["best_acc"]
        model1.load_state_dict(checkpoint1["model"])
        optimizer1.load_state_dict(checkpoint1["optimizer"])
        best_top1_acc2 = checkpoint2["best_acc"]
        model2.load_state_dict(checkpoint2["model"])
        optimizer2.load_state_dict(checkpoint2["optimizer"])
        logger.write(f"model is successfully loaded!!!!\n"
                     f"start epoch: {args.start_epoch}, alpha_T: {args.alpha_T}, best acc1: {best_top1_acc1}"
                     f"best acc2: {best_top1_acc2}\n")
        del checkpoint1; del checkpoint2
    for epoch in range(start_epoch, args.n_epochs):
        start_t = time.time()
        adjust_learning_rate(args, epoch ,optimizer1)
        adjust_learning_rate(args, epoch, optimizer2)
        train_sampler.set_epoch(epoch)
        alpha_t = args.alpha_T * (epoch / args.n_epochs)
        alpha_t = max(0, alpha_t)
        train_metrics = train_imagenet_DML(args, model1, model2, train_loader, criterion_CE, optimizer1,
                                           optimizer2, epoch, alpha_t, criterion_KD)

        valid_metrics1 = validation(args=args,
                                   model=model1,
                                   valid_loader=valid_loader,
                                   criterion=criterion_CE)
        valid_metrics2 = validation(args=args,
                                    model=model2,
                                    valid_loader=valid_loader,
                                    criterion=criterion_CE)
        valid_loss1 = valid_metrics1["valid_loss"]
        valid_top1_acc1 = valid_metrics1["valid_top1_acc"]
        valid_top5_acc1 = valid_metrics1["valid_top5_acc"]
        valid_ece1 = valid_metrics1["ece"]
        valid_aurc1 = valid_metrics1["aurc"]
        valid_eaurc1 = valid_metrics1["eaurc"]

        valid_loss2 = valid_metrics2["valid_loss"]
        valid_top1_acc2 = valid_metrics2["valid_top1_acc"]
        valid_top5_acc2 = valid_metrics2["valid_top5_acc"]
        valid_ece2 = valid_metrics2["ece"]
        valid_aurc2 = valid_metrics2["aurc"]
        valid_eaurc2 = valid_metrics2["eaurc"]

        best_top5_acc1 = max(best_top5_acc1, valid_top5_acc1)
        best_ece1 = max(best_ece1, valid_ece1)
        best_aurc1 = max(best_aurc1, valid_aurc1)
        best_eaurc1 = max(best_eaurc1, valid_eaurc1)
        best_top5_acc2 = max(best_top5_acc2, valid_top5_acc2)
        best_ece2 = max(best_ece2, valid_ece2)
        best_aurc2 = max(best_aurc2, valid_aurc2)
        best_eaurc2 = max(best_eaurc2, valid_eaurc2)

        if valid_top1_acc1 > best_top1_acc1 and args.local_rank == 0:
            best_top1_acc1 = valid_top1_acc1
            save_path = opj(args.save_model_dir, f"[Model1-{args.model1}]_[epoch-{epoch}]_[best_top1_acc1-{best_top1_acc1}].pth")
            model_save(args=args,
                       logger=logger,
                       model=model1,
                       optimizer=optimizer1,
                       epoch=epoch,
                       best_acc=best_top1_acc1,
                       save_path=save_path)
        if valid_top1_acc2 > best_top1_acc2 and args.local_rank == 0:
            best_top1_acc2 = valid_top1_acc1
            save_path = opj(args.save_model_dir, f"[Model2-{args.model2}]_[epoch-{epoch}]_[best_top1_acc1-{best_top1_acc2}].pth")
            model_save(args=args,
                       logger=logger,
                       model=model2,
                       optimizer=optimizer2,
                       epoch=epoch,
                       best_acc=best_top1_acc2,
                       save_path=save_path)
        logger.write(f"[Epoch-{epoch}]_[Time-{time.time()-start_t}]_[valid top1 acc1-{valid_top1_acc1:.4f}]_"
                     f"[valid_top5_acc1-{valid_top5_acc1:.4f}]_[valid ece1-{valid_ece1:.4f}]_"
                     f"[valid aurc1-{valid_aurc1:.4f}]_[valid eaurc1-{valid_eaurc1:.4f}]\n")
        logger.write(f"[Epoch-{epoch}]_[Time-{time.time() - start_t}]_[valid top1 acc2-{valid_top1_acc2:.4f}]_"
                     f"[valid_top5_acc2-{valid_top5_acc2:.4f}]_[valid ece2-{valid_ece2:.4f}]_"
                     f"[valid aurc2-{valid_aurc2:.4f}]_[valid eaurc2-{valid_eaurc2:.4f}]\n")
        dist_model_save_load(args, model1, suf=1)
        time.sleep(5)
        dist_model_save_load(args, model2, suf=2)

    logger.write(f"[Best]_[top1 acc1-{best_top1_acc1:.4f}]_[top5 acc1-{best_top5_acc1:.4f}]_"
                 f"[ece1-{best_ece1:.4f}]_[aurc1-{best_aurc1:.4f}]_[eaurc1-{best_eaurc1:.4f}]_"
                 f"[top1 acc2-{best_top1_acc2:.4f}]_[top5 acc1-{best_top5_acc2:.4f}]_[ece2-{best_ece2:.4f}]_"
                 f"[aurc2-{best_aurc2:.4f}]_[eaurc2-{best_eaurc2:.4f}]\n")

def main(args, logger):
    torch.cuda.set_device(args.local_rank)
    model = get_network(args).cuda(args.local_rank)
    if args.DDP:
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.local_rank)
        logger.write(f"DDP Using {torch.cuda.device_count()} GPUS\n")
        if args.model.lower() == "vit":
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    train_loader, valid_loader, train_sampler = get_dataloader(args)
    criterion_CE = nn.CrossEntropyLoss()
    criterion_KD = None
    #### Losses ####
    if args.KD_method == "PSKD": criterion_KD = Custom_CrossEntropy_PSKD(args.local_rank)
    elif args.KD_method == "PSKD_bin": criterion_KD = Custom_CrossEntropy_PSKD_version9(args.local_rank, args.alpha)
    elif args.KD_method == "CE": pass
    elif args.KD_method == "TFKD": criterion_KD = TFKD_regularization(args.model, args.local_rank)
    elif args.KD_method == "TFKD_bin": criterion_KD = TFKD_regularization_version9(args.model, args.local_rank, args.alpha)
    elif args.KD_method == "TFKD_t": criterion_KD = TFKD_t_regularization(args.model)
    elif args.KD_method == "TFKD_t_bin": criterion_KD = TFKD_t_regularization_bin(args.model, args.alpha, args.n_cls)
    elif args.KD_method == "onlineKD": criterion_KD = Custom_CrossEntropy_PSKD(args.local_rank)
    elif args.KD_method == "onlineKD_bin": criterion_KD = Custom_CrossEntropy_PSKD_version9(args.local_rank, args.alpha)
    else: raise NotImplementedError(f"KD method {args.KD_method} is not implemented!!!!")
    if args.KD_method == "PSKD" or args.KD_method == "PSKD_bin":
        all_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes),
                                      dtype=torch.float32).cuda(args.local_rank)
    else: all_predictions = None

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} is not implemented!!!!")
    teacher_model = None
    if args.KD_method == "TFKD_t" or args.KD_method == "TFKD_t_bin":
        teacher_model = get_network(args).cuda(args.local_rank)
        teacher_model.load_state_dict(torch.load(args.t_model_load_path)["model"])
        logger.write(f"teacher model load from {args.t_model_load_path}!!!!\n")
        teacher_model.eval()
    start_epoch = 1
    best_top1_acc = -1
    best_top5_acc = -1
    best_ece = -1
    best_aurc = -1
    best_eaurc = -1
    if args.model_load_path:
        logger.write(f"model load from {args.model_load_path}\n")
        if not args.DDP:
            checkpoint = torch.load(args.model_load_path)
        else:
            dist.barrier()
            checkpoint = torch.load(args.model_load_path, map_location={"cuda:0": f"cuda:{args.local_rank}"})
        start_epoch = checkpoint["epoch"] + 1
        best_top1_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.write(f"model is successfully loaded!!!!\n"
                     f"start epoch: {args.start_epoch}, alpha_T: {args.alpha_T}, best acc: {best_top1_acc}\n")
        del checkpoint
    for epoch in range(start_epoch, args.n_epochs):
        adjust_learning_rate(args, epoch, optimizer)
        if args.DDP:
            train_sampler.set_epoch(epoch)
        alpha_t = args.alpha_T * (epoch / args.n_epochs)
        alpha_t = max(0, alpha_t)
        train_loss, train_top1_acc, train_top5_acc, all_predictions = train(args=args,
                                                                            model=model,
                                                                            train_loader=train_loader,
                                                                            criterion_CE=criterion_CE,
                                                                            optimizer=optimizer,
                                                                            epoch=epoch,
                                                                            all_predictions=all_predictions,
                                                                            alpha_t=alpha_t,
                                                                            criterion_KD=criterion_KD,
                                                                            teacher_model=teacher_model)
        valid_metrics  = validation(args=args,
                                    model=model,
                                    valid_loader=valid_loader,
                                    criterion=criterion_CE)
        valid_loss = valid_metrics["valid_loss"]
        valid_top1_acc = valid_metrics["valid_top1_acc"]
        valid_top5_acc = valid_metrics["valid_top5_acc"]
        valid_ece = valid_metrics["ece"]
        valid_aurc = valid_metrics["aurc"]
        valid_eaurc = valid_metrics["eaurc"]

        best_top5_acc = max(best_top5_acc, valid_top5_acc)
        best_ece = max(best_ece, valid_ece)
        best_aurc = max(best_aurc, valid_aurc)
        best_eaurc = max(best_eaurc, valid_eaurc)
        if valid_top1_acc > best_top1_acc and args.local_rank == 0:
            best_top1_acc = valid_top1_acc
            save_path = opj(args.save_model_dir, f"[epoch-{epoch}]_[best_top1_acc-{best_top1_acc}].pth")
            model_save(args=args,
                       logger=logger,
                       model=model,
                       optimizer=optimizer,
                       epoch=epoch,
                       best_acc=best_top1_acc,
                       save_path=save_path)
        logger.write(f"[Epoch-{epoch}]_[valid top1 acc-{valid_top1_acc:.4f}]_[valid top5 acc-{valid_top5_acc:.4f}]_"
                     f"[valid ece-{valid_ece:.4f}]_[valid aurc-{valid_aurc:.4f}]_[valid eaurc-{valid_eaurc:.4f}]\n")
        if args.DDP:
            dist_model_save_load(args=args,
                                 model=model)
        if args.use_wandb and args.local_rank == 0:
            wandb_msg = {"train loss": train_loss,
                         "train acc1": train_top1_acc,
                         "train acc5": train_top5_acc,
                         "valid loss": valid_loss,
                         "valid acc1": valid_top1_acc,
                         "valid acc5": valid_top5_acc,
                         "valid ece": valid_ece,
                         "valid aurc": valid_aurc,
                         "valid eaurc": valid_eaurc}
            wandb.log(wandb_msg)
    logger.write(f"[Best]_[top1 acc-{best_top1_acc:.4f}]_[top5 acc-{best_top5_acc:.4f}]_[ece-{best_ece:.4f}]_"
                 f"[aurc-{best_aurc:.4f}]_[eaurc-{best_eaurc:.4f}]\n")
if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.logger_path)
    print_args(args, logger)
    if args.use_wandb and args.local_rank == 0:
        wandb.init(project="ICML 2022", name=args.wandb_name, notes=args.wandb_notes)
        wandb.config.update(args)
    start = time.time()
    if args.data_type != "imagenet": main(args, logger)
    elif args.data_type == "imagenet":
        if args.KD_method == "PSKD" or args.KD_method == "PSKD_bin": main_imagenet_PSKD(args, logger)
        elif args.KD_method == "DML" or args.KD_method == "DML_bin": main_imagenet_DML(args, logger)
        else: main(args, logger)
    else: raise NotImplementedError(f"Data type {args.data_type} is not implemented!!!!")
    logger.write(f"time: {time.time() - start}")
