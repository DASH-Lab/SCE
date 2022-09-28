import argparse
import time
import os
import wandb

from torch.cuda.amp import autocast, GradScaler

from utils import *
from loss.pskd_loss import *
from loss.tfkd_loss import *
from models.network import get_network
from dataset.dataloader import get_dataloader

def build_args():
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument('--data_type', type=str, default="cifar100", choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument('--data_root_dir', type=str, default="/home/data")
    parser.add_argument("--n_workers", type=int, default=12)

    #### train & test ####
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--model', type=str, default='resnet18', help='Select classifier',
                        choices=["resnet18", "resnet101", "densenet121", "resnext", "pyramidnet", "pyramidnet_sd",
                                 "resnet152"])
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', type=int, default=[150, 180, 210], nargs='+', help='when to drop lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
    parser.add_argument("--n_epochs", default=240, type=int)
    parser.add_argument("--KD_method", type=str, default="PSKD_bin",
                        choices=["PSKD", "PSKD_bin", "CSKD", "CE", "TFKD", "TFKD_bin"])
    parser.add_argument("--alpha_T", type=float, default=0.8)

    #### save & load ####
    parser.add_argument("--save_root_dir", default="/media/data1/jeonghokim/ICML_2022/save")
    parser.add_argument("--model_load_path", default=None)
    parser.add_argument("--save_note", default="test")

    #### config ####
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument("--DDP", action="store_true")
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_notes", type=str, default="test")

    args = parser.parse_args()
    if args.model == "resnext": args.alpha = 0.01
    else: args.alpha = 0.1
    if args.data_type == "imagenet":
        imagenet_args(args)
    args.save_name = f"[data_type-{args.data_type}]_[model-{args.model}]_[KD method-{args.KD_method}]_" \
                     f"[save_note-{args.save_note}]_[batch_size-{args.batch_size}]_[optimizer-{args.optimizer}]_" \
                     f"[alpha-{args.alpha}]"
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.save_model_dir = opj(args.save_dir, "save_models")
    args.logger_path = opj(args.save_dir, "log.txt")
    args.wandb_name=args.save_name
    os.makedirs(args.save_model_dir, exist_ok=True)
    return args
def imagenet_args(args):
    args.batch_size = 256
    args.weight_decay = 1e-4
    args.lr_decay_epochs = [30, 60]
    args.n_epochs = 90
def train(args,
          model,
          train_loader,
          criterion_CE,
          optimizer,
          epoch,
          all_predictions,
          alpha_t,
          criterion_KD = None):
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
            img = torch.autograd.Variable(img).cuda(args.local_rank)
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
def main(args, logger):
    torch.cuda.set_device(args.local_rank)
    model = get_network(args).cuda(args.local_rank)
    if args.DDP:
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.local_rank)
        logger.write(f"DDP Using {torch.cuda.device_count()} GPUS\n")
        args.n_workers  = args.n_workers // torch.cuda.device_count()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    train_loader, valid_loader, train_sampler = get_dataloader(args)
    criterion_CE = nn.CrossEntropyLoss()
    criterion_KD = None
    #### Losses ####
    if args.KD_method == "PSKD": criterion_KD = Custom_CrossEntropy_PSKD(args.local_rank)
    elif args.KD_method == "PSKD_bin": criterion_KD = Custom_CrossEntropy_PSKD_version9(args.local_rank, args.alpha)
    elif args.KD_method == "CSKD": pass
    elif args.KD_method == "CE": pass
    elif args.KD_method == "TFKD": criterion_KD = TFKD_regularization(args.model, args.local_rank)
    elif args.KD_method == "TFKD_bin": criterion_KD = TFKD_regularization_version9(args.model, args.local_rank)
    elif args.KD_method == "onlineKD": criterion_KD = Custom_CrossEntropy_PSKD(args.local_rank)
    elif args.KD_method == "onlineKD_bin": criterion_KD = Custom_CrossEntropy_PSKD_version9(args.local_rank, args.alpha)
    else: raise NotImplementedError(f"KD method {args.KD_method} is not implemented!!!!")

    if args.KD_method == "PSKD" or args.KD_method == "PSKD_bin":
        all_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes),
                                      dtype=torch.float32).cuda(args.local_rank)
    else:
        all_predictions = None

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
        if args.KD_method == "PSKD" or args.KD_method == "PSKD_bin":
            alpha_t = args.alpha_T * (epoch / args.n_epochs)
            alpha_t = max(0, alpha_t)
        else:
            alpha_t = None
        train_loss, train_top1_acc, train_top5_acc, all_predictions = train(args=args,
                                                                            model=model,
                                                                            train_loader=train_loader,
                                                                            criterion_CE=criterion_CE,
                                                                            optimizer=optimizer,
                                                                            epoch=epoch,
                                                                            all_predictions=all_predictions,
                                                                            alpha_t=alpha_t,
                                                                            criterion_KD=criterion_KD)
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
    main(args, logger)
    logger.write(f"time: {time.time() - start}")
