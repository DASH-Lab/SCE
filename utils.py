import sys
from os.path import join as opj

import torch
import torch.distributed as dist
import numpy as np


class Logger(object):
    def __init__(self, local_rank):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank

    def open(self, fp, mode=None):
        if self.local_rank != 0: return
        if mode is None: mode = 'w'
        self.file = open(fp, mode)

    def write(self, msg, is_terminal=1, is_file=1):
        if self.local_rank != 0: return
        if '\r' in msg: is_file = 0
        if is_terminal == 1:
            self.terminal.write(msg)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(msg)
            self.file.flush()

    def flush(self):
        pass
class AverageMeter (object):
    def __init__(self):
        self.reset ()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def print_args(args, logger=None):
  for k, v in vars(args).items():
      if logger is not None:
          logger.write('{:25s}: {}\n'.format(k, v))
      else:
          print('{:25s}: {}'.format(k, v))
def Accuracy(output, label, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = label.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def adjust_learning_rate(args, epoch, optimizer):
    lr = args.lr
    for m in args.lr_decay_epochs:
        lr *= args.lr_decay_rate if epoch >= m else 1.
    for g in optimizer.param_groups:
        g["lr"] = lr
def get_learning_rate(optimizer):
    lr = []
    for g in optimizer.param_groups:
        lr += [g['lr']]
    return lr
def model_save(args, logger, model, optimizer, epoch, best_acc, save_path):
    if args.local_rank == 0:
        checkpoint = {"optimizer": optimizer.state_dict(), "epoch": epoch, "best_acc": best_acc,
                      "model": model.module.state_dict() if args.DDP else model.state_dict()}
        torch.save(checkpoint, save_path)
        logger.write(f"[Epoch-{epoch}]_[best acc-{best_acc}] model save!!!!\n")
def dist_model_save_load(args, model, suf=None):
    if suf == 1:
        ckp_path = opj(args.save_model_dir, "checkpoint_1.pth")
    elif suf == 2:
        ckp_path = opj(args.save_model_dir, "checkpoint_2.pth")
    else:
        ckp_path = opj(args.save_model_dir, "checkpoint.pth")
    if args.local_rank==0:
        torch.save(model.module.state_dict(), ckp_path)
    dist.barrier()
    model.module.load_state_dict(torch.load(ckp_path, map_location={"cuda:0":f"cuda:{args.local_rank}"}))
def metric_ece_aurc_eaurc(confidences, truths, bin_size=0.1):
    confidences = np.asarray(confidences)
    truths = np.asarray(truths)

    total = len(confidences)
    predictions = np.argmax(confidences, axis=1)
    max_confs = np.amax(confidences, axis=1)

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    accs = []
    avg_confs = []
    bin_counts = []
    ces = []

    for upper_bound in upper_bounds:
        lower_bound = upper_bound - bin_size
        acc, avg_conf, bin_count = compute_bin(lower_bound, upper_bound, max_confs, predictions, truths)
        accs.append(acc)
        avg_confs.append(avg_conf)
        bin_counts.append(bin_count)
        ces.append(abs(acc - avg_conf) * bin_count)

    ece = 100 * sum(ces) / total

    aurc, e_aurc = calc_aurc(confidences, truths)

    return ece, aurc * 1000, e_aurc * 1000
def compute_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        avg_conf = sum([x[2] for x in filtered_tuples]) / len(filtered_tuples)
        accuracy = float(correct) / len(filtered_tuples)
        bin_count = len(filtered_tuples)
        return accuracy, avg_conf, bin_count
def calc_aurc(confidences, labels):
    confidences = np.array(confidences)
    labels = np.array(labels)
    predictions = np.argmax(confidences, axis=1)
    max_confs = np.max(confidences, axis=1)

    n = len(labels)
    indices = np.argsort(max_confs)
    labels, predictions, confidences = labels[indices][::-1], predictions[indices][::-1], confidences[indices][::-1]
    risk_cov = np.divide(np.cumsum(labels != predictions).astype(np.float), np.arange(1, n + 1))
    nrisk = np.sum(labels != predictions)
    aurc = np.mean(risk_cov)
    opt_aurc = (1. / n) * np.sum(
        np.divide(np.arange(1, nrisk + 1).astype(np.float), n - nrisk + np.arange(1, nrisk + 1)))
    eaurc = aurc - opt_aurc

    return aurc, eaurc




