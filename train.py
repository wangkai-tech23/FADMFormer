import os
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch
import sys
import yaml
from pathlib import Path
from colorama import Fore
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import TrainSetLoader, TestSetLoader
from metric import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA, ROCMetric
from FADMFormer import FADMFormer
from argparse import ArgumentParser
import torch.nn as nn
import os.path as ops
import time
torch.cuda.manual_seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resume = False
resume_dir = ''
def load_dataset (root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'val_test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,val_img_ids
class Loss(object):
    def __init__(self):
        self.reset()
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
class Loss1(nn.Module):
    def __init__(self):
        super(Loss1, self).__init__()
        self.iou = None
    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 0.00
        target = target.float()
        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)
        loss = 1 - torch.mean(loss)
        return loss
def parse_args():
    parser = ArgumentParser(description='Implementation of FADMFormer')
    parser.add_argument('--dataset', type=str, default='IRSTD-1k',help='dataset:IRSTD-1k; NUDT-SIRST; NUAA-SIRST;')
    parser.add_argument('--form', type=str, default='.png')
    parser.add_argument('--workers', type=int, default=8, metavar='N')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--optimizer', type=str, default='Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR')
    parser.add_argument('--min_lr', default=5e-3, type=float)
    args = parser.parse_args()
    return args
def train_one_epoch(model, optimizer, data_loader, device, epoch, loss):
    model.train()
    loss_function = loss
    losses = Loss()
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET))
    for step, data in enumerate(data_loader):
        images, labels = data
        labels[labels > 0] = 1
        labels = torch.Tensor(labels).long().to(device)
        pred = model(images.to(device))
        if isinstance(pred, list):
            loss = 0
            for p in pred:
                loss += loss_function(p, labels)
            loss /= len(pred)
            pred = pred[-1]
        else:
            loss = loss_function(pred, labels)
        losses.update(loss.item(), pred.size(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        data_loader.desc = "[train epoch {}] loss: {:.8f}".format(epoch, losses.avg)
    return losses.avg
def evaluate(model, data_loader, device, epoch, iou_metric, nIoU_metric, PD_FA,ROC, len_val, loss):
    loss_function = loss
    model.eval()
    iou_metric.reset()
    nIoU_metric.reset()
    PD_FA.reset()
    ROC.reset()
    losses = Loss()
    data_loader = tqdm(data_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.MAGENTA, Fore.RESET))
    for step, data in enumerate(data_loader):
        images, labels = data
        labels[labels > 0] = 1
        labels = torch.Tensor(labels).long().to(device)
        pred = model(images.to(device))
        if isinstance(pred, list):
            loss = 0
            for p in pred:
                loss += loss_function(p, labels)
            loss /= len(pred)
            pred = pred[-1]
        else:
            loss = loss_function(pred, labels)
        losses.update(loss.item(), pred.size(0))
        pred, labels = pred.cpu(), labels.cpu()
        iou_metric.update(pred, labels)
        nIoU_metric.update(pred, labels)
        ROC.update(pred, labels)
        PD_FA.update(pred, labels)
        FA, PD = PD_FA.get(len_val)
        ture_positive_rate, false_positive_rate, recall, precision = ROC.get()
        _, IoU = iou_metric.get()
        _, nIoU = nIoU_metric.get()
        data_loader.desc = "[train epoch {}] loss: {:.6f}, IoU: {:.6f}, nIoU: {:.6f}".format(epoch, losses.avg, IoU, nIoU)
    return losses.avg, IoU, nIoU, ture_positive_rate, false_positive_rate, recall, precision, PD, FA
def main(args):
    dataset = args.dataset
    cfg = yaml.load(open(Path(__file__).parent / "options/traintestcfg.yml", "r"), Loader=yaml.FullLoader)
    root, split_method, size, batch, aug = cfg['dataset']['root'], cfg['dataset'][dataset]['split_method'], \
                                      cfg['dataset'][dataset]['size'], cfg['dataset'][dataset]['batch'], cfg['dataset'][dataset]['aug']
    args.batch_size = batch
    args.aug = aug
    args.model = cfg['dataset'][dataset]['model']
    train_img_ids, val_img_ids= load_dataset(root, dataset, split_method)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + dataset + "_" + split_method + "_" + args.model)
    tb_writer = SummaryWriter(log_dir=log_dir)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    dataset_dir = root + '/' + dataset
    print('dataset_dir',dataset_dir)
    args.use_prior = True
    print('use_prior_loss: ', args.use_prior)
    trainset = TrainSetLoader(dataset_dir, img_id=train_img_ids, base_size=size, crop_size=size,
                              transform=input_transform,form=args.form, aug=args.aug, useprior=True)
    valset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=size, crop_size=size,
                            transform=input_transform,form=args.form)
    train_data = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=True, pin_memory=True)
    val_data = DataLoader(dataset=valset, batch_size=args.batch_size, num_workers=args.workers,drop_last=False)
    model = FADMFormer(input_channels=3,channel=[16, 32, 64, 128],
                      depth=[2, 2, 2], drop=0., attn_drop=0., drop_path=0.1,
                     num_heads=[[1, 2, 3, 4], [1, 3, 4], [1, 3], [1]],win_size=8, img_size=size)
    print('# model_restoration parameters: %.2f M' % (sum(param.numel() for param in model.parameters()) / 1e6))
    model = model.to(device)
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=args.min_lr)
    restart = 0
    if resume == True:
        ckpt = torch.load(resume_dir)
        print(ckpt['mean_IOU'])
        model.load_state_dict(ckpt['state_dict'], strict=True)
        restart = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt["scheduler"])
        print('resuming')
    best_iou = 0
    best_nIoU = 0
    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    roc = ROCMetric(1, 10)
    pdfa = PD_FA(1, 10)
    folder_name = '%s_%s_%s' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),args.dataset, args.model)
    save_folder = log_dir
    save_pkl = ops.join(save_folder, 'checkpoint')
    if not ops.exists('result'):
        os.mkdir('result')
    if not ops.exists(save_folder):
        os.mkdir(save_folder)
    if not ops.exists(save_pkl):
        os.mkdir(save_pkl)
    tb_writer.add_text(folder_name, 'Args:%s, ' % args)
    loss_fun = Loss1().to(device)
    miou_name = ' '
    niou_name = ' '
    for epoch in range(restart+1, args.epochs):
        train_loss= train_one_epoch(model, optimizer, train_data, device, epoch, loss_fun)
        val_loss, iou_, niou_,ture_positive_rate, false_positive_rate, recall, precision, pd, fa = \
        evaluate(model, val_data, device, epoch, iou_metric, niou_metric, pdfa,roc, len(valset), loss_fun)
        tags = ['train_loss', 'val_loss', 'IoU', 'nIoU', 'mIoU', 'PD', 'tp', 'fa', 'rc', 'pr']
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], iou_, epoch)
        tb_writer.add_scalar(tags[3], niou_, epoch)
        name = 'Epoch-%3d_IoU-%.4f_nIoU-%.4f.pth.tar' % (epoch, iou_, niou_)
        if resume == True or (resume == False and epoch >= 100):
            if iou_ > best_iou:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'loss': val_loss,
                    'IOU': iou_,
                    'n_IoU': niou_,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(save_pkl,'Best_IoU_' + name))
                best_iou = iou_
                if ops.exists(ops.join(save_pkl, 'Best_IoU_' + miou_name)):
                    os.remove(ops.join(save_pkl, 'Best_IoU_' + miou_name))
                miou_name = name
            if niou_ > best_nIoU:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'loss': val_loss,
                    'IOU': iou_,
                    'n_IoU': niou_,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(save_pkl,'Best_nIoU_' + name))
                best_nIoU = niou_
                if ops.exists(ops.join(save_pkl, 'Best_nIoU_' + niou_name)):
                    os.remove(ops.join(save_pkl, 'Best_nIoU_' + niou_name))
                niou_name = name
if __name__ == '__main__':
    args = parse_args()
    main(args)