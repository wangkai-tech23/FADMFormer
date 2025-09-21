import os
import shutil
import sys
import torch
from FADMFormer import FADMFormer
from argparse import ArgumentParser
from colorama import Fore
from PIL import Image
from dataset import TestSetLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from metric import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA, ROCMetric
model_dir=''
root_dir = model_dir.split('Best')[0]
model_name = model_dir.split('Best')[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def parse_args():
    parser = ArgumentParser(description='Implement of FADMFormer')
    parser.add_argument('--dataset', type=str, default='IRSTD-1k',help='dataset:IRSTD-1k; NUDT-SIRST; NUAA-SIRST;')
    parser.add_argument('--form', type=str, default='.png')
    args = parser.parse_args()
    return args
def load_dataset (root, dataset, split_method):
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'val_test.txt'
    val_img_ids = []
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return val_img_ids,test_txt
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
def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, form, size):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(size, size))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +form)
    img = Image.fromarray(labelsss.reshape(size, size))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + form)
def total_visulization_generation(dataset_dir, test_txt, form, target_image_path, target_dir, list, size):
    source_image_path = dataset_dir + '/images'
    target_dir = target_dir + '/fuse'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + form
        target_image = target_image_path + '/' + ids[i] + form
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + form
        img = Image.open(source_image)
        img = img.resize((size, size), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        iou = list[m]
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + form)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)
        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + form)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)
        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + form)
        plt.imshow(img, cmap='gray')
        if iou<0.01:
            iou=0.00
        plt.xlabel("Predicts"+str(iou)[:5], size=11)
        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + form, facecolor='w', edgecolor='red')
        plt.close()
@torch.no_grad()
def visual(model, data_loader, device, epoch, iou_metric, nIoU_metric, PD_FA,ROC, len_val, path, ids, form, dataset_dir, test_txt, size):
    model.eval()
    iou_metric.reset()
    nIoU_metric.reset()
    PD_FA.reset()
    ROC.reset()
    losses = Loss()
    data_loader = tqdm(data_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.MAGENTA, Fore.RESET))
    iou_list = []
    for step, data in enumerate(data_loader):
        images, labels = data
        labels[labels > 0] = 1
        labels = torch.Tensor(labels).long().to(device)
        pred = model(images.to(device))
        if isinstance(pred, list):
            loss = 0
            loss /= len(pred)
            pred = pred[-1]
        pred0 = torch.sigmoid(pred)
        pred0[pred0 > 0.5] = 1
        pred0[pred0 <= 0.5] = 0
        label = labels.float()
        label[label > 0] = 1
        intersection = pred0 * label
        loss_iou = (intersection.sum() + 1e-6) / (pred0.sum() + label.sum() - intersection.sum() + 1e-6)
        iou_list.append(loss_iou.tolist())
        pred, labels = pred.cpu(), labels.cpu()
        iou_metric.update(pred, labels)
        nIoU_metric.update(pred, labels)
        ROC.update(pred, labels)
        PD_FA.update(pred, labels)
        FA, PD = PD_FA.get(len_val)
        ture_positive_rate, false_positive_rate, recall, precision = ROC.get()
        _, IoU = iou_metric.get()
        _, nIoU = nIoU_metric.get()
        data_loader.desc = "[valid epoch {}] loss: {:.6f}, mIoU: {:.6f}, nIoU: {:.6f}".format(epoch, losses.avg, IoU, nIoU)
        save_Pred_GT(pred, labels, path, ids, step, form, size)
    total_visulization_generation(dataset_dir, test_txt, form, path, path, iou_list, size)
    return losses.avg, IoU, nIoU, ture_positive_rate, false_positive_rate, recall, precision, PD, FA
def main(args):
    dataset = args.dataset
    cfg = yaml.load(open(Path(__file__).parent / "options/traintestcfg.yml", "r"), Loader=yaml.FullLoader)
    root, split_method, size, batch= cfg['dataset']['root'], cfg['dataset'][dataset]['split_method'], \
        cfg['dataset'][dataset]['size'], cfg['dataset'][dataset]['batch']
    dataset_dir = root + '/' + dataset
    val_img_ids, test_txt = load_dataset(root, dataset, split_method)
    log_dir = os.path.join(root_dir, model_name + "_visual")
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    valset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=size, crop_size=size,
                           transform=input_transform, form=args.form)
    val_data = DataLoader(dataset=valset, batch_size=1, num_workers=0,drop_last=False)
    model = FADMFormer(input_channels=3,channel=[16, 32, 64, 128],
                      depth=[2, 2, 2], drop=0., attn_drop=0., drop_path=0.1,
                     num_heads=[[1, 2, 3, 4], [1, 3, 4], [1, 3], [1]],win_size=8, img_size=size)
    model = model.to(device)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True)
    print(f"FLOPs: {macs}")
    print('# model_restoration parameters: %.2f M' % (sum(param.numel() for param in model.parameters()) / 1e6))
    ckpt = torch.load(model_dir, map_location=device)
    try:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    except RuntimeError as e:
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.'):
                name = k[7:]
            else:
                name = k
            new_dict[name] = v
        model.load_state_dict(new_dict, strict=True)
    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    roc = ROCMetric(1, 10)
    pdfa = PD_FA(1, 10)
    save_folder = log_dir
    save_visual = os.path.join(save_folder, 'visual')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(save_visual):
        os.mkdir(save_visual)
    val_loss, iou_, niou_, miou_, ture_positive_rate, false_positive_rate, recall, precision, pd, fa = \
        visual(model, val_data, device, 0, iou_metric, niou_metric, pdfa, roc, len(valset),
               save_visual, val_img_ids, args.form, dataset_dir, test_txt, args.size)
    note = open(os.path.join(save_folder, args.dataset+args.split_method+'.txt'), mode='w')
    note.write('IoU:\n')
    note.write('{}\n'.format(iou_))
    note.write('nIoU:\n')
    note.write('{}\n'.format(niou_))
    note.write('TP:\n')
    note.write('{}\n'.format(ture_positive_rate))
    note.write('FP:\n')
    note.write('{}\n'.format(false_positive_rate))
    note.write('recall:\n')
    note.write('{}\n'.format(recall))
    note.write('precision:\n')
    note.write('{}\n'.format(precision))
    note.write('PD:\n')
    note.write('{}\n'.format(pd))
    note.write('FA:\n')
    note.write('{}\n'.format(fa))
if __name__ == '__main__':
    args = parse_args()
    main(args)