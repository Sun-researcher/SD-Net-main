import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from utils.get_picture import Dataset_Data
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from model import Detector
from utils.scheduler import LinearDecayLR
def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)
def compute_acc_video(pred, true):
    video_acc=float(compute_accuray(pred, true)>=0.5)
    return video_acc
def main(args):
    cfg = load_json(args.config)

    device = torch.device('cuda')

    image_size = cfg['image_size']
    batch_size = cfg['batch_size']
    test_dataset = Dataset_Data(phase='test', image_size=image_size, n_frames=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             collate_fn=test_dataset.collate_fn
                                             )
    model = Detector()
    model = model.to('cuda')
    cnn_sd = torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    model.train(mode=False)
    val_acc = 0.
    output_dict = []
    target_dict = []
    for step, data in enumerate(tqdm(test_loader)):
        img = data['img'].to(device, non_blocking=True).float()
        target = data['label'].to(device, non_blocking=True).long()
        with torch.no_grad():
            output_cla = model.forward_classifier(img)
        acc = compute_accuray(F.log_softmax(output_cla, dim=1), target)
        val_acc += acc
        output_dict += output_cla.softmax(1)[:, 1].cpu().data.numpy().tolist()
        target_dict += target.cpu().data.numpy().tolist()
    val_acc=val_acc / len(test_loader)
    val_auc = roc_auc_score(target_dict, output_dict)
    print(f'AUC: {val_auc:.4f}|  ACC: {val_acc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='store_true',
                        default='src/configs/sbi/base.json')
    parser.add_argument('-n', '--session_name', action='store_true', default='sbi')  # -n sbi
    parser.add_argument('--continue_train', dest='weight_name',type=str, default='output/10_1000_0.0727_0.8340_val_FPN_mask_CBAM_HB.tar')
    args = parser.parse_args()
    main(args)