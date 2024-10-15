import argparse
import os
import sys

sys.path.append("..")

from collections import OrderedDict
import imageio.v2 as imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc
from model.double_seg_head import network_du_heads_independent_config
from model.double_seg_head import network_du_heads_independent_config_cl
from model.double_seg_head import network_du_heads_independent_config_cl_gl
from model.double_seg_head import network_du_heads_independent_fusion_cam

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs, format_tabs_multi_metircs

parser = argparse.ArgumentParser()
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
# parser.add_argument("--model_path", default="workdir_voc_final2/2022-11-04-01-50-48-441426/checkpoints/model_iter_20000.pth", type=str, help="model_path")
parser.add_argument("--model_path", default="path_to_ur_checkpoints", type=str, help="model_path")

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--data_folder", default='../VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='train/val/test list file', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--scales", default=(1.0, 1.5, 1.25), help="multi_scales for seg")
parser.add_argument("--save_images", default=False, type=bool, help="save images")
import torch

def compute_classwise_iou(mask1, mask2, num_classes):

    iou_list = []


    for cls in range(1, num_classes):  # 从1开始，因为0通常是背景
        mask1_cls = (mask1 == cls)
        mask2_cls = (mask2 == cls)

        intersection = torch.sum(mask1_cls & mask2_cls)
        union = torch.sum(mask1_cls | mask2_cls)


        if union == 0:
            iou = torch.tensor(0.0) 
        else:
            iou = intersection / union
            iou_list.append(iou)
        
        
    if len(iou_list)==0:
        iou_list.append(0)
    mean_iou = torch.mean(torch.tensor(iou_list))

    return mean_iou, iou_list

def _validate(model=None, data_loader=None, args=None):
    # model.to('cuda:5')
    model.eval()
    color_map = plt.get_cmap("Blues")

    with torch.no_grad(), torch.cuda.device(5):
        model.cuda()

        gts, seg_pred = [], []
        mIOU_list = []
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            _, _, h, w = inputs.shape
            seg_list_1 = []
            seg_list_2 = []
            for sc in args.scales:
                _h, _w = int(h*sc), int(w*sc)

                _inputs  = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                # inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
                inputs_cat = _inputs
                (segs_1,segs_2) = model(inputs_cat,)[1]

                segs_1 = F.interpolate(segs_1, size=labels.shape[1:], mode='bilinear', align_corners=False)
                segs_2 = F.interpolate(segs_2, size=labels.shape[1:], mode='bilinear', align_corners=False)
                
                
                # seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))
                # seg = segs[:1,...] + segs[1:,...].flip(-1)

                seg_list_1.append(segs_1)
                seg_list_2.append(segs_2)
            seg_1 = torch.max(torch.stack(seg_list_1, dim=0), dim=0)[0]
            seg_2 = torch.max(torch.stack(seg_list_2, dim=0), dim=0)[0]   
            seg1_pred = torch.argmax(seg_1, dim=1).cpu()    
            seg2_pred = torch.argmax(seg_2, dim=1).cpu()    
            mIOU = compute_classwise_iou(seg1_pred,seg2_pred,20)
            mIOU_list.append(mIOU)
        
    mean_iou_on_datasets = torch.mean(torch.tensor(mIOU_list))

    print(mean_iou_on_datasets)
    
    return 0


def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.data_folder, 'JPEGImages',)
    labels_path = os.path.join(args.data_folder, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=1,   # 3
        pos_w=1,        # 3
        bi_xy_std=121,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=4,         # 4, 5
    )

    def _job(i):

        name = name_list[i]

        logit_name = args.logits_dir + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_seg']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.infer_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()
        # prob = logit[0]

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        #print(pred.shape)
        if args.save_images:
            imageio.imsave(args.segs_dir + "/" + name + ".png", np.squeeze(pred).astype(np.uint8))
            imageio.imsave(args.segs_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label
    
    
    
    n_jobs = int(os.cpu_count() * 0.9)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds)
    print(format_tabs_multi_metircs([crf_score], ["confusion","precision","recall",'iou'], cat_list=voc.class_list))
    return crf_score


def validate(args=None):

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)

    model = network_du_heads_independent_config_cl(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        aux_layer = -3
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()

    if 'model' in trained_state_dict:
        model_state_dict = trained_state_dict['model']
        for k, v in model_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
    else:
        for k, v in trained_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()

    seg_score = _validate(model=model, data_loader=val_loader, args=args)
    torch.cuda.empty_cache()

    crf_score = crf_proc()
    
    return True

if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints")[0]
    args.logits_dir = os.path.join(base_dir, "segs/logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs/seg_preds", args.infer_set)
    args.segs_rgb_dir = os.path.join(base_dir, "segs/seg_preds_rgb", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    print(args)
    validate(args=args)