import argparse
import datetime
import logging
import os
import random
import sys
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.append(".")

import sys
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc as voc
from model.losses import get_mask_cls_loss,get_masked_ptc_loss, get_seg_loss, get_mask_cls_loss_activation_sort,get_cam_refine_loss,get_pixel_refine_cam_loss

from model.double_seg_head import network_du_heads_independent_config_cl,network_CL,network_mask_cam
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import single_instance_crop_v2,single_bg_fg_crop_denoise_v3,cam_to_label, cam_to_roi_mask2, multi_scale_mask_cam, label_to_aff_mask, refine_cams_with_bkg_v2, get_sam_mask_label
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder", default='./VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='./datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--local_crop_num", default=1, type=int, help="crop_num")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_voc_wseg", type=str, help="work_dir_voc_wseg")

parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=6e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0,0.75,1.25,1.5), help="multi_scales for cam")

parser.add_argument("--w_ptc", default=0.2, type=float, help="w_ptc")
parser.add_argument("--w_ctc", default=0.45, type=float, help="w_ctc")
parser.add_argument("--w_seg", default=0.2, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--uncertain_region_thre", default=0.2 , type=float, help="uncertain_region_thre")
parser.add_argument("--t_b1_mix_cam", default=0.7, type=float, help="t_b1_mix_cam")
parser.add_argument("--t_b2_mix_cam", default=0.5, type=float, help="t_b2_mix_cam")


parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt",default=True, action="store_true", help="save_ckpt")

parser.add_argument("--local-rank", type=int, default=0)
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')


# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5679'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES']='3'

logging.getLogger().setLevel(logging.INFO)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def show_seg(cam,cls_label,low_thre,high_thre,seg_name):
    import matplotlib.pyplot as plt
    # roi_mask_crop = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=low_thre, hig_thre=high_thre)
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    # cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cam #cls_label_rep * 
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label = _pseudo_label-1
    _pseudo_label[_pseudo_label == -1] = -2
    plt.imshow(_pseudo_label.squeeze(0).cpu(), cmap='jet', vmin=-2, vmax=20)
    plt.colorbar()
    plt.title("seg" + seg_name)
    
    plt.savefig('output-image'+ "/" + f'seg' + seg_name + 'png')
    plt.close()

def show_mask_cam(cams_aux,cls_label,low_thre,high_thre, branch_name):
    import matplotlib.pyplot as plt
    roi_mask_crop = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=low_thre, hig_thre=high_thre)
    
    plt.imshow(roi_mask_crop[0].squeeze(0).cpu(), cmap='jet', vmin=-2, vmax=20)
    plt.colorbar()
    plt.title(branch_name + "mask")
    
    plt.savefig(f'cam visual' + branch_name + '.png')
    plt.close()

def validate(model=None, data_loader=None, args=None):

    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label,_ = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _, _ = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_mask_cam(model, inputs=inputs, scales=args.cam_scales,img_name=None,sam_mask=None)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
            
                        
            # import matplotlib.pyplot as plt
            # plt.imshow(cam_label[0].cpu().numpy(), cmap='jet', vmin=0, vmax=20)
            # plt.colorbar()
            # plt.savefig("sam_mask_label")
            # plt.close()
            
            
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))


            # valid_label = torch.nonzero(cls_label[0])[:, 0]
            # out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    cam_aux_score = evaluate.scores(gts, cams_aux)
    model.train()

    tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=voc.class_list)

    return cls_score, tab_results


def validate_sam_mask(model=None, data_loader=None, args=None):

    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label , sam_mask= data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)
            import cv2
            sam_mask = cv2.resize(sam_mask.squeeze(0).numpy(), [args.crop_size, args.crop_size], interpolation=cv2.INTER_NEAREST)
            sam_mask = torch.tensor(sam_mask).unsqueeze(0)
            
            cls, segs, _, _ = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            mask_cls_list, _, fmap, mask_cls_aux_list, cam_12th, cls_token,mask_cam_list,aux_mask_cam_list,_ = model(inputs, "#",crops='#',img_name = None,sam_mask=sam_mask)   
            sam_mask_label = get_sam_mask_label(mask_cam_list=mask_cam_list, cls_label=cls_label,sam_mask=sam_mask, bg_thre=0.72,mask_cls_list=mask_cls_list)
            
            h,w = labels.shape[1], labels.shape[2]
            sam_mask_label = torch.tensor(cv2.resize(sam_mask_label.squeeze(0).numpy(),(w,h),interpolation=cv2.INTER_NEAREST)).unsqueeze(0)

            gt_labels = labels
            # import matplotlib.pyplot as plt
            # plt.imshow(gt_labels[0].cpu().numpy(), cmap='jet', vmin=0, vmax=20)
            # plt.colorbar()
            # plt.savefig("gt_label")
            # plt.close()
            
            # plt.imshow(sam_mask_label[0].cpu().numpy(), cmap='jet', vmin=0, vmax=20)
            # plt.colorbar()
            # plt.savefig("sam_mask_label")
            # plt.close()
    
            # import time
            # time.sleep(2)

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(sam_mask_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(sam_mask_label.cpu().numpy().astype(np.int16))

            # valid_label = torch.nonzero(cls_label[0])[:, 0]
            # out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    cam_aux_score = evaluate.scores(gts, cams_aux)
    model.train()

    tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=voc.class_list)

    return cls_score, tab_results

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend)
    # dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = voc.VOC12SamClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        # resize_range=cfg.dataset.resize_range,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        #shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    model = network_mask_cam(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        init_momentum=args.momentum,
        aux_layer=args.aux_layer,
    )
    
    trained_state_dict = torch.load('/home/zhonggai/python-work-space/ToCo-B_WSSS_work/CVPR_wsss_baseline/jx_vit_base_p16_224-80ecf9dd.pth', map_location="cpu")
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

    model.load_state_dict(state_dict=new_state_dict, strict=False)


#add param ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    param_groups = model.get_param_groups()
    # ccf_param = CPC_loss.get_param_groups()
    # param_groups[2].append(ccf_param)
    model.to(device)

    # cfg.optimizer.learning_rate *= 2
    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)




    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    
    n_crops = 10
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()
    

    for n_iter in range(args.max_iters):

        try:
            img_name, inputs, cls_label, img_box, crops,image_origin,sam_mask = next(train_loader_iter)

        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops,image_origin,sam_mask = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        sam_mask = sam_mask.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

         
        

        sam_mask = sam_mask.to(torch.int32)
        sam_mask[sam_mask==255] = -1
        
               
        # cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
        cams, cams_aux = multi_scale_mask_cam(model, inputs=inputs, scales=args.cam_scales,img_name=None,sam_mask=None)
        # show_mask_cam(cams,cls_label,low_thre=args.low_thre,high_thre=args.high_thre,branch_name='main')


#get pseudo label --------------------------------------------------------------------------------------------------------------------------------
        _28x_pseudo_label='#'
        local_pic = '#'
        
        


# model forward-------------------------------------------------------------------------------------------------------------------------------
        if n_iter< args.warmup_iters:
            cls, segs, fmap, cls_aux, cam_12th, cls_token = model(inputs, _28x_pseudo_label,crops='#',img_name = img_name,sam_mask=None)   

        else:
            mask_cls_list, segs, fmap, mask_cls_aux_list, cam_12th, cls_token,mask_cam_list,aux_mask_cam_list,global_cls_both = model(inputs, _28x_pseudo_label,crops='#',img_name = img_name,sam_mask=sam_mask)   
            cls ,cls_aux = global_cls_both
            

            
        
#--------------------------------------------------------------
        network_sim_loss = torch.tensor(0)
        contrast_loss = torch.tensor(0)
#----------------------------------------------------------------




# cls loss & aux cls loss =================================================================================================================
        cls_pred = torch.zeros_like(cls_label).type(torch.long)
        cls_pred[cls>0] = 1
        
        
        if n_iter< args.warmup_iters:
            cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
            cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)
        else:
            # mask_cls_list = get_sorted_mask_cls_dict(mask_cls_list,cls_label)
            # mask_cls_aux_list = get_sorted_mask_cls_dict(mask_cls_aux_list, cls_label)
            global_cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
            global_cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)
            # print("global: ",global_cls_loss, global_cls_loss_aux)
            # cls_loss = get_mask_cls_loss_activation_sort(mask_cls_list,cls_label)
            # cls_loss_aux = get_mask_cls_loss_activation_sort(mask_cls_aux_list, cls_label)
            # print("mask: ", cls_loss, cls_loss_aux)
            
            cls_loss = 1*global_cls_loss
            cls_loss_aux = 1*global_cls_loss_aux
            
        
        
        # ctc_loss
        ctc_loss = torch.tensor(0)




# seg_loss & reg_loss==========================================================================================================================
        
                
        if n_iter>= args.warmup_iters:
            refined_pseudo_label = sam_mask_label = get_sam_mask_label(mask_cam_list,cls_label,sam_mask,0.7,mask_cls_list=mask_cls_list)
            aux_sam_mask_label = get_sam_mask_label(aux_mask_cam_list,cls_label,sam_mask,0.7,mask_cls_aux_list)
        
        else:
            
            valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
            refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
            # refined_pseudo_label = gt_label.cuda()
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
        # print('seg loss (GT refine): ' , seg_loss)
        
        reg_loss = torch.tensor(0)

# SAM refined CAM & row CAM=========================================================================================================
        # if n_iter>= args.warmup_iters:
        #     cam_12th = F.sigmoid(cam_12th)
            
        #     upper_cam_12th = F.interpolate(cam_12th, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        #     cam_refine_loss = get_cam_refine_loss(upper_cam_12th, refined_pseudo_label)
        # else:
        #     cam_refine_loss = torch.tensor(0)
        # print(cam)

        if n_iter>= args.warmup_iters:
            
            pixel_refine_cam_loss = get_pixel_refine_cam_loss(cam_12th, refined_pseudo_label, False)
            print('prc lossv2: ' , pixel_refine_cam_loss)
        else:
            pixel_refine_cam_loss = torch.tensor(0)
            



# ptc loss=======================================================================================================================================
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # ptc_loss = torch.tensor(0)

        # warmup
        if n_iter <= 2000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.2 * ptc_loss + 0.0 * seg_loss  + 0.1 * pixel_refine_cam_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.2 * ptc_loss + args.w_seg * seg_loss +  0.1 * pixel_refine_cam_loss
        # 如果你增加了 cls_loss 的权重值，使其在整体优化中起到更大的作用，那么模型在训练过程中会更加关注优化 cls_loss
        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
        spacial_bce_loss = torch.tensor(0)
        cpc_loss = torch.tensor(0)

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'aur_loss': ctc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),
            'dcc_loss': cpc_loss.item(),
            'spacial_bce_loss' :spacial_bce_loss.item(),
            'network_sim_loss': network_sim_loss.item(),
            'contrast_loss':contrast_loss.item()
        })

        optim.zero_grad()
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:
            if args.local_rank ==0:
                delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                cur_lr = optim.param_groups[0]['lr']


                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, spacial_bce_loss: %.4f..., network_sim_loss: %.4f... , CL: %.4f... " % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'), avg_meter.pop('spacial_bce_loss'),avg_meter.pop('network_sim_loss'),avg_meter.pop('contrast_loss')))

        if (n_iter + 1) % args.eval_iters == 0:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
            val_cls_score, tab_results = validate_sam_mask(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)
                print(tab_results)
            

    return True


if __name__ == "__main__":

    args = parser.parse_args()

    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank ==0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)

