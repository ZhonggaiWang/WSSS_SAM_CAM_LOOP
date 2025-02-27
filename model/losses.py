import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from collections import OrderedDict
sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.8")
# from bilateralfilter import bilateralfilter, bilateralfilter_batch

def get_spacial_bce_loss(cam,label,per_pic_thre):
    #cam [b,c,h,w] label [b,c] per_pic_thre [b,c+1] cam_flatten [b,c,h*w]
    b,c,h,w = cam.size()
    cam_flatten = cam.view(b, c, -1)
    
    
    
    fg_per_pic_thre = per_pic_thre[:,1:]
    fg_per_pic_thre_t = torch.round(fg_per_pic_thre *(h*w)).to(int).cuda()
    spacial_bce_loss = 0
    for i in range(b):
        fg_channel = cam_flatten.detach()[i][label[i] == 1]
        fg_channel_sorted,fg_channel_sorted_index = torch.sort(fg_channel,dim=-1,descending=True)
        thre_t_idx = fg_per_pic_thre_t[i][label[i] == 1]
        thre_t = fg_channel_sorted[torch.arange(fg_channel.size()[0]),thre_t_idx]
        # thre_t_idx = fg_channel_sorted_index[fg_per_pic_thre_t[i][label[i]==1]]
        # thre_t = 
        thre_t_dim_class = torch.full((c,), 9999.0).cuda()
        thre_t_dim_class[label[i]==1] = thre_t
        # [2,1]->[20]         
        #    [c,h*w]                        [c,h*w]       [c,1]        
        generate_label = torch.where((cam_flatten.detach()[i] >= thre_t_dim_class.unsqueeze(-1)),torch.tensor(1).cuda(),torch.tensor(0).cuda())


#focal_loss——————————————————————————————————————————————————————————————————————————————————————————————————————
        generate_label_class_dim = torch.sum(generate_label,dim=0)
        negetive_sample = torch.where(generate_label_class_dim == 0)
        positive_sample = torch.where(generate_label_class_dim > 0)
        
        _negetive_sample = negetive_sample[0]
        _positive_sample = positive_sample[0]
        
        spacial_bce_loss_neg = F.multilabel_soft_margin_loss(cam_flatten[i, :, _negetive_sample.tolist()], generate_label[:, _negetive_sample.tolist()])
        spacial_bce_loss_pos = F.multilabel_soft_margin_loss(cam_flatten[i, :, _positive_sample.tolist()], generate_label[:, _positive_sample.tolist()])
        
        spacial_bce_loss += 0.1 * spacial_bce_loss_neg + 0.9 * spacial_bce_loss_pos
        
        # print('pos:',spacial_bce_loss_pos,' neg:',spacial_bce_loss_neg)
#origin loss——————————————————————————————————————————————————————————————————————————————————————————————————————
        # spacial_bce_loss += F.multilabel_soft_margin_loss(cam_flatten[i],generate_label)
    # print("spacial_bce_loss:PRINTING IN SPACIAL-BCE BLOCK",spacial_bce_loss/b)
        
    return spacial_bce_loss / b

def get_PABCE_loss(cam,label,per_pic_thre,masks):
    #cam [b,c,h,w] label [b,c] per_pic_thre [b,c+1] cam_flatten [b,c,h*w]
    b,c,h,w = cam.size()
    cam_flatten = cam.view(b, c, -1)
    
    masks_flatten = masks.view(b,-1)
    #de-bg   bg = 0  cls = [1,20]
    masks_flatten = masks_flatten + 1
    masks_flatten[masks_flatten == -1] = 0
     
    
    fg_per_pic_thre = per_pic_thre[:,1:]
    fg_per_pic_thre_t = torch.round(fg_per_pic_thre *(h*w)).to(int).cuda()
    spacial_bce_loss = 0
    for i in range(b):
        current_mask = masks_flatten[i]
        current_cam = cam_flatten[i]
        # [2,1]->[20]         
        #    [c,h*w]                        [c,h*w]       [c,1]        
        generate_label = torch.zeros_like(current_cam)
        generate_label = F.one_hot(current_mask,c+1).permute(1,0)[1:,:]



#focal_loss——————————————————————————————————————————————————————————————————————————————————————————————————————
        generate_label_class_dim = torch.sum(generate_label,dim=0)
        negetive_sample = torch.where(generate_label_class_dim == 0)
        positive_sample = torch.where(generate_label_class_dim > 0)
        
        _negetive_sample = negetive_sample[0]
        _positive_sample = positive_sample[0]
        
        print(_positive_sample)
        spacial_bce_loss_neg = F.multilabel_soft_margin_loss(cam_flatten[i, :, _negetive_sample.tolist()], generate_label[:, _negetive_sample.tolist()])
        spacial_bce_loss_pos = F.multilabel_soft_margin_loss(cam_flatten[i, :, _positive_sample.tolist()], generate_label[:, _positive_sample.tolist()])
        
        print(spacial_bce_loss_pos)
        spacial_bce_loss += 0.0 * spacial_bce_loss_neg + 1.0 * spacial_bce_loss_pos
        
        # print('pos:',spacial_bce_loss_pos,' neg:',spacial_bce_loss_neg)
#origin loss——————————————————————————————————————————————————————————————————————————————————————————————————————
        # spacial_bce_loss += F.multilabel_soft_margin_loss(cam_flatten[i],generate_label)
    # print("spacial_bce_loss:PRINTING IN SPACIAL-BCE BLOCK",spacial_bce_loss/b)
        
    return spacial_bce_loss / b


def get_mask_consistency_loss(cam,flags, box ,masks):
    #cam [b,c,h,w] label [b,c] per_pic_thre [b,c+1] cam_flatten [b,c,h*w]
    #[448,448]
    
    cam = F.interpolate(cam , size= (448,448), mode='bilinear',align_corners=False)
    b,c,h,w = cam.size()
    
    
    cam_flatten = cam.view(b, c, -1)
    masks_flatten = masks.view(b,-1)
    #de-bg   bg = 0  cls = [1,20]

    #-1 uncertain -2 certain bg

    spacial_bce_loss = 0
    for i in range(b):
        current_cls = flags[i] - 1
        current_idx = box[i].view(-1)
        current_mask = masks_flatten[i]
        current_cam = cam_flatten[i]
        # [2,1]->[20]         
        #    [c,h*w]                        [c,h*w]       [c,1]        

        # CAM + PAR result
        # consistency_mask = current_mask.clone().cuda()
        # consistency_mask[current_idx] = current_cls
        
        # PAR result only
        consistency_mask = torch.full_like(current_mask,-1).cuda()

        consistency_mask[current_mask == -1] = -1
        consistency_mask[current_idx] = current_cls 
        
        #denoise 
        denoise_idx1 = current_mask != current_cls 
        denoise_idx2 = current_mask >= 0
        denoise_idx = denoise_idx1 & denoise_idx2
        #other class 
        consistency_mask[denoise_idx] = -1
        consistency_mask[current_mask == -2] = -2
        
        
        # import torchvision.transforms.functional as TF
        # crop_img = consistency_mask.view(h,w)
        # plt.imshow(crop_img.cpu(), cmap='jet', vmin=-2, vmax=20)
        # plt.colorbar()
        
        # plt.savefig('output-image'+ "/" + 'crop_mask' + 'png')
        # plt.close()
        
        # CAM result only
        
        refined_current_idx = consistency_mask == current_cls
        
        generate_label = torch.zeros_like(current_cam)
        generate_label[current_cls,refined_current_idx] = 1 
#focal_loss——————————————————————————————————————————————————————————————————————————————————————————————————————

        negetive_sample = torch.where(consistency_mask == -2)
        positive_sample = torch.where(consistency_mask == current_cls)
        
        _negetive_sample = negetive_sample[0]
        _positive_sample = positive_sample[0]
        
        # print(_positive_sample)
        spacial_bce_loss_neg = F.multilabel_soft_margin_loss(cam_flatten[i, :, _negetive_sample.tolist()], generate_label[:, _negetive_sample.tolist()])
        spacial_bce_loss_pos = F.multilabel_soft_margin_loss(cam_flatten[i, :, _positive_sample.tolist()], generate_label[:, _positive_sample.tolist()])
        
        # print(spacial_bce_loss_pos)
        spacial_bce_loss += 0.1 * spacial_bce_loss_neg + 0.9 * spacial_bce_loss_pos
        
        # print('pos:',spacial_bce_loss_pos,' neg:',spacial_bce_loss_neg)
#origin loss——————————————————————————————————————————————————————————————————————————————————————————————————————
        # spacial_bce_loss += F.multilabel_soft_margin_loss(cam_flatten[i],generate_label)
    # print("spacial_bce_loss:PRINTING IN SPACIAL-BCE BLOCK",spacial_bce_loss/b)
        
    return spacial_bce_loss / b




def get_spacial_bce_loss_focal_bg(cam,label,per_pic_thre):
    #cam [b,c,h,w] label [b,c] per_pic_thre [b,c+1] cam_flatten [b,c,h*w]
    b,c,h,w = cam.size()
    cam_flatten = cam.view(b, c, -1)
    
    
    
    fg_per_pic_thre = per_pic_thre[:,1:]
    fg_per_pic_thre_t = torch.round(fg_per_pic_thre *(h*w)).to(int).cuda()
    spacial_bce_loss = 0
    for i in range(b):
        fg_channel = cam_flatten.detach()[i][label[i] == 1]
        fg_channel_sorted,fg_channel_sorted_index = torch.sort(fg_channel,dim=-1,descending=True)
        thre_t_idx = fg_per_pic_thre_t[i][label[i] == 1]
        thre_t = fg_channel_sorted[torch.arange(fg_channel.size()[0]),thre_t_idx]
        # thre_t_idx = fg_channel_sorted_index[fg_per_pic_thre_t[i][label[i]==1]]
        # thre_t = 
        thre_t_dim_class = torch.full((c,), 9999.0).cuda()
        thre_t_dim_class[label[i]==1] = thre_t
        # [2,1]->[20]         
        #    [c,h*w]                        [c,h*w]       [c,1]        
        generate_label = torch.where((cam_flatten.detach()[i] >= thre_t_dim_class.unsqueeze(-1)),torch.tensor(1).cuda(),torch.tensor(0).cuda())


#focal_loss——————————————————————————————————————————————————————————————————————————————————————————————————————
        generate_label_class_dim = torch.sum(generate_label,dim=0)
        negetive_sample = torch.where(generate_label_class_dim == 0)
        positive_sample = torch.where(generate_label_class_dim > 0)
        
        _negetive_sample = negetive_sample[0]
        _positive_sample = positive_sample[0]
        
        spacial_bce_loss_neg = F.multilabel_soft_margin_loss(cam_flatten[i, :, _negetive_sample.tolist()], generate_label[:, _negetive_sample.tolist()])
        spacial_bce_loss_pos = F.multilabel_soft_margin_loss(cam_flatten[i, :, _positive_sample.tolist()], generate_label[:, _positive_sample.tolist()])
        
        spacial_bce_loss += 0.3 * spacial_bce_loss_neg + 0.7 * spacial_bce_loss_pos
        
        # print('pos:',spacial_bce_loss_pos,' neg:',spacial_bce_loss_neg)
#origin loss——————————————————————————————————————————————————————————————————————————————————————————————————————
        # spacial_bce_loss += F.multilabel_soft_margin_loss(cam_flatten[i],generate_label)
    # print("spacial_bce_loss:PRINTING IN SPACIAL-BCE BLOCK",spacial_bce_loss/b)
        
    return spacial_bce_loss / b    



def get_masked_ptc_loss(inputs, mask):

    b, c, h, w = inputs.shape
    # 2 768 28 28
    inputs = inputs.reshape(b, c, h*w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1,2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5*(1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum()+1)) + 0.5 * torch.sum(neg_mask * inputs_cos) / (neg_mask.sum()+1)
    return loss
#这个相当于是分割网络的seg——loss
def get_seg_loss(pred, label, ignore_index=255):
    #[b,21,h,w]   [b,h,w] 
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_seg_consistence_loss(pred_1, pred_2, ignore_index=255):
    #[b,21,h,w]   [b,h,w] 
    pred_2_label = torch.argmax(F.softmax(pred_2,dim=1),dim=1)
    bg_label = pred_2_label.clone()
    bg_label[pred_2_label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred_1, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = pred_2_label.clone()
    fg_label[pred_2_label==0] = ignore_index
    fg_loss = F.cross_entropy(pred_1, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5

def get_cam_consistence_loss(pred_1, pred_2, ignore_index=255):
    #[b,21,h,w]   [b,h,w] 
    pred_2_label = torch.argmax(F.softmax(pred_2,dim=1),dim=1)
    bg_label = pred_2_label.clone()
    bg_label[pred_2_label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred_1, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = pred_2_label.clone()
    fg_label[pred_2_label==0] = ignore_index
    fg_loss = F.cross_entropy(pred_1, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):

    pred_prob = F.softmax(logit, dim=1)
    crop_mask = torch.zeros_like(pred_prob[:,0,...])

    for idx, coord in enumerate(img_box):
        crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1

    _img = torch.zeros_like(img)
    _img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
    _img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
    _img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()
class CTCLoss_neg(nn.Module):
    def __init__(self, ncrops=10, temp=1.0,):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum
        self.ncrops = ncrops
        # self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))

class CTCLoss_neg(nn.Module):
    def __init__(self, ncrops=10, temp=1.0,):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum
        self.ncrops = ncrops
        # self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))

    def forward(self, local_output, global_output, flags,cls_input):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        b = flags.shape[0]
        cls = 20
        # student_out = student_output.reshape(self.ncrops, b, -1).permute(1,0,2)
        # teacher_out = teacher_output.reshape(2, b, -1).permute(1,0,2)
        cnt = 0
        global_output = global_output.reshape(2,b,cls).permute(1,0,2) # b 1 20 d
        local_output = local_output.reshape(self.ncrops-2,b,cls).permute(1,0,2) # b 2 20
        total_loss = torch.tensor(0.0).cuda()
        for i in range(b):
            for global_idx in range(2):
                total_loss += F.multilabel_soft_margin_loss(global_output[i][global_idx],cls_input[i])
                cnt+=1

        for i in range(b):
            cls_label = torch.nonzero(cls_input[i]==1)
            cls_label = cls_label.tolist()
            cls_label = [index[0] for index in cls_label]
            for crop_idx in range(self.ncrops-2):

                temp_flag = flags[i][crop_idx+2]
                if temp_flag[0] == 1:
                #bg
                    # local_cls =local_output[i][crop_idx]
                    # assert torch.all(temp_flag[1:] == 0), "Not all elements are zero."
                    # total_loss += F.multilabel_soft_margin_loss(local_cls,temp_flag[1:].cuda())
                    continue
                else:
                    #uncertain or fg
                    unique_elements = temp_flag.unique(return_counts=False)
                    if len(unique_elements) == 1:
                    #uncertain
                        continue
                            

                    if len(unique_elements) >= 2:
                        temp_flag_rv = temp_flag[1:].cuda()
                        # indices = torch.nonzero(temp_flag_rv == 1)
                        # indices = indices.tolist()  # 转换为普通的Python列表
                        # indices = [index[0] for index in indices]  # 获取具体数字
                        local_cls =local_output[i][crop_idx]
                        total_loss += F.multilabel_soft_margin_loss(local_cls,temp_flag_rv)
                        cnt+=1
                        
        return total_loss / (cnt)






class CPCLoss(nn.Module):   #softmax + 阈值门版本
    def __init__(self, ncrops=10, temp=1.0,num_cls = 20,num_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.num_cls = num_cls
        self.num_dim = num_dim
        self.feature_contrast = torch.zeros(self.num_cls,self.num_dim)
        self.proj_classifier = nn.Conv2d(in_channels=self.num_dim, out_channels=self.num_cls, kernel_size=1, bias=False,)
        # self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))
    
    def get_param_groups(self):

        return self.proj_classifier.weight
    

    def forward(self, fmap, cam,cls_label,hig_thre,low_thre,bg_thre):
        self.proj_classifier.cuda()
        b, c, h, w = cam.shape
        #pseudo_label = torch.zeros((b,h,w))
        
        cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
        valid_cam = cls_label_rep * cam
        cam_value, _pseudo_label = torch.topk(valid_cam,k=2,dim=1)
        #[b 2 448 448]
        _pseudo_label = _pseudo_label[:, 0, :, :]
        _pseudo_label += 1
        _pseudo_label[cam_value[:, 0, :, :]< hig_thre] = 255
        _pseudo_label[cam_value[:, 0, :, :]< low_thre] = 0
        _pseudo_label[cam_value[:, 0, :, :]< bg_thre] = 0
        _pseudo_label[(cam_value[:, 0, :, :]-cam_value[:, 1, :, :]< 0.3)&(cam_value[:, 0, :, :]>hig_thre)] = 255
        #可能是交界处图片，有两种特征
        # plt.imshow((_pseudo_label[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
        # plt.colorbar()
        # plt.title("cam_feature_refine")
        
        # plt.savefig(f'cam_feature_refine.png')
        # plt.close()
        fmap = F.interpolate(fmap , size= (h,w), mode='bilinear',align_corners=False)
        # cam_grad = F.interpolate(cam_grad , size= (h,w), mode='bilinear',align_corners=False)
        # mask = torch.zeros_like(cam.detach())
        # fmap_cls = self.proj_classifier(fmap.detach())
        # fmap_cls = F.softmax(fmap_cls,dim=-1)
        # fmap_bg = torch.zeros([b,self.num_cls+1,h,w])
        # fmap_bg[:,1:] = fmap_cls
        # fmap_bg[:,0] = 0.25
        # fmap_cls = torch.argmax(fmap_bg,dim=1)
        # plt.imshow((fmap_cls[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
        # plt.colorbar()
        # plt.title("cam_feature_refine")
        
        # plt.savefig(f'cam_feature_refine.png')
        # plt.close()
        
        #B 20 H W
        loss_ccf = 0
        loss_clsifier = 0
        for i in range(b):
            feature_vector = {}
            arg = torch.nonzero(cls_label[i]) + 1
            arg = arg.tolist()
            mutli_arg = []
            for cls in arg:
                cls = cls[0]
                if torch.all(_pseudo_label[i] != cls):
                    mutli_arg.append(cls-1)
                    top_values, top_indices_1d = torch.topk(valid_cam[i,cls-1].view(-1), k=25)  # 获取值最大的三个数及其一维索引
                    indices = torch.vstack((top_indices_1d // 448, top_indices_1d % 448)).T
                    row_indices = indices[:, 0]
                    col_indices = indices[:, 1]
                    sub_fmap = fmap[i, :, row_indices, col_indices]
                    feature_vector[cls-1] = (torch.mean(sub_fmap, dim=-1))
                else:
                    indices = torch.nonzero(_pseudo_label[i] == cls)
                    row_indices = indices[:, 0]
                    col_indices = indices[:, 1]
                    # 使用高级索引提取子张量
                    sub_fmap = fmap[i, :, row_indices, col_indices]
                    feature_vector[cls-1] = (torch.mean(sub_fmap, dim=-1))
            # indices_bg = torch.nonzero(_pseudo_label[i] == 0)
            # bg_row_indices = indices_bg[:, 0]
            # bg_col_indices = indices_bg[:, 1]
            # sub_fmap_bg = fmap[i, :, bg_row_indices, bg_col_indices]
            # feature_bg = torch.mean(sub_fmap_bg, dim=-1)
            
            feature_single_map = torch.zeros(self.num_cls,self.num_dim)
            indentity_matrix = torch.zeros(self.num_cls,self.num_cls)
            # feature_list = list(feature_vector.values())
            # feature_stack = torch.stack(feature_list,dim=0)
            # feature_stack = torch.cat((feature_stack,feature_bg.unsqueeze(0)),dim=0)
            for cls,feature in feature_vector.items():
                #feature = F.normalize(feature,p=2,dim=0)
                feature_single_map[cls][:] = feature
                indentity_matrix[cls][cls] = 1

                
                # expanded_feature = self.feature_contrast[cls].detach().view(self.num_dim, 1, 1).expand(self.num_dim, h, w).cuda()
                # mask[i][cls] = mask[i][cls] = torch.abs(torch.nn.functional.cosine_similarity(expanded_feature, fmap[i].detach(), dim=0))


            # feature_cos_smi = torch.clamp(torch.abs(F.cosine_similarity(feature_stack.unsqueeze(1),feature_stack.unsqueeze(0),dim=-1)),min=1e-5,max=1-1e-5)   
            # feature_indentity_single_map = torch.eye(feature_cos_smi.shape[0],feature_cos_smi.shape[1]).cuda()
            # print(feature_cos_smi)
            # loss += 0.5 * F.binary_cross_entropy(feature_cos_smi,feature_indentity_single_map)
            # mask_value , mask_indices = torch.max(mask,dim = 1)
            # mask_indices[mask_value < hig_thre] = -1
            # # # # # cam_grad = F.sigmoid(cam_grad)
            # # # # # ce_loss = F.cross_entropy(cam_grad,mask.detach())
            # plt.imshow(mask_indices[i].cpu(),cmap='jet', vmin=-1, vmax=20)
            # plt.colorbar()
            # plt.title("cos-smi_mask")
            
            # plt.savefig('cos-smi_mask.png')
            # plt.close()
            feature_single_map_normalized = torch.nn.functional.normalize(feature_single_map, dim=1)
            self.feature_contrast_normalized = torch.nn.functional.normalize(self.feature_contrast, dim=1)

            # 计算余弦相似度矩阵
            cos_similarity_matrix = torch.abs(torch.mm(feature_single_map_normalized, self.feature_contrast_normalized.T))
            # print(cos_similarity_matrix)
            cos_smi_clamp = torch.clamp(cos_similarity_matrix,min=1e-5,max=1-1e-5)
            cls_index = []
            
            #阈值门，当和其他cls的太相似时，不加入到prototype中
            for cls,feature in feature_vector.items():
                # if cls not in mutli_arg:
                row = cos_smi_clamp[cls]
                row_without_cls = torch.cat([row[:cls], row[cls+1:]])
                if torch.all(row_without_cls < 0.6):
                    cls_index.append(cls)
                    pred_label = self.proj_classifier(feature.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
                    gd_label = torch.zeros(self.num_cls).cuda()
                    gd_label[cls] = 1
                    loss_clsifier += F.binary_cross_entropy(F.softmax(pred_label,dim=0),gd_label)     
                # elif torch.all(row_without_cls < row[cls]):
                #     pred_label = F.conv2d((feature.unsqueeze(-1).unsqueeze(-1)),self.proj_classifier.weight).squeeze(-1).squeeze(-1)
                #     gd_label = torch.zeros(self.num_cls).cuda()
                #     gd_label[cls] = 1
                #     loss_clsifier += F.binary_cross_entropy(F.softmax(pred_label,dim=0),gd_label)  
                # elif torch.any(row_without_cls > row[cls]):
                #     true_cls = torch.argmax(row)
                #     pred_label = F.conv2d((feature.unsqueeze(-1).unsqueeze(-1)),self.proj_classifier.weight).squeeze(-1).squeeze(-1)
                #     gd_label = torch.zeros(self.num_cls).cuda()
                #     gd_label[true_cls] = 1
                #     loss_clsifier += F.binary_cross_entropy(F.softmax(pred_label,dim=0),gd_label)  
            loss_ccf += F.binary_cross_entropy(cos_smi_clamp,indentity_matrix.detach())
            self.feature_contrast[cls_index] = 0.95 * self.feature_contrast[cls_index] + 0.05 * feature_single_map[cls_index].detach()
            # print(cos_smi)
            
            
            # pred_label = self.proj_classifier(self.feature_contrast.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            # pred_label = F.softmax(pred_label,dim=-1)
            # gd_label = torch.eye(pred_label.shape[0],pred_label.shape[1]).cuda()
            # loss_clsifier = F.binary_cross_entropy(pred_label,gd_label)
            if len(cls_index):
                loss_clsifier /= len(cls_index)
            # print(loss_ccf,loss_clsifier)
        
        return loss_ccf+loss_clsifier 




class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None
    

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor, recompute_scale_factor=True) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False, recompute_scale_factor=True)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor, recompute_scale_factor=True).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest', recompute_scale_factor=True)
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )
        

class ContrastLoss(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 768 ,buffer_dim = 512):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls * 2
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_lenth ,self.buffer_dim)
        self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, feature_contrast, cls_label):
        #                   [0,20]        [0,19]
        b, _ = cls_label.shape
        tempreture = 1.0
        contrastive_loss = 0
        for i in range(b):
            bg_feature = feature_contrast[i][0]
            feature_fg_contrast = feature_contrast[i][1:]
            cls_index = torch.where(cls_label[i]==1)
            for cls in cls_index[0]:
                cls = cls.item()
                cls_feature = feature_fg_contrast[cls]
                
                buffer_index = 2*cls + 1 #buffer_current_index
                #positive
                
                
                self.is_used_tag[buffer_index][0] = 1 
                self.is_used_tag[buffer_index-1][0] = 1
               
                self.feature_contrast[buffer_index, 0] = cls_feature.detach()
                self.feature_contrast[buffer_index-1, 0] = bg_feature.detach()

                
                
                positive_select_tag = self.is_used_tag[buffer_index] == 1
                
                positive_group = [torch.mean(self.feature_contrast[buffer_index][positive_select_tag],dim = 0)]
                
                negetive_in_buffer = []
                #negetive 现在是所有与cls不同的不管bg or fg都为negative
                for x in range(self.buffer_cls):
                    if x!=buffer_index:
                        negative_select_tag = self.is_used_tag[x] == 1
                        if sum(negative_select_tag)!=0:
                            negetive_in_buffer.append(torch.mean(self.feature_contrast[x][negative_select_tag],dim=0)) 

                    
                
                # negetive_in_pic = [feature_fg_contrast[x].detach() for x in cls_index[0] if x.item()!=cls] + [bg_feature.detach()]
                
                
                negative_group = negetive_in_buffer #+ negetive_in_pic
                
                
                #contrast loss and infoNCE
                # 计算正类pair的相似度
                # positive_similarity = torch.cosine_similarity(cls_feature, torch.stack(positive_group))

                # # 计算负类pair的相似度

                # negative_similarity = torch.cat([torch.cosine_similarity(cls_feature, negative_pair.unsqueeze(0)) for negative_pair in negative_group])
                
                positive_similarity = torch.dot(cls_feature, torch.stack(positive_group).squeeze(0)).unsqueeze(0)
                negative_similarity = torch.stack([torch.dot(cls_feature, negative_pair) for negative_pair in negative_group])
                
                n_positive = len(positive_group)
                n_negative = len(negative_group)

                total_samples = n_positive + n_negative
                labels = torch.zeros(total_samples)
                labels[:n_positive] = 1

                logits = torch.cat([positive_similarity, negative_similarity])

                div = torch.sum(torch.exp(logits/tempreture))
                head = torch.exp(positive_similarity/tempreture)
                prob_logits = torch.div(head , div)
                infoNCE_loss = -(torch.log(prob_logits))
                infoNCE_loss = infoNCE_loss / len(cls_index[0])
               
                # 构建InfoNCE loss
                # targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])
                # logits = torch.abs(torch.cat([positive_similarity, negative_similarity]))
                # loss = F.binary_cross_entropy(logits, targets)
                contrastive_loss  = contrastive_loss + infoNCE_loss
                #logits?
                self.is_used_tag[buffer_index] = torch.roll(self.is_used_tag[buffer_index], shifts=1, dims=0)
                self.is_used_tag[buffer_index-1] = torch.roll(self.is_used_tag[buffer_index-1], shifts=1, dims=0)
                self.feature_contrast[buffer_index] = torch.roll(self.feature_contrast[buffer_index], shifts=1, dims=0)
                self.feature_contrast[buffer_index-1] = torch.roll(self.feature_contrast[buffer_index-1], shifts=1, dims=0)
                

                                


        return contrastive_loss.cuda() / b
    
    
    
class ContrastLoss_wobg(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 256 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_lenth ,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, feature_contrast, cls_label):
        #                   [0,20]        [0,19]
        b, _ = feature_contrast.shape
        tempreture = 1
        contrastive_loss = 0
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            cls = cls_label[i]
            cls_feature = feature_contrast[i]
            
            buffer_index = cls
            #positive
            
            
            # self.is_used_tag[buffer_index][0] = 1         
            self.feature_contrast[buffer_index, 0] = cls_feature.detach()
        
        
            
            positive_group = [torch.mean(self.feature_contrast[buffer_index],dim = 0)]
            
            
            #negetive 现在是所有与cls不同的不管bg or fg都为negative
            
            negetive_in_buffer = [torch.mean(self.feature_contrast[x],dim = 0) for x in range(self.buffer_cls) if x!=buffer_index]              
            
            # negetive_in_pic = [feature_fg_contrast[x].detach() for x in cls_index[0] if x.item()!=cls] + [bg_feature.detach()]
            
            
            negative_group = negetive_in_buffer #+ negetive_in_pic
            
            
            #contrast loss and infoNCE
            # 计算正类pair的相似度
            # positive_similarity = torch.cosine_similarity(cls_feature, torch.stack(positive_group))

            # # 计算负类pair的相似度

            # negative_similarity = torch.cat([torch.cosine_similarity(cls_feature, negative_pair.unsqueeze(0)) for negative_pair in negative_group])
            
            positive_similarity = torch.dot(cls_feature, torch.stack(positive_group).squeeze(0).cuda()).unsqueeze(0)
            negative_similarity = torch.stack([torch.dot(cls_feature, negative_pair.cuda()) for negative_pair in negative_group])
            
            logits = torch.cat([positive_similarity, negative_similarity])

            div = torch.sum(torch.exp(logits/tempreture))
            head = torch.exp(positive_similarity/tempreture)
            prob_logits = torch.div(head , div)
            infoNCE_loss = -(torch.log(prob_logits))
            infoNCE_loss = infoNCE_loss / b
            
            # 构建InfoNCE loss
            # targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])
            # logits = torch.abs(torch.cat([positive_similarity, negative_similarity]))
            # loss = F.binary_cross_entropy(logits, targets)
            contrastive_loss  = contrastive_loss + infoNCE_loss
            #logits?

            self.feature_contrast[buffer_index] = torch.roll(self.feature_contrast[buffer_index], shifts=1, dims=0)

                            


        return contrastive_loss.cuda()
    
    
    
    
    
    
class ContrastLoss_denoise(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_lenth ,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, feature_contrast, cls_label):
        #                   [0,20]        [0,19]
        b, _ = feature_contrast.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls = cls_label[i]
            cls_feature = feature_contrast[i]
            
            buffer_index = cls
            
            #denoise
            cos_sim_in_pool = F.cosine_similarity(cls_feature, self.feature_contrast[buffer_index].cuda(),dim=-1)

            mean_cos_sim = torch.mean(cos_sim_in_pool ,dim=0)
            cos_sim_in_pool[cos_sim_in_pool==0] = 1
            mean_cos_sim_cmp = torch.mean(cos_sim_in_pool ,dim=0)
            # positive_group = [torch.mean(self.feature_contrast[buffer_index],dim = 0)]
            # positive_similarity = torch.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())
            # if positive_similarity == torch.tensor(0):
            #     positive_similarity = 1
            
            cos_tag_in_pool = torch.where(cos_sim_in_pool  >=  mean_cos_sim - 0.1, torch.tensor(1), torch.tensor(0))
            if (sum(cos_tag_in_pool) > 0.7 * self.buffer_lenth) & (mean_cos_sim_cmp > 0.5):

                #只是取决于更不更新缓冲区，而不是进不进行loss损失
                self.feature_contrast[buffer_index, 0] = cls_feature.detach()
                self.feature_contrast[buffer_index] = torch.roll(self.feature_contrast[buffer_index], shifts=1, dims=0)
                update = True
            #loss-function
            
            #positive
            
            positive_group = [torch.mean(self.feature_contrast[buffer_index],dim = 0)]
            
            #negetive 现在是所有与cls不同的不管bg or fg都为negative
            
            negetive_in_buffer = [torch.mean(self.feature_contrast[x],dim = 0) for x in range(self.buffer_cls) if x!=buffer_index]              
            
            # negetive_in_pic = [feature_fg_contrast[x].detach() for x in cls_index[0] if x.item()!=cls] + [bg_feature.detach()]
            
            
            negative_group = negetive_in_buffer #+ negetive_in_pic
            
            
            #contrast loss and infoNCE
            # 计算正类pair的相似度
            positive_similarity = torch.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())

            # 计算负类pair的相似度

            negative_similarity = torch.cat([torch.cosine_similarity(cls_feature, negative_pair.unsqueeze(0).cuda()) for negative_pair in negative_group])
            
            # positive_similarity = F.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())
            # negative_similarity = torch.stack([torch.dot(cls_feature, negative_pair.cuda()) for negative_pair in negative_group])
            
            logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)

            # div = torch.sum(torch.exp(logits/tempreture))
            # head = torch.exp(positive_similarity/tempreture)
            # prob_logits = torch.div(head , div)
            # infoNCE_loss = -(torch.log(prob_logits))            
            # 构建InfoNCE loss
            targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

            print(update , cls , " with ",logits)
            infoNCE_loss = F.binary_cross_entropy(logits, targets)
            if logits[0] < 0.1:
                print('mis-crop refine')
                infoNCE_loss = 0
            
            contrastive_loss  = contrastive_loss + infoNCE_loss

                
        

        return contrastive_loss.cuda() / b
    
    
class ContrastLoss_mixbranch(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_lenth ,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, feature_contrast,another_feature_contrast, cls_label,n_iter):
        #                   [0,20]        [0,19]
        n_iter = min(n_iter , 6000)
        # thre_cos_in_group =0.8 - (n_iter-3000)/3000 * 0.5
        thre_high_posi = (n_iter-3000)/3000 * 0.7
        thre_low_posi = (n_iter-3000)/3000 * 0.3
        
        #能收敛了
        b, _ = feature_contrast.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls = cls_label[i]
            cls_feature = feature_contrast[i]
            
            buffer_index = cls
            
            #positive
            
            positive_group = [another_feature_contrast[i]]
            
            #negetive 现在是所有与cls不同的不管bg or fg都为negative
            
            
            negetive_in_buffer = [torch.mean(self.feature_contrast[x],dim = 0) for x in range(self.buffer_cls) if x!=buffer_index]              
            
            # negetive_in_pic = [feature_fg_contrast[x].detach() for x in cls_index[0] if x.item()!=cls] + [bg_feature.detach()]
            
            negative_group = negetive_in_buffer #+ negetive_in_pic
            
            
            #contrast loss and infoNCE
            # 计算正类pair的相似度
            positive_similarity = torch.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())

            # 计算负类pair的相似度

            negative_similarity = torch.cat([torch.cosine_similarity(cls_feature, negative_pair.unsqueeze(0).cuda()) for negative_pair in negative_group])
            
            # positive_similarity = F.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())
            # negative_similarity = torch.stack([torch.dot(cls_feature, negative_pair.cuda()) for negative_pair in negative_group])
            if (positive_similarity < thre_high_posi):
                pass
                #只是取决于更不更新缓冲区，而不是进不进行loss损失
            else:
                self.feature_contrast[buffer_index, 0] = cls_feature.detach()
                self.feature_contrast[buffer_index] = torch.roll(self.feature_contrast[buffer_index], shifts=1, dims=0)
                update = True
            
            
            logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)

            # div = torch.sum(torch.exp(logits/tempreture))
            # head = torch.exp(positive_similarity/tempreture)
            # prob_logits = torch.div(head , div)
            # infoNCE_loss = -(torch.log(prob_logits))            
            # 构建InfoNCE loss
            targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

            print(update , cls , " with ",logits)
            infoNCE_loss = F.binary_cross_entropy(logits, targets)
            if logits[0] < thre_low_posi:
                print('mis-crop refine')
                infoNCE_loss = 0
            
            contrastive_loss  = contrastive_loss + infoNCE_loss

                
        

        return contrastive_loss.cuda() / b
    
    
class ContrastLoss_classify(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_lenth ,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)
        

    def forward(self, feature_contrast,another_feature_contrast, cls_label,n_iter):
        #                   [0,20]        [0,19]
        n_iter = min(n_iter , 6000)
        # thre_cos_in_group =0.8 - (n_iter-3000)/3000 * 0.5
        thre_high_posi = (n_iter-3000)/3000 * 0.7
        thre_low_posi = (n_iter-3000)/3000 * 0.3
        
        #能收敛了
        b, _ = feature_contrast.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls = cls_label[i]
            cls_feature = feature_contrast[i]
            
            buffer_index = cls
            
            #positive
            
            positive_group = [another_feature_contrast[i]]
            
            #negetive 现在是所有与cls不同的不管bg or fg都为negative
            import random
            random_idx = random.sample(range(0, self.buffer_lenth-1), 20)

            negetive_in_buffer = [self.feature_contrast[x][random_idx[x]].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
            
            # negetive_in_pic = [feature_fg_contrast[x].detach() for x in cls_index[0] if x.item()!=cls] + [bg_feature.detach()]
            
            negative_group = negetive_in_buffer #+ negetive_in_pic
            
            
            #contrast loss and infoNCE
            # 计算正类pair的相似度
            positive_similarity = torch.abs(torch.cosine_similarity(cls_feature, torch.stack(positive_group).cuda()))

            # 计算负类pair的相似度

            negative_similarity = torch.cat([torch.cosine_similarity(cls_feature, negative_pair.unsqueeze(0).cuda()) for negative_pair in negative_group])
            
            # positive_similarity = F.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())
            # negative_similarity = torch.stack([torch.dot(cls_feature, negative_pair.cuda()) for negative_pair in negative_group])
            if (positive_similarity < thre_high_posi):
                pass
                #只是取决于更不更新缓冲区，而不是进不进行loss损失
            else:
                self.feature_contrast[buffer_index, 0] = cls_feature.detach()
                self.feature_contrast[buffer_index] = torch.roll(self.feature_contrast[buffer_index], shifts=1, dims=0)
                update = True
            
            
            logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)

            # div = torch.sum(torch.exp(logits/tempreture))
            # head = torch.exp(positive_similarity/tempreture)
            # prob_logits = torch.div(head , div)
            # infoNCE_loss = -(torch.log(prob_logits))            
            # 构建InfoNCE loss
            targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

            print(update , cls , " with ",logits)
            infoNCE_loss = F.binary_cross_entropy(logits, targets)
            if logits[0] < thre_low_posi:
                print('mis-crop refine')
                infoNCE_loss = 0
            
            contrastive_loss  = contrastive_loss + infoNCE_loss

                
        

        return contrastive_loss.cuda() / b
    

    
    
class ContrastLoss_moco(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_lenth ,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, feature_contrast,another_feature_contrast, cls_label,n_iter):
        #                   [0,20]        [0,19]
        n_iter = min(n_iter , 6000)
        # thre_cos_in_group =0.8 - (n_iter-3000)/3000 * 0.5
        thre_high_posi = (n_iter-3000)/3000 * 0.7
        thre_low_posi = (n_iter-3000)/3000 * 0.3
        
        #能收敛了
        b, _ = feature_contrast.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls = cls_label[i]
            cls_feature = feature_contrast[i]
            
            buffer_index = cls
            
            #positive
            
            positive_group = [another_feature_contrast[i]]
            
            #negetive 现在是所有与cls不同的不管bg or fg都为negative
            import random
            random_idx = random.sample(range(0, self.buffer_lenth-1), 20)

            negetive_in_buffer = [self.feature_contrast[x][random_idx[x]].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
            
            # negetive_in_pic = [feature_fg_contrast[x].detach() for x in cls_index[0] if x.item()!=cls] + [bg_feature.detach()]
            
            negative_group = negetive_in_buffer #+ negetive_in_pic
            
            
            #contrast loss and infoNCE
            # 计算正类pair的相似度
            positive_similarity = torch.abs(torch.cosine_similarity(cls_feature, torch.stack(positive_group).cuda()))

            # 计算负类pair的相似度

            negative_similarity = torch.cat([torch.cosine_similarity(cls_feature, negative_pair.unsqueeze(0).cuda()) for negative_pair in negative_group])
            
            # positive_similarity = F.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())
            # negative_similarity = torch.stack([torch.dot(cls_feature, negative_pair.cuda()) for negative_pair in negative_group])
            if (positive_similarity < thre_high_posi):
                pass
                #只是取决于更不更新缓冲区，而不是进不进行loss损失
            else:
                self.feature_contrast[buffer_index, 0] = cls_feature.detach()
                self.feature_contrast[buffer_index] = torch.roll(self.feature_contrast[buffer_index], shifts=1, dims=0)
                update = True
            
            
            logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)

            # div = torch.sum(torch.exp(logits/tempreture))
            # head = torch.exp(positive_similarity/tempreture)
            # prob_logits = torch.div(head , div)
            # infoNCE_loss = -(torch.log(prob_logits))            
            # 构建InfoNCE loss
            targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

            print(update , cls , " with ",logits)
            infoNCE_loss = F.binary_cross_entropy(logits, targets)
            if logits[0] < thre_low_posi:
                print('mis-crop refine')
                infoNCE_loss = 0
            
            contrastive_loss  = contrastive_loss + infoNCE_loss

                
        

        return contrastive_loss.cuda() / b

class feature_prototype(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, global_cls_token,local_cls_token, cls_label,flags):
        #                   [0,20]        [0,19]

        
        #用来修正global proj的 能收敛了
        b, _ = global_cls_token.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        
        posi_flags = [x-1 for x in flags if x>0]
        tensor_flags = torch.tensor(flags)
        posi_local_cls_token = local_cls_token[tensor_flags>0]
        
        
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls_current = cls_label[i]
            if torch.sum(cls_current == 1) == 1:
                cls = torch.where(cls_current == 1)
                cls = cls[0].item()
                buffer_index  = cls
                current_global_cls_token = global_cls_token[i]

                #update prototype
                
                positive_similarity = F.cosine_similarity(current_global_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
                
                
                # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
                if positive_similarity == 0:
                    self.feature_contrast[buffer_index] =  current_global_cls_token.detach()
                else:
                    self.feature_contrast[buffer_index] = 0.9 * self.feature_contrast[buffer_index].cuda() + 0.1 * abs(positive_similarity.item()) * current_global_cls_token.detach()

                    

                self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
            
                #prototype loss (to seperate different cls features)
                negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
                negetive_item = torch.stack(negetive_in_buffer)
                negative_similarity = F.cosine_similarity(current_global_cls_token, negetive_item)

                logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)
                targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

                print(update , cls , " with ",logits)
                infoNCE_loss = F.binary_cross_entropy(logits, targets)
                contrastive_loss  = contrastive_loss + infoNCE_loss

            
        return contrastive_loss.cuda() / b    




    
    
class ContrastLoss_prototype(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, global_cls_token,local_cls_token, cls_label,flags):
        #                   [0,20]        [0,19]

        
        #能收敛了
        b, _ = global_cls_token.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        
        posi_flags = [x-1 for x in flags if x>0]
        tensor_flags = torch.tensor(flags)
        posi_local_cls_token = local_cls_token[tensor_flags>0]
        posi_flags_tensor = torch.tensor(posi_flags)
        
        
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls_current = cls_label[i]
            if torch.sum(cls_current == 1) == 1:
                cls = torch.where(cls_current == 1)
                cls = cls[0].item()
                buffer_index  = cls
                current_global_cls_token = global_cls_token[i]

                #update prototype
                
                positive_similarity = F.cosine_similarity(current_global_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
                
                
                # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
                if positive_similarity == 0:
                    self.feature_contrast[buffer_index] =  current_global_cls_token.detach()
                else:
                    self.feature_contrast[buffer_index] = 0.9 * self.feature_contrast[buffer_index].cuda() + 0.1 * abs(positive_similarity.item()) * current_global_cls_token.detach()

                    

                self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
            
                #prototype loss (to seperate different cls features)
                negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
                negetive_item = torch.stack(negetive_in_buffer)
                negative_similarity = F.cosine_similarity(current_global_cls_token, negetive_item)

                logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)
                targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

                # pri  t(update , cls , " with ",logits)
                infoNCE_loss = F.binary_cross_entropy(logits, targets)
                contrastive_loss  = contrastive_loss + infoNCE_loss

        local_posi_contrast = [self.feature_contrast[x].cuda() for x in posi_flags]
        local_posi_contrast = torch.stack(local_posi_contrast,dim=0)
        
        #现在不是info nce，可以改成infoNCE 更合理
        # local_posi_logits = torch.clamp(abs(F.cosine_similarity(posi_local_cls_token,local_posi_contrast)),min=1e-5,max=1-1e-5)
        local_posi_logits = torch.clamp((F.cosine_similarity(posi_local_cls_token,local_posi_contrast)),min=1e-5,max=1-1e-5)
        local_posi_targets = torch.ones_like(local_posi_logits)
        local_posi_loss = F.binary_cross_entropy(local_posi_logits,local_posi_targets)
        # print(local_posi_logits, ' ',local_posi_loss)
        contrastive_loss = contrastive_loss + local_posi_loss
            
        return contrastive_loss.cuda() / b    


class ContrastLoss_prototypeV2(nn.Module):   
    def __init__(self,temp=0.5,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, global_cls_token,local_cls_token, cls_label,flags):
        #                   [0,20]        [0,19]

        #能收敛了
        b, _ = global_cls_token.shape
        contrastive_loss = torch.tensor(0)
        cnt = 0
        
        posi_flags = [x-1 for x in flags if x>0]
        tensor_flags = torch.tensor(flags)
        posi_local_cls_token = local_cls_token[tensor_flags>0]

        
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls_current = cls_label[i]
            if torch.sum(cls_current == 1) == 1:
                cls = torch.where(cls_current == 1)
                cls = cls[0].item()
                buffer_index  = cls
                current_global_cls_token = global_cls_token[i]

                #update prototype
                
                positive_similarity = F.cosine_similarity(current_global_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
                
                
                # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
                if positive_similarity == 0:
                    self.feature_contrast[buffer_index] =  current_global_cls_token.detach()
                else:
                    self.feature_contrast[buffer_index] = (1-0.1 * abs(positive_similarity.item())) * self.feature_contrast[buffer_index].cuda() + 0.1 * abs(positive_similarity.item()) * current_global_cls_token.detach()

                    

                self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
            
                #prototype loss (to seperate different cls features)
                negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
                negetive_items = torch.stack(negetive_in_buffer)

                all_items = torch.cat((self.feature_contrast[buffer_index].unsqueeze(0).cuda(),negetive_items),dim=0)
                logits = torch.matmul(current_global_cls_token, all_items.T)
                print( cls , " with ",logits)
                logits = torch.exp(logits / self.temp)
                
                loss = -torch.log(logits[0] / (logits.sum(dim=0) + 1e-4 ))

                contrastive_loss  = contrastive_loss + loss

        #local - global contrast (fg)
        local_posi_contrast = [self.feature_contrast[x].cuda() for x in posi_flags]
        local_posi_contrast = torch.stack(local_posi_contrast,dim=0)
        
        #现在不是info nce，可以改成infoNCE 更合理
        # local_posi_logits = torch.clamp(abs(F.cosine_similarity(posi_local_cls_token,local_posi_contrast)),min=1e-5,max=1-1e-5)
        local_posi_logits = torch.clamp((F.cosine_similarity(posi_local_cls_token,local_posi_contrast)),min=1e-5,max=1-1e-5)
        local_posi_targets = torch.ones_like(local_posi_logits)
        local_posi_loss = F.binary_cross_entropy(local_posi_logits,local_posi_targets)
        # print(local_posi_logits, ' ',local_posi_loss)
        contrastive_loss = contrastive_loss + local_posi_loss
        
  
            
        return contrastive_loss.cuda() / b    

class ContrastLoss_prototypeV3(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, global_cls_token,local_cls_token, cls_label,flags):
        #                   [0,20]        [0,19]

        
        #能收敛了
        b, _ = global_cls_token.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        
        posi_flags = [x-1 for x in flags if x>0]
        tensor_flags = torch.tensor(flags)
        posi_flags_tensor = torch.tensor(posi_flags)
        posi_local_cls_token = local_cls_token[tensor_flags>0]
        bg_local_cls_token = local_cls_token[tensor_flags==0]
        
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls_current = cls_label[i]
            if torch.sum(cls_current == 1) == 1:
                cls = torch.where(cls_current == 1)
                cls = cls[0].item()
                buffer_index  = cls
                current_global_cls_token = global_cls_token[i]

                #update prototype
                
                positive_similarity = F.cosine_similarity(current_global_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
                
                
                # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
                if positive_similarity == 0:
                    self.feature_contrast[buffer_index] =  current_global_cls_token.detach()
                else:
                    self.feature_contrast[buffer_index] = 0.9 * self.feature_contrast[buffer_index].cuda() + 0.1 * abs(positive_similarity.item()) * current_global_cls_token.detach()

                    

                self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
            
                #prototype loss (to seperate different cls features)
                negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
                negetive_item = torch.stack(negetive_in_buffer)
                negative_similarity = F.cosine_similarity(current_global_cls_token, negetive_item)

                logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)
                targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

                print(update , cls , " with ",logits)
                infoNCE_loss = F.binary_cross_entropy(logits, targets)
                contrastive_loss  = contrastive_loss + infoNCE_loss

        #local - global contrast (fg)
        
        num_fg = posi_local_cls_token.shape[0]
        local_posi_contrast = self.feature_contrast.repeat(num_fg,1,1).cuda()
        
        
        #infoNCE
        # local_posi_logits = torch.clamp(abs(F.cosine_similarity(posi_local_cls_token,local_posi_contrast)),min=1e-5,max=1-1e-5)
        local_posi_logits = torch.clamp(abs(F.cosine_similarity(posi_local_cls_token.unsqueeze(1),local_posi_contrast,dim=-1)),min=1e-5,max=1-1e-5)
        local_posi_targets = torch.zeros_like(local_posi_logits)
        local_posi_targets[range(num_fg),posi_flags_tensor.tolist()] = 1
        
        local_posi_loss = F.binary_cross_entropy(local_posi_logits,local_posi_targets)
        # print(local_posi_logits, ' ',local_posi_loss)
        
        print(local_posi_logits[0])
        contrastive_loss = contrastive_loss + local_posi_loss
        
        #local - global contrast (bg)
        # num_bg = bg_local_cls_token.shape[0]
        # local_bg_contrast = self.feature_contrast.repeat(num_bg,1,1).cuda()
        # local_bg_logits = torch.clamp(abs(F.cosine_similarity(bg_local_cls_token.unsqueeze(1),local_bg_contrast,dim=-1)),min=1e-5,max=1-1e-5)
        # local_bg_targets = torch.zeros_like(local_bg_logits)
        # local_bg_loss = F.binary_cross_entropy(local_bg_logits,local_bg_targets)
        # contrastive_loss = contrastive_loss + local_bg_loss
        # print(local_bg_logits[0])
            
        return contrastive_loss.cuda() / b    
    

class ContrastLoss_prototype_instance(nn.Module):   
    def __init__(self,temp=0.5,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, global_cls_token,local_cls_token, cls_label,flags):
        #                   [0,20]        [0,19]

        
        #能收敛了
        b, _ = global_cls_token.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        
        posi_flags = [x-1 for x in flags if x>0]

        for i , flag in enumerate(posi_flags):
            cls = flag
            buffer_index  = cls
            current_instance_cls_token = local_cls_token[i]

            #update prototype
            
            positive_similarity = F.cosine_similarity(current_instance_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
            
            
            # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
            if positive_similarity == 0:
                self.feature_contrast[buffer_index] =  current_instance_cls_token.detach()
            else:
                self.feature_contrast[buffer_index] = (1-0.01 * abs(positive_similarity.item())) * self.feature_contrast[buffer_index].cuda() + 0.01 * abs(positive_similarity.item()) * current_instance_cls_token.detach()

    
            self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
        
            #prototype loss (to seperate different cls features)
            negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
            negetive_item = torch.stack(negetive_in_buffer)
            
            # current_instance_cls_token_norm = F.normalize(current_instance_cls_token,dim=-1,p=2)
            all_items = torch.cat((self.feature_contrast[buffer_index].unsqueeze(0).cuda(),negetive_item),dim=0)
            logits = torch.matmul(current_instance_cls_token, all_items.T)
            # print( cls , " with ",logits)
            logits = torch.exp(logits / self.temp)
            
            loss = -torch.log(logits[0] / (logits.sum(dim=0) + 1e-4 ))

            contrastive_loss  = contrastive_loss + loss
        
        
        

        return contrastive_loss.cuda() / len(flags)


class ContrastLoss_prototype_instance_bg(nn.Module):   
    def __init__(self,temp=0.5,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, global_cls_token,local_cls_token, cls_label,flags):
        #                   [0,20]        [0,19]

        
        #能收敛了
        b, _ = global_cls_token.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        
        posi_flags = [x-1 for x in flags]

        for i , flag in enumerate(posi_flags):
            if flag >=0:
                cls = flag
                buffer_index  = cls
                current_instance_cls_token = local_cls_token[i]

                #update prototype
                
                positive_similarity = F.cosine_similarity(current_instance_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
                
                
                # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
                if positive_similarity == 0:
                    self.feature_contrast[buffer_index] =  current_instance_cls_token.detach()
                else:
                    self.feature_contrast[buffer_index] = (1-0.1 * abs(positive_similarity.item())) * self.feature_contrast[buffer_index].cuda() + 0.1 * abs(positive_similarity.item()) * current_instance_cls_token.detach()

        
                self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
            
                #prototype loss (to seperate different cls features)
                negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
                negetive_item = torch.stack(negetive_in_buffer)
                
                # current_instance_cls_token_norm = F.normalize(current_instance_cls_token,dim=-1,p=2)
                all_items = torch.cat((self.feature_contrast[buffer_index].unsqueeze(0).cuda(),negetive_item),dim=0)
                logits = torch.matmul(current_instance_cls_token, all_items.T)
                # print( cls , " with ",logits)
                logits = torch.exp(logits / self.temp)
                
                loss = -torch.log(logits[0] / logits.sum(dim=0) + 1e-4 )

                contrastive_loss  = contrastive_loss + loss
            else:
                current_instance_cls_token = local_cls_token[i]
                negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls)]              
                negetive_item = torch.stack(negetive_in_buffer)
                logits = torch.matmul(current_instance_cls_token, negetive_item.T)
                logits = torch.exp(logits / self.temp)
                loss = -torch.log(1 / logits.sum(dim=0) + 1e-4 )
                contrastive_loss  = contrastive_loss + loss
        
        

        return contrastive_loss.cuda() / len(flags) 
    
    
class ContrastLoss_prototype_instance_gl(nn.Module):   
    def __init__(self,temp=0.5,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, global_cls_token,local_cls_token, global_masked_cls_token,cls_label,flags):
        #                   [0,20]        [0,19]

        
        #能收敛了
        b, _ = global_cls_token.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        
        posi_flags = [x-1 for x in flags if x>0]

        for i , flag in enumerate(posi_flags):
            cls = flag
            buffer_index  = cls
            current_instance_cls_token = local_cls_token[i]

            #update prototype
            
            positive_similarity = F.cosine_similarity(current_instance_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
            
            
            # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
            if positive_similarity == 0:
                self.feature_contrast[buffer_index] =  current_instance_cls_token.detach()
            else:
                self.feature_contrast[buffer_index] = (1-0.1 * abs(positive_similarity.item())) * self.feature_contrast[buffer_index].cuda() + 0.1 * abs(positive_similarity.item()) * current_instance_cls_token.detach()

    
            self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
        
            #prototype loss (to seperate different cls features)
            negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
            negetive_item = torch.stack(negetive_in_buffer)
            
            # current_instance_cls_token_norm = F.normalize(current_instance_cls_token,dim=-1,p=2)
            all_items = torch.cat((self.feature_contrast[buffer_index].unsqueeze(0).cuda(),negetive_item),dim=0)
            logits = torch.matmul(current_instance_cls_token, all_items.T)
            # print( cls , " with ",logits)
            logits = torch.exp(logits / self.temp)
            
            loss = -torch.log(logits[0] / (logits.sum(dim=0) + 1e-4 ))

            contrastive_loss  = contrastive_loss + loss
        
        global_logits = torch.clamp(abs(F.cosine_similarity(global_cls_token,global_masked_cls_token)),min=1e-5,max=1-1e-5)
        global_targets = torch.ones_like(global_logits)
        contrastive_loss += F.binary_cross_entropy(global_logits,global_targets)
        

        return contrastive_loss.cuda() / b        

    
    
class ContrastLoss_prototype_mosaic(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, global_cls_token,global_mosaic_token,local_cls_token, cls_label,flags,n_iter=0):
        #                   [0,20]        [0,19]

        
        #能收敛了
        b, _ = global_cls_token.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        
        posi_flags = [x-1 for x in flags if x>0]
        tensor_flags = torch.tensor(flags)
        posi_flags_tensor = torch.tensor(posi_flags)
        posi_local_cls_token = local_cls_token[tensor_flags>0]
        bg_local_cls_token = local_cls_token[tensor_flags==0]
        
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls_current = cls_label[i]
            if torch.sum(cls_current == 1) == 1:
                cls = torch.where(cls_current == 1)
                cls = cls[0].item()
                buffer_index  = cls
                current_global_cls_token = global_cls_token[i]

                #update prototype
                
                positive_similarity = F.cosine_similarity(current_global_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
                
                
                # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
                if positive_similarity == 0:
                    self.feature_contrast[buffer_index] =  current_global_cls_token.detach()
                else:
                    self.feature_contrast[buffer_index] = 0.9 * self.feature_contrast[buffer_index].cuda() + 0.1  * current_global_cls_token.detach()

                    

                self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
            
                #prototype loss (to seperate different cls features)
                negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
                negetive_item = torch.stack(negetive_in_buffer)
                negative_similarity = F.cosine_similarity(current_global_cls_token, negetive_item)

                logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)
                targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

                print(update , cls , " with ",logits)
                infoNCE_loss = F.binary_cross_entropy(logits, targets)
                contrastive_loss  = contrastive_loss + infoNCE_loss
            
            else:
                cls = posi_flags[i]
                buffer_index  = cls
                current_global_cls_token = global_mosaic_token[i]

                #update prototype
                
                positive_similarity = F.cosine_similarity(current_global_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
                
                
                # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
                if positive_similarity == 0:
                    self.feature_contrast[buffer_index] =  current_global_cls_token.detach()
                else:
                    self.feature_contrast[buffer_index] = 0.95 * self.feature_contrast[buffer_index].cuda() + 0.05 * current_global_cls_token.detach()

                    

                self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
            
                #prototype loss (to seperate different cls features)
                negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
                negetive_item = torch.stack(negetive_in_buffer)
                negative_similarity = F.cosine_similarity(current_global_cls_token, negetive_item)

                logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)
                targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

                print(update , cls , " with ",logits)
                infoNCE_loss = F.binary_cross_entropy(logits, targets)
                contrastive_loss  = contrastive_loss + infoNCE_loss
            
        #local - global contrast (fg)
        
        num_fg = posi_local_cls_token.shape[0]
        local_posi_contrast = self.feature_contrast.repeat(num_fg,1,1).cuda()
        #infoNCE
        # local_posi_logits = torch.clamp(abs(F.cosine_similarity(posi_local_cls_token,local_posi_contrast)),min=1e-5,max=1-1e-5)
        local_posi_logits = torch.clamp(abs(F.cosine_similarity(posi_local_cls_token.unsqueeze(1),local_posi_contrast,dim=-1)),min=1e-5,max=1-1e-5)
        local_posi_targets = torch.zeros_like(local_posi_logits)
        local_posi_targets[range(num_fg),posi_flags_tensor.tolist()] = 1
        
        local_posi_loss = F.binary_cross_entropy(local_posi_logits,local_posi_targets)
        # print(local_posi_logits, ' ',local_posi_loss)
        
        print(local_posi_logits[0])
        contrastive_loss = contrastive_loss + local_posi_loss
        
            
        return contrastive_loss.cuda() / b  
    
class ContrastLoss_prototype_bg(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, global_cls_token,local_cls_token, cls_label,flags,n_iter):
        #                   [0,20]        [0,19]

        
        #能收敛了
        b, _ = global_cls_token.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        
        posi_flags = [x-1 for x in flags if x>0]
        tensor_flags = torch.tensor(flags)
        posi_flags_tensor = torch.tensor(posi_flags)
        posi_local_cls_token = local_cls_token[tensor_flags>0]
        bg_local_cls_token = local_cls_token[tensor_flags==0]
        
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls_current = cls_label[i]
            if torch.sum(cls_current == 1) == 1:
                cls = torch.where(cls_current == 1)
                cls = cls[0].item()
                buffer_index  = cls
                current_global_cls_token = global_cls_token[i]

                #update prototype
                
                positive_similarity = F.cosine_similarity(current_global_cls_token , self.feature_contrast[buffer_index].unsqueeze(0).cuda())
                
                
                # if (abs(positive_similarity) + 0.05 >= thre_high_posi):
                if positive_similarity == 0:
                    self.feature_contrast[buffer_index] =  current_global_cls_token.detach()
                else:
                    self.feature_contrast[buffer_index] = 0.9 * self.feature_contrast[buffer_index].cuda() + 0.1 * abs(positive_similarity.item()) * current_global_cls_token.detach()

                    

                self.feature_contrast[buffer_index] = F.normalize(self.feature_contrast[buffer_index],dim=-1,p=2)
            
                #prototype loss (to seperate different cls features)
                negetive_in_buffer = [self.feature_contrast[x].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
                negetive_item = torch.stack(negetive_in_buffer)
                negative_similarity = F.cosine_similarity(current_global_cls_token, negetive_item)

                logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)
                targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

                print(update , cls , " with ",logits)
                infoNCE_loss = F.binary_cross_entropy(logits, targets)
                contrastive_loss  = contrastive_loss + infoNCE_loss


        #local - global contrast (bg)
        num_bg = bg_local_cls_token.shape[0]
        local_bg_contrast = self.feature_contrast.repeat(num_bg,1,1).cuda()
        local_bg_logits = torch.clamp(abs(F.cosine_similarity(bg_local_cls_token.unsqueeze(1),local_bg_contrast,dim=-1)),min=1e-5,max=1-1e-5)
        local_bg_targets = torch.zeros_like(local_bg_logits)
        local_bg_loss = F.binary_cross_entropy(local_bg_logits,local_bg_targets)
        contrastive_loss = contrastive_loss + local_bg_loss
        print(local_bg_logits[0])
            
        return contrastive_loss.cuda() / b        


    

class ContrastLoss_single_branch(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 256 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_lenth ,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, feature_contrast_raw, cls_label,n_iter):
        #                   [0,20]        [0,19]
        n_iter = min(n_iter , 6000)
        # thre_cos_in_group =0.8 - (n_iter-3000)/3000 * 0.5
        thre_high_posi = (n_iter-3000)/3000 * 0.7
        thre_low_posi = (n_iter-3000)/3000 * 0.3
        num_list = len(cls_label)
        
        feature_contrast = feature_contrast_raw[:num_list,:]
        another_feature_contrast = feature_contrast_raw[num_list:,:].detach()
        
        #能收敛了
        b, _ = feature_contrast.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls = cls_label[i]
            cls_feature = feature_contrast[i]
            
            buffer_index = cls
            
            #positive
            
            positive_group = [another_feature_contrast[i]]
            
            #negetive 现在是所有与cls不同的不管bg or fg都为negative
            import random
            random_idx = random.sample(range(0, self.buffer_lenth-1), 20)

            negetive_in_buffer = [self.feature_contrast[x][random_idx[x]].cuda() for x in range(self.buffer_cls) if x!=buffer_index]              
            
            # negetive_in_pic = [feature_fg_contrast[x].detach() for x in cls_index[0] if x.item()!=cls] + [bg_feature.detach()]
            
            negative_group = negetive_in_buffer #+ negetive_in_pic
            
            
            #contrast loss and infoNCE
            # 计算正类pair的相似度
            positive_similarity = torch.abs(torch.cosine_similarity(cls_feature, torch.stack(positive_group).cuda()))

            # 计算负类pair的相似度

            negative_similarity = torch.cat([torch.cosine_similarity(cls_feature, negative_pair.unsqueeze(0).cuda()) for negative_pair in negative_group])
            
            # positive_similarity = F.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())
            # negative_similarity = torch.stack([torch.dot(cls_feature, negative_pair.cuda()) for negative_pair in negative_group])
            if (positive_similarity < 0.7):
                pass
                #只是取决于更不更新缓冲区，而不是进不进行loss损失
            else:
                self.feature_contrast[buffer_index, 0] = cls_feature.detach()
                self.feature_contrast[buffer_index] = torch.roll(self.feature_contrast[buffer_index], shifts=1, dims=0)
                update = True
            
            
            logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)

            # div = torch.sum(torch.exp(logits/tempreture))
            # head = torch.exp(positive_similarity/tempreture)
            # prob_logits = torch.div(head , div)
            # infoNCE_loss = -(torch.log(prob_logits))            
            # 构建InfoNCE loss
            targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

            print(update , cls , " with ",logits)
            infoNCE_loss = F.binary_cross_entropy(logits, targets)
            if logits[0] < 0.5:
                print('mis-crop refine')
                infoNCE_loss = 0
            
            contrastive_loss  = contrastive_loss + infoNCE_loss

                
        

        return contrastive_loss.cuda() / b


    

    
    
    
    
    
    
    
    
class ContrastLoss_mixbranch_bug(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 368 ,buffer_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls 
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_lenth ,self.buffer_dim)
        # self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, feature_contrast,another_feature_contrast, cls_label,n_iter):
        #                   [0,20]        [0,19]
        thre =0.8 - (n_iter-3000)/5000 * 0.5
        
        #能收敛了
        b, _ = feature_contrast.shape
        tempreture = 1
        contrastive_loss = torch.tensor(0)
        cnt = 0
        for i in range(b):
            # bg_feature = feature_contrast[i][0]
            update = False
            cls = cls_label[i]
            cls_feature = feature_contrast[i]
            
            buffer_index = cls
            
            #positive
            
            positive_group = [another_feature_contrast[i]]
            
            #negetive 现在是所有与cls不同的不管bg or fg都为negative
            
            negetive_in_buffer = [torch.mean(self.feature_contrast[x],dim = 0) for x in range(self.buffer_cls) if x!=buffer_index]              
            
            # negetive_in_pic = [feature_fg_contrast[x].detach() for x in cls_index[0] if x.item()!=cls] + [bg_feature.detach()]
            
            negative_group = negetive_in_buffer #+ negetive_in_pic
            
            
            #contrast loss and infoNCE
            # 计算正类pair的相似度
            positive_similarity = torch.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())

            # 计算负类pair的相似度

            negative_similarity = torch.cat([torch.cosine_similarity(cls_feature, negative_pair.unsqueeze(0).cuda()) for negative_pair in negative_group])
            
            # positive_similarity = F.cosine_similarity(cls_feature, torch.stack(positive_group).cuda())
            # negative_similarity = torch.stack([torch.dot(cls_feature, negative_pair.cuda()) for negative_pair in negative_group])
            if (torch.any(negative_similarity > thre) | (positive_similarity < 0.5)):
                pass
                #只是取决于更不更新缓冲区，而不是进不进行loss损失
            else:
                self.feature_contrast[buffer_index, 0] = cls_feature.detach()
                self.feature_contrast[buffer_index] = torch.roll(self.feature_contrast[buffer_index], shifts=1, dims=0)
                update = True
            
            
            logits = torch.clamp(torch.abs(torch.cat([positive_similarity, negative_similarity])),min=1e-5,max=1-1e-5)

            # div = torch.sum(torch.exp(logits/tempreture))
            # head = torch.exp(positive_similarity/tempreture)
            # prob_logits = torch.div(head , div)
            # infoNCE_loss = -(torch.log(prob_logits))            
            # 构建InfoNCE loss
            targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])

            print(update , cls , " with ",logits)
            infoNCE_loss = F.binary_cross_entropy(logits, targets)
            if logits[0] < 0.2:
                print('mis-crop refine')
                infoNCE_loss = 0
            
            contrastive_loss  = contrastive_loss + infoNCE_loss

                
        

        return contrastive_loss.cuda() / b
    
    
def get_bg_contrastive_loss(global_cls_token, local_cls_token, flags,num_crop):
    #local cls token dual branch
    
    b ,_ = global_cls_token.shape
    
    flags = torch.tensor(flags)
    local_bg_token = local_cls_token[flags==0]

    contrastive_loss = torch.tensor(0)
    for i in range(b):
        
        current_token = local_bg_token[i*num_crop:(i+1)*num_crop,:]

        loss = -torch.log(torch.clamp(1- torch.mean(abs(F.cosine_similarity(global_cls_token[i],current_token))),torch.tensor(1e-2).cuda(),torch.tensor(1-1e-2).cuda()))
        contrastive_loss = contrastive_loss + loss
    
    return contrastive_loss/b

def get_bg_fg_contrastive_loss(global_cls_token, local_cls_token,num_crop):
    #local_cls_token single branch
    
    b ,_ = global_cls_token.shape               #2b,dim
    contrastive_loss = torch.tensor(0)
    for i in range(b):
        current_token = local_cls_token[i*2*num_crop:2*(i+1)*num_crop,:]
        targets = torch.tensor([1.0,0.0]*num_crop)
        logits = abs(F.cosine_similarity(global_cls_token[i],current_token))
        # print(logits)bn
        loss = F.binary_cross_entropy(logits,targets.cuda())
        contrastive_loss = contrastive_loss + loss
    
    return contrastive_loss/b
        
def get_bg_fg_contrastive_clamp_loss(global_cls_token, local_cls_token,num_crop):
    #local_cls_token single branch
    
    b ,_ = global_cls_token.shape               #2b,dim
    contrastive_loss = torch.tensor(0)
    for i in range(b):
        current_token = local_cls_token[i*2*num_crop:2*(i+1)*num_crop,:]
        targets = torch.tensor([1.0,0.0]*num_crop)
        logits = torch.clamp(abs(F.cosine_similarity(global_cls_token[i],current_token)),torch.tensor(1e-5).cuda(),torch.tensor(1-1e-5).cuda()) 
        # print(logits)
        
        # posi = torch.mean(current_token[targets == 1],dim=0)
        # nege = torch.mean(current_token[targets == 0],dim=0)
        # simi_between_posi_neg = torch.clamp(abs(F.cosine_similarity(posi,nege.unsqueeze(0))),torch.tensor(1e-5).cuda(),torch.tensor(1-1e-5).cuda()) 
        # loss_proj_simi = F.binary_cross_entropy(simi_between_posi_neg,torch.tensor([0.]).cuda())
        # print(loss_proj_simi)
        print(logits)
        loss = F.binary_cross_entropy(logits,targets.cuda())
        contrastive_loss = contrastive_loss + loss
    return contrastive_loss/b


def get_bg_fg_contrastive_clamp_infoNCEloss(global_cls_token, local_cls_token,num_crop,flags):
    #local_cls_token single branch
    temp = 0.5
    b ,_ = global_cls_token.shape               #2b,dim
    contrastive_loss = torch.tensor(0.).cuda()
    for i in range(b):
        current_token = local_cls_token[i*2*num_crop:2*(i+1)*num_crop,:]
        current_flags = flags[i*2*num_crop:2*(i+1)*num_crop]
        current_flags_tensor = torch.tensor(current_flags)
        current_global_token = global_cls_token[i]
        
        logits = torch.matmul(current_global_token,current_token.T)
        logits = torch.exp(logits / temp)
        
        neg_logits = logits[current_flags_tensor == 0]
        for j, flag in enumerate(current_flags):
            if flag > 0:
                contrastive_loss += -torch.log((logits[j] /(logits[j] + neg_logits.sum(dim=0) + 1e-4))+1e-4) 
            
        
    return contrastive_loss/b

def get_bg_fg_explict_class_loss(token_class , flags, num_crop = 0):

    targets_cls = [x-1 for x in flags if x > 0]
    flags = torch.tensor(flags)
    token_fg_class = token_class[flags > 0]
    

    
    n,_ = token_fg_class.shape
    targets = torch.zeros(n, 20)

    for i, cls in enumerate(targets_cls):
        targets[i, cls] = 1
    
    token_fg_class = F.softmax(token_fg_class,dim = -1)
    contrastive_loss = F.binary_cross_entropy(token_fg_class, targets.cuda())

    
    return contrastive_loss

def feature_enhance(fmap , box , roi_mask, flags):

    b , c, h , w = fmap.shape
    num_crop = len(flags) // b
    posi_flags = [x-1 for x in flags]
    tensor_flags = torch.tensor(posi_flags)
    feature_enhance_loss = torch.tensor(0)
    for i in range(b):
        for j in range(num_crop):
            crop_idx = i*num_crop + j
            fmap_idx = i
            cls = posi_flags[crop_idx]
            
            
            # import torchvision.transforms.functional as TF
            # crop_img = roi_mask[i][box[crop_idx][0]:box[crop_idx][1],box[crop_idx][2]:box[crop_idx][3]]
            # plt.imshow(crop_img.cpu(), cmap='jet', vmin=-2, vmax=20)
            # plt.colorbar()
            
            # plt.savefig('output-image'+ "/" + 'crop_mask' + 'png')
            # plt.close()
            
            
            fmap_current = fmap[fmap_idx]
            fmap_mask = roi_mask[fmap_idx] == cls
            if torch.sum(fmap_mask) > 0:
                feature_prototype = torch.mean(fmap_current[:,fmap_mask],dim=-1).detach()

                fmap_crop = fmap[fmap_idx][:,box[crop_idx][0]:box[crop_idx][1],box[crop_idx][2]:box[crop_idx][3]]
                fmap_crop = F.adaptive_avg_pool2d(fmap_crop,(1,1))
                loss = 1-torch.clamp(F.cosine_similarity(fmap_crop.squeeze(-1).squeeze(-1).unsqueeze(0),feature_prototype),min=1e-5,max = 1-1e-5)
                feature_enhance_loss = feature_enhance_loss + loss
    
    return feature_enhance_loss / len(flags)


def get_masked_instance_consistency_loss(global_cls_token, global_masked_cls_token):
    #global_cls_token          patch_global_proj        patch_local_proj
    masked_consistency_loss = torch.tensor(0).cuda()
    global_logits = torch.clamp((F.cosine_similarity(global_cls_token,global_masked_cls_token.detach())),min=1e-5,max=1-1e-5)
    global_targets = torch.ones_like(global_logits)
    masked_consistency_loss =masked_consistency_loss +  F.binary_cross_entropy(global_logits,global_targets)
    return masked_consistency_loss


def get_pixel_refine_cam_loss(cam , gt_label, detach):
    b,h,w = gt_label.size()
    gt_label.cuda()
    cam.cuda()

    cam = F.sigmoid(cam)

    cambgmax = True
    ignore_bg = False
    
    cambg=1-torch.mean(cam,dim=1,keepdim=True) if not cambgmax else 1-torch.max(cam,dim=1,keepdim=True)[0]
    cam_mix = torch.cat([cambg,cam],dim=1)
    cam_mix = F.interpolate(cam_mix,size=(h,w),mode='bilinear',align_corners=False)
    # cam shape B C+1 H 
    # seg seg_label shape B H W

    _, C, _, _ = cam_mix.size()

    prediction = cam_mix.permute(0, 2, 3, 1).contiguous().view(-1, C)  
    target = gt_label.view(-1, 1)  


    loss = 0
    for c in range(C):
        if (ignore_bg & (c==0)):
            continue
        else:
            class_prediction = prediction[:, c].unsqueeze(1) 
            class_target = (target == c).float()

            class_loss = F.binary_cross_entropy(class_prediction, class_target)
            
            loss += class_loss
    loss /= C
    
    return loss

def get_pixel_refine_cam_loss_v2(cam , gt_label, detach):
    b,h,w = gt_label.size()
    _,c,_,_ = cam.size()
    gt_label.cuda()
    cam.cuda()
    # cam=F.relu(cam)
    
    cambgmax = True
    # cambg=1-torch.mean(cam,dim=1,keepdim=True) if not cambgmax else 1-torch.max(cam,dim=1,keepdim=True)[0]
    cam_up = F.interpolate(cam,size=(h,w),mode='bilinear',align_corners=False)
    cam_up_flat = cam_up.view(b,c,-1).cuda()
    
    gt_label_flat = gt_label.view(b,-1)
    gt_valid_mask_flat = (gt_label_flat != 255).cuda()
    gt_label_fg = torch.zeros_like(cam_up_flat).cuda()
    for idx in range(b):
        for i in range(1,c+1):
            gt_label_fg[idx,i-1][gt_label_flat[idx] == i] = 1
    
    
    prc_loss = torch.tensor(0.).cuda()
    for i in range(b):
        prc_loss += F.multilabel_soft_margin_loss(cam_up_flat[i,:,gt_valid_mask_flat[i]],gt_label_fg[i,:,gt_valid_mask_flat[i]])


    return prc_loss / b



        
def get_mask_cls_loss_activation_sort(mask_cls_list , cls_label, multi = False):
    cls_loss = torch.tensor(0.).cuda()
    b = len(mask_cls_list)
    for i in range(b):
        cls_idx = torch.where(cls_label[i]==1)[0]
        current_mask_cls_list = mask_cls_list[i]
        
        if len(current_mask_cls_list)>=2 & list(current_mask_cls_list.items())[0][0] == -1:
            current_mask_cls_list.pop(-1)
        
        loss = torch.tensor(0.).cuda()
        for idx in cls_idx.tolist():
            temp_label = torch.zeros_like(cls_label[i]).to(torch.long)
            temp_label[idx] = 1
            def custom_sort(item):
                return item[1][idx].item()

            sorted_items = sorted(current_mask_cls_list.items(), key=custom_sort, reverse=True)
            sorted_dict = {k: v for k, v in sorted_items}
            mask_idx , cls = list(sorted_dict.items())[0]
            loss += F.multilabel_soft_margin_loss(cls , temp_label)

        loss /= len(cls_idx.tolist())
        cls_loss += loss
    return cls_loss

def get_mask_cls_loss(mask_cls_list , cls_label, multi = False):
    cls_loss = torch.tensor(0.).cuda()
    b = len(mask_cls_list)
    for i in range(b):
        cls_idx = torch.where(cls_label[i]==1)[0]
        current_mask_cls_list = mask_cls_list[i]
        current_mask_cls_list.pop(-1)
        loss = torch.tensor(0.).cuda()
        for idx in cls_idx.tolist():
            loss_dict = {}
            temp_label = torch.zeros_like(cls_label[i]).to(torch.long)
            temp_label[idx] = 1
            for key, value in current_mask_cls_list.items():
                temp_loss = F.multilabel_soft_margin_loss(value, temp_label)
                loss_dict[key] = temp_loss
            sorted_loss_dict = OrderedDict(sorted(loss_dict.items(), key=lambda x: x[1]))
            loss += list(sorted_loss_dict.items())[0][1]
            #匈牙利匹配 应该是可以用的,这里的排序完全可以排loss最小的序
        loss /= len(cls_idx.tolist())
        cls_loss += loss
    return cls_loss

def get_cam_refine_loss(cam_12th, pseudo_label):
    b,c,h,w = cam_12th.shape
    max_vals, _ = torch.max(cam_12th, dim=1, keepdim=True) #[b,1,h,w]
    bg_channel = 1 - max_vals
    cam_with_bg = torch.cat([bg_channel,cam_12th], dim=1)
    
    # import imageio
    # from utils import imutils
    # seg_pred = torch.argmax(cam_with_bg, dim=1).cpu().numpy().astype(np.int16)
    # import imageio
    # imageio.imsave("seg.png", imutils.encode_cmap(np.squeeze(seg_pred)).astype(np.uint8))
    
    
    cam_refine_loss = get_seg_loss(cam_with_bg, pseudo_label, 255)
    return cam_refine_loss