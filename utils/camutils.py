import pdb
import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random
import cv2
#448尺度下
def cam_to_label(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value<=bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=high_thre] = ignore_index
        _pseudo_label[cam_value<=low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return valid_cam, pseudo_label

def random_with_probability(p):
    if random.randrange(0, 100) < p * 100:
        return 0
    else:
        return 255


def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    #训练过程中
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    #(b,c) -> (b, c, 1, 1) -> (b,c,h,w)
    valid_cam = cls_label_rep * cam + cls_label_rep * 1e-8
    #表明哪些需要验证
    cam_value, cam_indecies = valid_cam.max(dim=1, keepdim=False)
    #b h w
        # _pseudo_label += 1
    roi_mask = torch.clone(cam_indecies)
    # for b in range(valid_cam.shape[0]):
    #     for channel in range(valid_cam.shape[1]):
    #         print_cam = valid_cam[b][channel].cpu()
            
    #         if torch.sum(print_cam)> 1e-2:
    #             plt.imshow(print_cam, cmap='jet')
    #             plt.colorbar()
    #             plt.title(f'Activation Map - Channel {channel}')
    #             plt.savefig(f'activation_map_{b}_{channel}.png')
    #             plt.close()
    
    roi_mask[cam_value<=hig_thre] = -1
    roi_mask[cam_value<=low_thre] = -2    #roi区域（uncertainty区域mask设置为1,bg为0，物体区为2）
    
    
    return roi_mask

def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    #训练过程中
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    #(b,c) -> (b, c, 1, 1) -> (b,c,h,w)
    valid_cam = cls_label_rep * cam + cls_label_rep * 1e-8
    #只有fg了现在是
    cam_value, cam_indecies = valid_cam.max(dim=1, keepdim=False)
    #b h w
        # _pseudo_label += 1
    roi_mask = torch.clone(cam_indecies)

    
    roi_mask[cam_value<=hig_thre] = -1
    roi_mask[cam_value<=low_thre] = -2    #roi区域（uncertainty区域mask设置为1,bg为0，物体区为2）
    
    
    return roi_mask

def cam_to_roi_mask_uncertain_assign(cam, cls_label, hig_thre=None, low_thre=None):
    #训练过程中
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    #(b,c) -> (b, c, 1, 1) -> (b,c,h,w)
    valid_cam = cls_label_rep * cam + cls_label_rep * 1e-8
    #只有fg了现在是
    cam_value, cam_indecies = valid_cam.max(dim=1, keepdim=False)
    #b h w
        # _pseudo_label += 1
    roi_mask = torch.clone(cam_indecies)

    

    roi_mask[cam_value<=low_thre] = -2    #roi区域（uncertainty区域mask设置为1,bg为0，物体区为2）
    
    
    return roi_mask


def get_valid_cam(cam, cls_label):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam

    return valid_cam

def ignore_img_box(label, img_box, ignore_index):

    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label

def single_class_crop(images,cls_label = None, roi_mask=None, crop_num=1, crop_size=64):
    #image 2 3 448 448
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    for i in range(b):
        cls_index = torch.where(cls_label[i] == 1)
        current_mask = roi_mask[i]
        for cls in cls_index[0]:
            cls = cls.item()
            fg_index = torch.where(current_mask == cls+1)
            if len(fg_index[0]):
                random_index = random.randint(0, fg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = fg_index[0][random_index], fg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x*16, crop_index_y*16
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin


                
                
            else:
                random_index_x = random.randint(0, h - 1)
                random_index_y = random.randint(0, h - 1)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            box.append((x_start,x_end,y_start,y_end))
            crops.append(padded_image[i][:,x_start:x_end,y_start:y_end])
            flags.append(cls)


    return crops, flags, box



def single_bg_crop(images,cls_label = None, roi_mask=None, crop_num=1, crop_size=64):
    #image 2 3 448 448
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = [0]
    for i in range(b):

        for j in range(crop_num):
            current_mask = roi_mask[i]
            
            #current_mask[current_mask==255] = 0
            bg_index = torch.where(current_mask == 0)

            if len(bg_index[0]):
                random_index = random.randint(0, bg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = bg_index[0][random_index], bg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x*16, crop_index_y*16
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin


                
                
            else:
                random_index_x = random.randint(0, h - 1)
                random_index_y = random.randint(0, h - 1)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            box.append((x_start,x_end,y_start,y_end))
            crops.append(padded_image[i][:,x_start:x_end,y_start:y_end])


    return crops, flags, box

def single_bg_fg_crop(images,cls_label = None, roi_mask=None, crop_num=1, crop_size=64):
    #image 2 3 448 448
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    for i in range(b):
        for j in range(crop_num):
            current_mask = roi_mask[i]
            cls_index = torch.where(cls_label[i] == 1)
            cls = random.sample(cls_index[0].tolist(),1)[0]
            
            fg_index = torch.where(current_mask == cls+1)
            bg_index = torch.where(current_mask == 0)
            
            
            if len(fg_index[0]):
                random_index = random.randint(0, fg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = fg_index[0][random_index], fg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x*16, crop_index_y*16
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            else:
                random_index_x = random.randint(0, h - 1)
                random_index_y = random.randint(0, h - 1)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            box.append((x_start,x_end,y_start,y_end))
            crops.append(padded_image[i][:,x_start:x_end,y_start:y_end])
            flags.append(cls+1)
            
            
            # denoise strategy is important to improve performance
            if len(bg_index[0]):
                random_index = random.randint(0, bg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = bg_index[0][random_index], bg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x*16, crop_index_y*16
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            else:
                random_index_x = random.randint(0, h - 1)
                random_index_y = random.randint(0, h - 1)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            
            box.append((x_start,x_end,y_start,y_end))
            crops.append(padded_image[i][:,x_start:x_end,y_start:y_end])
            flags.append(0)

    return crops, flags, box

def single_bg_fg_crop_denoise(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, roi_mask= None):
    #image 2 3 448 448
    #roi mask 28 28
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    denoise_thre = 0.35
    
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    
    for i in range(b):
        for j in range(crop_num):
            current_mask = roi_mask_28x[i]
            cls_index = torch.where(cls_label[i] == 1)
            cls = random.sample(cls_index[0].tolist(),1)[0]
            
            fg_index = torch.where(current_mask == cls+1)
            bg_index = torch.where(current_mask == 0)
            
            uncertain_remove = roi_mask[i].clone()
            for_denoise_mask = torch.where(uncertain_remove>=0, torch.tensor(1).cuda(),torch.tensor(0).cuda()) 
            padded_denoise_mask = F.pad(for_denoise_mask, padding, mode='constant', value=0)
            
            if len(fg_index[0]):
                random_index = random.randint(0, fg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = fg_index[0][random_index], fg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x*16, crop_index_y*16
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            else:
                random_index_x = random.randint(0, h - 1)
                random_index_y = random.randint(0, h - 1)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            box.append((x_start,x_end,y_start,y_end))
            crops.append(padded_image[i][:,x_start:x_end,y_start:y_end])
            flags.append(cls+1)
            
            
            # denoise strategy is important to maintain stability
            if len(bg_index[0]):
                random_index = random.randint(0, bg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = bg_index[0][random_index], bg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x*16, crop_index_y*16
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            else:
                random_index_x = random.randint(0, h - 1)
                random_index_y = random.randint(0, h - 1)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            crop_mask = padded_denoise_mask[x_start:x_end, y_start:y_end]
            
            # plt.imshow(crop_mask.cpu(), cmap='jet', vmin=0, vmax=1)
            # plt.colorbar()
            # plt.title("crop" + "mask")
            
            # plt.savefig('output-image'+ "/" +f'mask' + str(j) + '.png')
            # plt.close()
            
            posi_count = torch.sum(crop_mask == 1)
            if (posi_count / (crop_size*crop_size)) > denoise_thre:
                crops.append(torch.zeros((3,crop_size,crop_size)))
                box.append((x_start,x_end,y_start,y_end))
                flags.append(0)
            else:
                box.append((x_start,x_end,y_start,y_end))
                crops.append(padded_image[i][:,x_start:x_end,y_start:y_end])
                flags.append(0)
                        
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')



    return crops, flags, box

def single_bg_fg_crop_denoise_v2(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, roi_mask= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    denoise_thre = 0.15
    
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        for j in range(crop_num):
            current_mask = roi_mask_28x[i]
            cls_index = torch.where(cls_label[i] == 1)
            cls = random.sample(cls_index[0].tolist(),1)[0]
            
            fg_index = torch.where(current_mask == cls)
            bg_index = torch.where(current_mask == -2)
            
            uncertain_remove = roi_mask[i].clone()
            #roi mask 448 * 448
            for_denoise_mask = torch.where(uncertain_remove>=0, torch.tensor(1).cuda(),torch.tensor(0).cuda()) 
            padded_denoise_mask = F.pad(for_denoise_mask, padding, mode='constant', value=0)
            
            if len(fg_index[0]):
                random_index = random.randint(0, fg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = fg_index[0][random_index], fg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x, crop_index_y
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            else:
                random_index_x = random.randint(0, h - 1)
                random_index_y = random.randint(0, h - 1)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            box.append((x_start,x_end,y_start,y_end))
            crops.append(padded_image[i][:,x_start:x_end,y_start:y_end])
            flags.append(cls+1)
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')

            
            # denoise strategy is important to maintain stability
            if len(bg_index[0]):
                random_index = random.randint(0, bg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = bg_index[0][random_index], bg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x, crop_index_y
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            else:
                random_index_x = random.randint(0, h - 1)
                random_index_y = random.randint(0, h - 1)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            crop_mask = padded_denoise_mask[x_start:x_end, y_start:y_end]
            
            # plt.imshow(crop_mask.cpu(), cmap='jet', vmin=0, vmax=1)
            # plt.colorbar()
            # plt.title("crop" + "mask")
            
            # plt.savefig('output-image'+ "/" +f'mask' + str(j) + '.png')
            # plt.close()
            
            posi_count = torch.sum(crop_mask == 1)
            if (posi_count / (crop_size*crop_size)) > denoise_thre:
                crops.append(torch.zeros((3,crop_size,crop_size)))
                box.append((x_start,x_end,y_start,y_end))
                flags.append(0)
            else:
                box.append((x_start,x_end,y_start,y_end))
                crops.append(padded_image[i][:,x_start:x_end,y_start:y_end])
                flags.append(0)
                        
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')



    return crops, flags, box

def single_bg_fg_crop_denoise_v3(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, roi_mask= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    denoise_thre = 0.25
    
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        for j in range(crop_num):
            current_mask = roi_mask_28x[i][margin:h-margin, margin:w-margin]
            cls_index = torch.where(cls_label[i] == 1)
            cls = random.sample(cls_index[0].tolist(),1)[0]
            
            fg_index = torch.where(current_mask == cls)
            bg_index = torch.where(current_mask == -2)
            
            uncertain_remove = roi_mask[i].clone()
            #roi mask 448 * 448
            for_denoise_mask = torch.where(uncertain_remove>=0, torch.tensor(1).cuda(),torch.tensor(0).cuda()) 
    
            
            if len(fg_index[0]):
                random_index = random.randint(0, fg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = fg_index[0][random_index], fg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x, crop_index_y
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            else:
                random_index_x = random.randint(margin, h - margin)
                random_index_y = random.randint(margin, h - margin)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2

            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            box.append((x_start,x_end,y_start,y_end))
            crops.append(images[i][:,x_start:x_end,y_start:y_end])
            flags.append(cls+1)
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')

            
            # denoise strategy is important to maintain stability
            if len(bg_index[0]):
                random_index = random.randint(0, bg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = bg_index[0][random_index], bg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x, crop_index_y
                x_start = select_index_x - crop_size // 2
                x_end = select_index_x + crop_size // 2
                y_start = select_index_y - crop_size // 2
                y_end = select_index_y + crop_size // 2
                x_start += margin
                x_end += margin
                y_start += margin
                y_end += margin
            else:
                random_index_x = random.randint(margin, h - margin)
                random_index_y = random.randint(margin, h - margin)
                x_start = random_index_x - crop_size // 2
                x_end = random_index_x + crop_size // 2
                y_start = random_index_y - crop_size // 2
                y_end = random_index_y + crop_size // 2
  
            
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            crop_mask = for_denoise_mask[x_start:x_end, y_start:y_end]
            
            # plt.imshow(crop_mask.cpu(), cmap='jet', vmin=0, vmax=1)
            # plt.colorbar()
            # plt.title("crop" + "mask")
            # plt.savefig('output-image'+ "/" +f'mask' + str(j) + '.png')
            # plt.close()
            
            posi_count = torch.sum(crop_mask == 1)
            if (posi_count / (crop_size*crop_size)) > denoise_thre:
                crops.append(torch.zeros((3,crop_size,crop_size)))
                box.append((x_start,x_end,y_start,y_end))
                flags.append(0)
            else:
                box.append((x_start,x_end,y_start,y_end))
                crops.append(images[i][:,x_start:x_end,y_start:y_end])
                flags.append(0)
                        
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')



    return crops, flags, box


def single_instance_crop(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, roi_mask= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop s

    for i in range(b):
        for j in range(crop_num):
            current_mask = roi_mask[i]
            cls_index = torch.unique(current_mask)
            cls_index = cls_index.tolist()
            cls_index = [x for x in cls_index if x>=0]
            if len(cls_index):
                cls = random.sample(cls_index,1)[0]
            else:
                cls_index = torch.where(cls_label[i] == 1)
                cls = random.sample(cls_index[0].tolist(),1)[0]
            
            crop_idx = current_mask == cls 
            crop_uncertain = current_mask == -1
            crop_idx = crop_idx | crop_uncertain
            crop_image = torch.zeros((3,h,w)).cuda()
            
            crop_image[:,crop_idx] = images[i][:,crop_idx]
            
            #crop_image = F.interpolate(crop_image.detach(), size=(int(h/4),int(w/4)), mode="bilinear", align_corners=False)
            
            crops.append(crop_image)
            flags.append(cls+1)
        
            crop_img = crops[-1]
            crop_img = TF.to_pil_image(crop_img)
            crop_img.save('crop_img_in_block_fg_crop.png')




    return crops, flags, box

def single_instance_crop_abl(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, refine_pesudo_label= None,boundary_box= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    refine_pesudo_label = refine_pesudo_label.float()
    refine_pesudo_label[refine_pesudo_label == 255] = -2
    refine_pesudo_label[refine_pesudo_label == 0] = -2
        
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        
        hs,he,ws,we = boundary_box[i]
        
        for j in range(crop_num):
            current_mask = refine_pesudo_label[i]
            cls_index = torch.unique(current_mask)
            cls_index = cls_index.tolist()
            cls_index = [x for x in cls_index if x>=0]
            if len(cls_index):
                cls = random.sample(cls_index,1)[0]
            else:
                cls_index = torch.where(cls_label[i] == 1)
                cls = random.sample(cls_index[0].tolist(),1)[0]
            
            crop_idx = current_mask == cls 
            crop_uncertain = current_mask == -1
            _crop_idx = crop_idx | crop_uncertain
            
            
            
            crop_idx = torch.zeros((h,w),dtype=torch.bool).cuda()
            crop_idx[hs:he,ws:we] = _crop_idx[hs:he,ws:we]
            
            
            
            crop_image = torch.zeros((3,h,w)).cuda()
            
            crop_image[:,crop_idx] = images[i][:,crop_idx]
            
            #for coco
            # crop_image = F.interpolate(crop_image.unsqueeze(0).detach(), size=(int(h/2),int(w/2)), mode="bilinear", align_corners=False)
            # crop_image = crop_image.squeeze(0)
            
            crops.append(crop_image)
            flags.append(int(cls))

            box.append(crop_idx)
            
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')


    return crops, flags, box

def single_instance_crop_v2(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, refine_pesudo_label= None,boundary_box= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    refine_pesudo_label[refine_pesudo_label == 255] = -1
    refine_pesudo_label[refine_pesudo_label == 0] = -2
        
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        
        hs,he,ws,we = boundary_box[i]
        
        for j in range(crop_num):
            current_mask = refine_pesudo_label[i]
            cls_index = torch.unique(current_mask)
            cls_index = cls_index.tolist()
            cls_index = [x for x in cls_index if x>=0]
            if len(cls_index):
                cls = random.sample(cls_index,1)[0]
            else:
                cls_index = torch.where(cls_label[i] == 1)
                cls = random.sample(cls_index[0].tolist(),1)[0]
            
            crop_idx = current_mask == cls 
            crop_uncertain = current_mask == -1
            _crop_idx = crop_idx | crop_uncertain
            
            
            
            crop_idx = torch.zeros((h,w),dtype=torch.bool).cuda()
            crop_idx[hs:he,ws:we] = _crop_idx[hs:he,ws:we]
            
            
            
            crop_image = torch.zeros((3,h,w)).cuda()
            
            crop_image[:,crop_idx] = images[i][:,crop_idx]
            
            #for coco
            # crop_image = F.interpolate(crop_image.unsqueeze(0).detach(), size=(int(h/2),int(w/2)), mode="bilinear", align_corners=False)
            # crop_image = crop_image.squeeze(0)
            
            crops.append(crop_image)
            flags.append(int(cls))

            box.append(crop_idx)
            
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')


    return crops, flags, box


def single_instance_crop_cam(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, refine_pesudo_label= None,boundary_box= None,label = None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    refine_pesudo_label[refine_pesudo_label == 255] = -1
    refine_pesudo_label[refine_pesudo_label == 0] = -2
        
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        
        hs,he,ws,we = boundary_box[i]
        
        for j in range(crop_num):
            current_mask = refine_pesudo_label[i]
            cls_index = torch.unique(current_mask)
            cls_index = cls_index.tolist()
            cls_index = [x for x in cls_index if x>=0]
            if len(cls_index):
                cls = random.sample(cls_index,1)[0]
            else:
                cls_index = torch.where(cls_label[i] == 1)
                cls = random.sample(cls_index[0].tolist(),1)[0]
            if label != None:
                cls = label 
            
            crop_idx = current_mask == cls 
            crop_uncertain = current_mask == -1
            _crop_idx = crop_idx | crop_uncertain
            
            
            
            crop_idx = torch.zeros((h,w),dtype=torch.bool).cuda()
            crop_idx[hs:he,ws:we] = _crop_idx[hs:he,ws:we]
            
            
            
            crop_image = torch.zeros((3,h,w)).cuda()
            
            crop_image[:,crop_idx] = images[i][:,crop_idx]
            
            #for coco
            # crop_image = F.interpolate(crop_image.unsqueeze(0).detach(), size=(int(h/2),int(w/2)), mode="bilinear", align_corners=False)
            # crop_image = crop_image.squeeze(0)
            
            crops.append(crop_image)
            flags.append(int(cls))

            box.append(crop_idx)
            
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')


    return crops, flags, box


def single_instance_crop_denoise(images,cls_label = None, roi_mask_uncertain_assign=None, crop_num=1, crop_size=64, refine_pesudo_label= None,boundary_box= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    
    '''
    use CAM uncertain assign to denoise
    obtaining pure one class mask
    '''
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    refine_pesudo_label[refine_pesudo_label == 255] = -1
    refine_pesudo_label[refine_pesudo_label == 0] = -2
        
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        
        hs,he,ws,we = boundary_box[i]
        
        for j in range(crop_num):
            current_mask = refine_pesudo_label[i]
            current_mask_uncertain = roi_mask_uncertain_assign[i]
            cls_index = torch.unique(current_mask)
            cls_index = cls_index.tolist()
            cls_index = [x for x in cls_index if x>=0]
            if len(cls_index):
                cls = random.sample(cls_index,1)[0]
            else:
                cls_index = torch.where(cls_label[i] == 1)
                cls = random.sample(cls_index[0].tolist(),1)[0]
            
            cls = int(cls)
            crop_idx = current_mask == cls 
            crop_uncertain = current_mask == -1
            _crop_idx = crop_idx | crop_uncertain
            
            crop_idx = torch.zeros((h,w),dtype=torch.bool).cuda()
            crop_idx[hs:he,ws:we] = _crop_idx[hs:he,ws:we]
            
            
            
            consistency_mask = torch.full_like(current_mask,-1).cuda()
            # consistency_mask[current_mask == -1] = -1
            consistency_mask[crop_idx] = cls-1
            
            #denoise other fg cls
            denoise_idx1 = current_mask_uncertain != (cls-1)
            denoise_idx2 = current_mask_uncertain >= 0
            denoise_idx = denoise_idx1 & denoise_idx2
            #其他前景类别周边的uncertain区域，通过这个操作被赋值为-1，只取cls作为前景即可。得到pure的某一前景类
            consistency_mask[denoise_idx] = -1
            consistency_mask[current_mask == -2] = -2
            crop_idx = consistency_mask == (cls-1)
        
            
            
            crop_image = torch.zeros((3,h,w)).cuda()
            
            crop_image[:,crop_idx] = images[i][:,crop_idx]
            
            crop_image = F.interpolate(crop_image.unsqueeze(0).detach(), size=(int(h/2),int(w/2)), mode="bilinear", align_corners=False)
            crop_image = crop_image.squeeze(0)
            
            crops.append(crop_image)
            flags.append(int(cls))

            box.append(crop_idx)
            
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')


    return crops, flags, box

def single_instance_crop_bg(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, refine_pesudo_label= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    refine_pesudo_label[refine_pesudo_label == 255] = -1
    refine_pesudo_label[refine_pesudo_label == 0] = -2
        
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        for j in range(crop_num):
            current_mask = refine_pesudo_label[i]
            cls_index = torch.unique(current_mask)
            cls_index = cls_index.tolist()
            cls_index = [x for x in cls_index if x>=0]
            if len(cls_index):
                cls = random.sample(cls_index,1)[0]
            else:
                cls_index = torch.where(cls_label[i] == 1)
                cls = random.sample(cls_index[0].tolist(),1)[0]
            
            crop_idx = current_mask == cls 
            crop_uncertain = current_mask == -1
            crop_idx = crop_idx | crop_uncertain
            crop_image = torch.zeros((3,h,w)).cuda()
            
            crop_image[:,crop_idx] = images[i][:,crop_idx]
            
            crop_image = F.interpolate(crop_image.unsqueeze(0).detach(), size=(int(h/2),int(w/2)), mode="bilinear", align_corners=False)
            crop_image = crop_image.squeeze(0)
            
            crops.append(crop_image)
            flags.append(int(cls))
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')
            
            
            crop_bg_idx = current_mask == -2
            crop_bg_image = torch.zeros((3,h,w)).cuda()
            
            crop_bg_image[:,crop_bg_idx] = images[i][:,crop_bg_idx]
            
            crop_bg_image = F.interpolate(crop_bg_image.unsqueeze(0).detach(), size=(int(h/2),int(w/2)), mode="bilinear", align_corners=False)
            crop_bg_image = crop_bg_image.squeeze(0)
            
            
            crops.append(crop_bg_image)
            flags.append(0)
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')

    return crops, flags, box


def single_instance_crop_all_cls(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, refine_pesudo_label= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    refine_pesudo_label[refine_pesudo_label == 255] = -1
    refine_pesudo_label[refine_pesudo_label == 0] = -2
        
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        current_mask = refine_pesudo_label[i]
        cls_index = torch.unique(current_mask)
        cls_index = cls_index.tolist()
        cls_index = [x for x in cls_index if x>=0]
        if len(cls_index):
            cls_index = cls_index
        else:
            cls_index = torch.where(cls_label[i] == 1)
            cls_index = cls_index[0].tolist()
            
        for cls in cls_index:      
            crop_idx = current_mask == cls 
            crop_uncertain = current_mask == -1
            crop_idx = crop_idx | crop_uncertain
            crop_image = torch.zeros((3,h,w)).cuda()
            
            crop_image[:,crop_idx] = images[i][:,crop_idx]
            
            crop_image = F.interpolate(crop_image.unsqueeze(0).detach(), size=(int(h/2),int(w/2)), mode="bilinear", align_corners=False)
            crop_image = crop_image.squeeze(0)
            
            crops.append(crop_image)
            flags.append(int(cls))
            
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')


    return crops, flags, box


def single_instance_cropv3(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, refine_pesudo_label= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    all_cls_masked_crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    refine_pesudo_label[refine_pesudo_label == 255] = -1
    refine_pesudo_label[refine_pesudo_label == 0] = -2
        
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        current_mask = refine_pesudo_label[i]
        cls_index = torch.unique(current_mask)
        cls_index = cls_index.tolist()
        cls_index = [x for x in cls_index if x>=0]
        if len(cls_index):
            cls = random.sample(cls_index,1)[0]
        else:
            cls_index = torch.where(cls_label[i] == 1)
            cls = random.sample(cls_index[0].tolist(),1)[0]
        
        crop_idx = current_mask == cls 
        crop_uncertain = current_mask == -1
        crop_idx = crop_idx | crop_uncertain
        crop_image = torch.zeros((3,h,w)).cuda()
        
        crop_image[:,crop_idx] = images[i][:,crop_idx]
        
        crop_image = F.interpolate(crop_image.unsqueeze(0).detach(), size=(int(h/2),int(w/2)), mode="bilinear", align_corners=False)
        
        crops.append(crop_image.squeeze(0))
        flags.append(int(cls))
        
        all_cls_crop_idx = current_mask != -2
        all_cls_crop_image = torch.zeros((3,h,w)).cuda()

        all_cls_crop_image[:,all_cls_crop_idx] = images[i][:,all_cls_crop_idx]
        all_cls_crop_image = F.interpolate(all_cls_crop_image.unsqueeze(0).detach(), size=(int(h/2),int(w/2)), mode="bilinear", align_corners=False)
        
        all_cls_masked_crops.append(all_cls_crop_image.squeeze(0))

    
        
        # crop_img = crops[-1]
        # crop_img = TF.to_pil_image(crop_img)
        # crop_img.save('crop_img_in_block_fg_crop.png')
        # crop_img = all_cls_masked_crops[-1]
        # crop_img = TF.to_pil_image(crop_img)
        # crop_img.save('crop_img_in_block_fg_crop.png')


    return crops, flags, box , all_cls_masked_crops



def single_instance_crop_down_scale(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, refine_pesudo_label= None, down_scale = 4):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    margin = int(crop_size/2)
    padding = (margin, margin, margin, margin)
    # padded_image = F.pad(images, padding, mode='constant', value=0)

    b, c, h, w = images.shape
    refine_pesudo_label[refine_pesudo_label == 255] = -1
    refine_pesudo_label[refine_pesudo_label == 0] = -2
        
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        current_mask = refine_pesudo_label[i]
        cls_index = torch.unique(current_mask)
        cls_index = cls_index.tolist()
        cls_index = [x for x in cls_index if x>=0]
        if len(cls_index):
            cls_index = cls_index
        else:
            cls_index = torch.where(cls_label[i] == 1)[0].tolist()
            
            
        for cls in cls_index:  
            crop_idx = current_mask == cls 
            crop_uncertain = current_mask == -1
            crop_idx = crop_idx | crop_uncertain
            crop_image = torch.zeros((3,h,w)).cuda()
            
            crop_image[:,crop_idx] = images[i][:,crop_idx]
            
            crop_image = F.interpolate(crop_image.detach(), size=(int(h/down_scale),int(w/down_scale)), mode="bilinear", align_corners=False)
            
            crops.append(crop_image)
            flags.append(int(cls))
            
            
            # crop_img = crops[-1]
            # crop_img = TF.to_pil_image(crop_img)
            # crop_img.save('crop_img_in_block_fg_crop.png')


    return crops, flags, box





def single_fg_crop_28x(images,cls_label = None, roi_mask_28x=None, crop_num=1, crop_size=64, roi_mask= None):
    #image 2 3 448 448
    #roi_mask 448 448 
    crops = []
    
    crop_size_28x = 5


    b, h, w = roi_mask_28x.shape
    
    box = []
        # 2 8 3 96 96 
    num_class = 20
    flags = []
    #roi middle crop 

    for i in range(b):
        for j in range(crop_num):
            current_mask = roi_mask_28x[i]
            cls_index = torch.where(cls_label[i] == 1)
            cls = random.sample(cls_index[0].tolist(),1)[0]
            
            fg_index = torch.where(current_mask == cls)
            
            
            if len(fg_index[0]):
                random_index = random.randint(0, fg_index[0].shape[0] - 1)
                crop_index_x, crop_index_y = fg_index[0][random_index], fg_index[1][random_index]
                select_index_x,select_index_y = crop_index_x, crop_index_y
                x_start = select_index_x - crop_size_28x // 2
                x_end = select_index_x + crop_size_28x // 2
                y_start = select_index_y - crop_size_28x // 2
                y_end = select_index_y + crop_size_28x // 2

            else:
                random_index_x = random.randint(0, h - 1)
                random_index_y = random.randint(0, h - 1)
                select_index_x,select_index_y = random_index_x,random_index_y
                x_start = select_index_x - crop_size_28x // 2
                x_end = select_index_x + crop_size_28x // 2
                y_start = select_index_y - crop_size_28x // 2
                y_end = select_index_y + crop_size_28x // 2
                
            if (x_end-x_start)!=(y_end-y_start):
                AssertionError("crop_out_margin")
            
            x_end = min(x_end,28)
            y_end = min(y_end,28)
            x_start = max(x_start,0)
            y_start = max(x_start,0)
            
            box.append((x_start,x_end,y_start,y_end))
            flags.append(cls+1)
            

    return crops, flags, box




def multi_scale_cam2(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape #（batch channel h w） inputs:原图
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        #dim = 0 ，batch维度上拼接
        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        #直接上采样  aug
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
        #选取同一个像素上的较大值

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        #缩放操作
        for s in scales:
            if s != 1.0:  #原图缩放
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
                #b c h w
                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))
        #torch.stack(cam_list, dim=0) 将 cam_list 中的 CAM 张量在新的维度 dim=0 上进行堆叠。
        # 这将创建一个形状为 (num_cams, b, c, h, w) 的新张量，
        # 其中 num_cams 是 CAM 的数量，b 是批量大小，c 是通道数，h 和 w 是 CAM 的高度和宽度。
        #作用是混合同一张图的不同scales的不同cam
        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)

        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1)) #保证cam最小值为0
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5 #归一化

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux


def multi_scale_cam2_du_heads(model, branch,inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    if branch == 'b1':
        branch = 0
    elif branch == 'b2':
        branch = 1
    else:
        AssertionError('Invalid branch')
    
    b, c, h, w = inputs.shape #（batch channel h w） inputs:原图
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        #dim = 0 ，batch维度上拼接
        _cam_aux, _cam = model(inputs_cat, cam_only=True)
        _cam_aux, _cam = _cam_aux[branch], _cam[branch]
        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        #直接上采样  aug
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
        #选取同一个像素上的较大值

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        #缩放操作
        for s in scales:
            if s != 1.0:  #原图缩放
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)
                _cam_aux, _cam = _cam_aux[branch], _cam[branch]

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
                #b c h w
                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))
        #torch.stack(cam_list, dim=0) 将 cam_list 中的 CAM 张量在新的维度 dim=0 上进行堆叠。
        # 这将创建一个形状为 (num_cams, b, c, h, w) 的新张量，
        # 其中 num_cams 是 CAM 的数量，b 是批量大小，c 是通道数，h 和 w 是 CAM 的高度和宽度。
        #作用是混合同一张图的不同scales的不同cam
        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)

        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1)) #保证cam最小值为0
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5 #归一化

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux




def multi_scale_cam_grad(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape #（batch channel h w） inputs:原图

    inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
    #dim = 0 ，batch维度上拼接
    _cam_aux, _cam = model(inputs_cat, cam_only=True)

    _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
    #直接上采样  aug
    _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
    _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
    _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
    #选取同一个像素上的较大值

    cam_list = [F.relu(_cam)]
    cam_aux_list = [F.relu(_cam_aux)]

    #缩放操作
    for s in scales:
        if s != 1.0:  #原图缩放
            _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
            inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

            _cam_aux, _cam = model(inputs_cat, cam_only=True)

            _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
            _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
            _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
            _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
            #b c h w
            cam_list.append(F.relu(_cam))
            cam_aux_list.append(F.relu(_cam_aux))
    #torch.stack(cam_list, dim=0) 将 cam_list 中的 CAM 张量在新的维度 dim=0 上进行堆叠。
    # 这将创建一个形状为 (num_cams, b, c, h, w) 的新张量，
    # 其中 num_cams 是 CAM 的数量，b 是批量大小，c 是通道数，h 和 w 是 CAM 的高度和宽度。
    #作用是混合同一张图的不同scales的不同cam
    cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)

    cam = cam + F.adaptive_max_pool2d(-cam, (1, 1)) #保证cam最小值为0
    cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5 #归一化

    cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
    cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
    cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux

def multi_scale_cam_test(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape #（batch channel h w） inputs:原图
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        #dim = 0 ，batch维度上拼接
        _cam_aux, _cam, _cam_crop = model(inputs_cat, cam_crop=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        #直接上采样  aug
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))

        _cam_crop = F.interpolate(_cam_crop, size=(h,w), mode='bilinear', align_corners=False)
        #直接上采样  aug
        _cam_crop = torch.max(_cam_crop[:b,...], _cam_crop[b:,...].flip(-1))


        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
        #选取同一个像素上的较大值

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]
        cam_crop_list = [F.relu(_cam_crop)]
        #缩放操作
        for s in scales:
            if s != 1.0:  #原图缩放
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam ,  _cam_crop = model(inputs_cat, cam_crop=True)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_crop = F.interpolate(_cam_crop, size=(h,w), mode='bilinear', align_corners=False)
              #直接上采样  aug
                _cam_crop = torch.max(_cam_crop[:b,...], _cam_crop[b:,...].flip(-1))

                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
                #b c h w
                cam_list.append(F.relu(_cam))
                cam_crop_list.append(F.relu(_cam_crop))
                cam_aux_list.append(F.relu(_cam_aux))
        #torch.stack(cam_list, dim=0) 将 cam_list 中的 CAM 张量在新的维度 dim=0 上进行堆叠。
        # 这将创建一个形状为 (num_cams, b, c, h, w) 的新张量，
        # 其中 num_cams 是 CAM 的数量，b 是批量大小，c 是通道数，h 和 w 是 CAM 的高度和宽度。
        #作用是混合同一张图的不同scales的不同cam
        # cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)

        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1)) #保证cam最小值为0
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5 #归一化

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5
        
        cam_crop = torch.sum(torch.stack(cam_crop_list, dim=0), dim=0)
        cam_crop = cam_crop + F.adaptive_max_pool2d(-cam_crop, (1, 1))
        cam_crop /= F.adaptive_max_pool2d(cam_crop, (1, 1)) + 1e-5

    return cam, cam_aux,cam_crop


def label_to_aff_mask(cam_label, ignore_index=255):

    #cam_label 2 28 28
    b,h,w = cam_label.shape
    # 2 784 784(判patch之间的pair)
    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0,2,1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    #转置相等就是同一类一个是表示 第0个patch的种类，一个是表示所有patch的class，合起来就是多少个和0th是同类
    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :]==ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :]==ignore_index, :] = ignore_index  #横纵都要做
    aff_label[:, range(h*w), range(h*w)] = ignore_index #对角线是自身，所以忽略
    return aff_label


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None, ignore_index=False,  img_box=None, down_scale=2):

    b,_,h,w = images.shape
    _images = F.interpolate(images, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b,1,h,w))*high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b,1,h,w))*low_thre
    bkg_l = bkg_l.to(cams.device)
    #两个阈值
    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()
    #拼接上原cam
    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)
    #down_scale cam 到 b 21 224 224
    
    for idx, coord in enumerate(img_box):

        valid_key = torch.nonzero(cls_labels[idx,...])[:,0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)  
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        #这里就是只取valid维度的，其他我都不管
        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))
        
        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1], coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1], coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0
    #同时为0才是0 bg区域
    return refined_label

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label



def cam_patch_contrast_loss(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None,fmap=None):
    #way1
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)  #pseudo_label 是对应得索引
    _pseudo_label += 1
    _pseudo_label[cam_value<(high_thre)] = 0
    count_fg = torch.sum(_pseudo_label,dim=(1,2))
    
    for i in range(b):
        feature_vector = []
        arg = torch.nonzero(cls_label[i]) + 1
        arg = arg.tolist()
        for cls in arg:
            cls = cls[0]
            if torch.all(_pseudo_label[i] != cls):
                value,indices = torch.topk(valid_cam[i][cls-1],k=3)
                row_indices = indices[:, 0]
                col_indices = indices[:, 1]
                sub_fmap = fmap[i, :, row_indices, col_indices]
                feature_vector.append(torch.mean(sub_fmap, dim=-1))
            else:
                indices = torch.nonzero(_pseudo_label[i] == cls)
                row_indices = indices[:, 0]
                col_indices = indices[:, 1]

                # 使用高级索引提取子张量
                sub_fmap = fmap[i, :, row_indices, col_indices]
                feature_vector.append(torch.mean(sub_fmap, dim=-1))

        features_vector = torch.stack(feature_vector,dim=0)
        similarity_matrix = torch.matmul(features_vector, features_vector.t())
        norms = torch.norm(features_vector, dim=1, keepdim=True)
        similarity_matrix /= torch.matmul(norms, norms.t())
        similarity_matrix = abs(similarity_matrix)
        similarity_matrix = torch.clamp(similarity_matrix, min=0, max=1)
        identity_matrix = torch.eye(similarity_matrix.size(0)).cuda()
        
        bce_loss = F.binary_cross_entropy(similarity_matrix,identity_matrix)



        print(similarity_matrix)

    

    #就是裁剪框外边的忽略掉，只看里边的
    return bce_loss

def cam_to_label_resized(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None,printornot = False,clip = False):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)  #pseudo_label 是对应得索引
    _pseudo_label += 1
    _pseudo_label[cam_value<=bkg_thre] = 0
            #b h w

    #cam value [b 448 448] _pseudo_label [b 448 448]每个位置是索引
    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=high_thre] = ignore_index
        _pseudo_label[cam_value<=low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    
    for idx, coord in enumerate(img_box):
        coord = coord // 16
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    if printornot:            
        plt.imshow((pseudo_label[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
        plt.colorbar()
        plt.title("aux_mask")
            
        plt.savefig(f'aux_mask.png')
        plt.close()

    return valid_cam, pseudo_label

def get_per_pic_thre(pesudo_label,gd_label,uncertain_region_thre):
    #pesudo_label [b,h,w]
    
    b,h,w = pesudo_label.size()
    _,c = gd_label.size()
    flatten_pesudo_label = pesudo_label.view(b,h*w)
    flatten_pesudo_label = flatten_pesudo_label + 1

    # -1 是确定的背景类 0是uncertain  -> 0是bg区域
    #elements_list = {}
    thre_list = []
    for i in range(b):
        count_uncertain = torch.where(flatten_pesudo_label[i]==0)
        count_uncertain = count_uncertain[0].size()[0]
        flatten_pesudo_label[i][flatten_pesudo_label[i]==-1] = 0        
        elements,counts = torch.unique(flatten_pesudo_label[i],dim = -1,return_counts = True)
        thre = counts / (h*w)
        thre_uncertain = count_uncertain / (h*w)
        temp_thre = torch.zeros(c+1).cuda()
        temp_thre[elements.tolist()] = torch.minimum(thre + uncertain_region_thre * (thre_uncertain/ ((len(elements)-1)+1e-6)),torch.tensor(0.99))
        thre_list.append(temp_thre)
    per_pic_thre = torch.stack(thre_list)
    return per_pic_thre

def get_per_pic_thre_v2(pesudo_label,gd_label,uncertain_region_thre):
    #pesudo_label [b,h,w]
    
    b,h,w = pesudo_label.size()
    _,c = gd_label.size()
    flatten_pesudo_label = pesudo_label.view(b,h*w)
    flatten_pesudo_label = flatten_pesudo_label + 1

    # -1 是确定的背景类 0是uncertain  -> 0是bg区域
    #elements_list = {}
    thre_list = []
    for i in range(b):
        count_uncertain = torch.where(flatten_pesudo_label[i]==0)
        count_uncertain = count_uncertain[0].size()[0]
        flatten_pesudo_label[i][flatten_pesudo_label[i]==-1] = 0        
        elements,counts = torch.unique(flatten_pesudo_label[i],dim = -1,return_counts = True)
        thre = counts / (h*w)
        thre_uncertain = count_uncertain / (h*w)
        temp_thre = torch.zeros(c+1).cuda()
        temp_thre[elements.tolist()] = torch.minimum(thre,torch.tensor(0.99))
        thre_list.append(temp_thre)
    per_pic_thre = torch.stack(thre_list)
    return per_pic_thre



def multi_scale_mask_cam(model, inputs, scales, img_name,sam_mask):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape #（batch channel h w） inputs:原图
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        #dim = 0 ，batch维度上拼接
        _cam_aux, _cam = model(inputs_cat, cam_only=True,img_name=img_name,sam_mask=sam_mask)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        #直接上采样  aug
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
        #选取同一个像素上的较大值

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        #缩放操作
        for s in scales:
            if s != 1.0:  #原图缩放
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True,img_name=img_name, sam_mask=sam_mask)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
                #b c h w
                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))
        #torch.stack(cam_list, dim=0) 将 cam_list 中的 CAM 张量在新的维度 dim=0 上进行堆叠。
        # 这将创建一个形状为 (num_cams, b, c, h, w) 的新张量，
        # 其中 num_cams 是 CAM 的数量，b 是批量大小，c 是通道数，h 和 w 是 CAM 的高度和宽度。
        #作用是混合同一张图的不同scales的不同cam
        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)

        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1)) #保证cam最小值为0
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5 #归一化

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux


def get_sorted_mask_cls_dict(mask_cls_list , cls_label):

    b = len(mask_cls_list)
    for i in range(b):
        cls_idx = torch.where(cls_label[i]==1)
        current_mask_cls_list = mask_cls_list[i]
        for idx in cls_idx:
            idx = idx.item()
            def custom_sort(item):
                return item[1][idx].item()

            sorted_items = sorted(current_mask_cls_list.items(), key=custom_sort, reverse=True)
            sorted_dict = {k: v for k, v in sorted_items}

    return sorted_dict

def get_sam_mask_label(mask_cam_list, cls_label,sam_mask,bg_thre,mask_cls_list=None):
    b = len(mask_cam_list)
    sam_mask_label = torch.zeros_like(sam_mask).to(torch.long)
    
    for i in range(b):
        current_sam_mask_label = sam_mask_label[i]
        current_cam_dict = mask_cam_list[i]
        current_label = cls_label[i]
        for key,value in current_cam_dict.items():
            if key == -1:
                continue
            else:
                cls = mask_cls_list[i][key].detach()
                cls = F.sigmoid(cls)
                
                value[current_label==0] = 0
                value = torch.cat((torch.tensor([bg_thre]).cuda(),value))
                idx_matrix = sam_mask[i] == key
                argmax = torch.max(value,dim=-1)[1]
                if argmax == 0: #bg class
                    current_sam_mask_label[idx_matrix] = 0
                elif cls[argmax-1] <= 0.75: #low activation
                    current_sam_mask_label[idx_matrix] = 0
                else: #high activation
                    current_sam_mask_label[idx_matrix] = argmax
            
    return sam_mask_label