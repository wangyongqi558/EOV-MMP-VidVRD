import os
import numpy as np
import pickle
import _pickle as cPickle
import json
from collections import defaultdict
from os.path import join
import random
from PIL import Image

import clip
import clip_tagclip
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import copy

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w):
    return Compose([
        #Resize(n_px, interpolation=BICUBIC),
        Resize((h,w), interpolation=BICUBIC),
        # CenterCrop(224),
        #RandomHorizontalFlip(1.0),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class Dataset_new(BaseDataset):

    def __init__(self, args, split):
        super().__init__()
        self.dataset = args.dataset
        self.split = split
        base_vidvrd = '/data/wyq/jixf/jixf/VidVRD-baseline/dataset/vidvrd'
        data_dir = join(base_vidvrd, 'data')
        self.anno_train_dir = join(base_vidvrd, 'anno', 'train')
        if split=='train':
            feat_path = join(data_dir, 'train_object_trajectories_gt.json')
        elif split=='val':
            self.gt_rels = json.load(open(join(data_dir, "test_relation_gt.json"), "r"))
            feat_path = join(data_dir, 'test_object_trajectories_gt.json')

        self.FEAT_ROOT = join(base_vidvrd, 'frames')
        self.gt_traj = json.load(open(feat_path,"r"))
        self.path_list = list(self.gt_traj.keys())
        self.pre_split = json.load(open(
            join(data_dir, 'openvoc_pred_class_spilt_info.json'), "r"))
        self.obj_split = json.load(open(
            join(data_dir, 'openvoc_obj_class_spilt_info.json'), "r"))
        self.pre_num = len(self.pre_split['id2cls'])
        self.prior = pickle.load(open(join(data_dir, 'prior.pkl'), 'rb'))
        self.preprocess = _transform_resize(336, 336)
        self.clip, _= clip.load('ViT-L/14@336px', device='cuda')
        self.tagclip,_ = clip_tagclip.load('ViT-L/14@336px', device='cuda')
        self.id2pre = json.load(open(join(data_dir, 'id2predicate.json'), "r"))
        self.pre2id = json.load(open(join(data_dir, 'predicate2id.json'), "r"))
        self.id2obj = json.load(open(join(data_dir, 'id2object.json'), "r"))
        self.obj2id = json.load(open(join(data_dir, 'object2id.json'), "r"))

    def __getitem__(self, index):
        video_name = self.path_list[index]
        data_path = self.FEAT_ROOT + '/' + video_name
        frame_list = sorted(os.listdir(data_path))
        video_len = len(frame_list)
        item = {}
        patch_ = []
        patch_proj = []
        global_proj = []
        for i in frame_list:
            # if int(i.split('.')[0])%30 == 1:
                with torch.no_grad():
                    frame_path = data_path +'/'+i
                    # frame_path = '../dataset/vidvrd/frames/ILSVRC2015_val_00046001/000100.jpg'
                    image = Image.open(frame_path).convert("RGB")
                    image_w, image_h = image.size
                    image_resize = self.preprocess(image).unsqueeze(0).cuda()
                    _, gp = self.clip.encode_image(image_resize)
                    gp = gp.cpu()
                    global_proj.append(gp)
                    p, pp = self.tagclip.encode_image_tagclip(image_resize, 336, 336, attn_mask=1)
                    p = p.cpu()
                    pp = pp.cpu()
                    patch_.append(p)
                    patch_proj.append(pp)
            # else:
            #     global_proj.append(gp)
            #     patch_.append(p)
            #     patch_proj.append(pp)
        begin_fid = video_len
        end_fid = 0
        for t in self.gt_traj[video_name]:
            if t['begin_fid'] < begin_fid:
                begin_fid = t['begin_fid']
            if t['end_fid'] > end_fid:
                end_fid = t['end_fid']
        item['video_name'] = video_name
        item['video_len'] = video_len
        item['patch_'] = torch.cat(patch_,dim=0)
        item['patch_proj'] = torch.cat(patch_proj,dim=0)
        item['global_proj'] = torch.cat(global_proj,dim=0)
        item['image_size'] = [image_w,image_h]
        item['gt_traj'] = self.gt_traj[video_name]
        item['begin_fid'] = begin_fid
        item['end_fid'] = end_fid

        if self.split == 'train':
            anno_path = join(self.anno_train_dir, video_name + '.json')
            anno = json.load(open(anno_path,"r"))
            gt_rel = anno['relation_instances']
            item['gt_rel'] = gt_rel
        
        return item

    def __len__(self):
        return len(self.path_list)
