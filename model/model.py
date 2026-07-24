from clip import clip
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import math
import json
import os
from os.path import join
from text_encoder import CustomCLIP

class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.visual_encoder = CLIPFeatureEncoder(args.clip_backbone)
        base_obj_cls = []
        novel_obj_cls = []
        self.obj_cls_split = json.load(open(
            '../dataset/vidvrd/data/openvoc_obj_class_spilt_info.json', "r"))
        self.obj_cls_split = self.obj_cls_split["cls2split"]
        for i in self.obj_cls_split:
            if self.obj_cls_split[i] == "base":
                base_obj_cls.append(i)
            else:
                novel_obj_cls.append(i)
        self.base_obj_cls = base_obj_cls
        self.novel_obj_cls = novel_obj_cls
        self.all_obj_cls = base_obj_cls + novel_obj_cls
        
        classnames = [name.replace("_", " ") for name in self.all_obj_cls]
        clip_model, _ = clip.load(self.args.clip_backbone, device='cpu')
        clip_model = clip_model.cuda()

        self.text_encoder = CustomCLIP(self.args, classnames, clip_model)
        self.criterion = nn.CrossEntropyLoss()


    def build_clip_fixed_prompts(self, is_train):
        if is_train == "train":
            split = self.args.train_split
        else:
            split = self.args.test_split
        classnames = getattr(self, f"{split}_obj_cls")
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [f"An image of {name}." for name in classnames]
        prompts = clip.tokenize(prompts).cuda()
        model, _ = clip.load(name=self.args.clip_backbone, device='cpu')
        model = model.cuda().eval()
        with torch.no_grad():
            text_embeddings = model.encode_text(prompts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings

    def forward(self, visual_feats):
        text_feats = self.text_encoder(visual_feats)
        visual_feats = visual_feats.unsqueeze(dim=1).float()
        # visual_feats /= visual_feats.norm(dim=-1, keepdim=True)
        similarity = torch.bmm(visual_feats, text_feats.transpose(1, 2))*100
        similarity = similarity.squeeze(dim=1)

        if not self.training:
            similarity = torch.softmax(similarity, dim=-1)
            return similarity
