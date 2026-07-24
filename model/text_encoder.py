import json
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import torch
import torch.nn as nn
from collections import OrderedDict
import math
import numpy as np
from PIL import Image
import clip_tagclip
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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
    
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features.float())  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx = ctx.expand(bias.shape[0],ctx.shape[1],ctx.shape[2])
        bias_reshape = bias.expand(bias.shape[0],self.n_ctx,bias.shape[2])
        mask = [[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]]
        mask = torch.BoolTensor(mask).cuda()
        
        bias_masked = bias_reshape.masked_fill(mask,value=torch.tensor(0))
        ctx_shifted = ctx + bias_masked           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

    def forward(self,image_features):
        prompts = self.prompt_learner(image_features).cuda()
        tokenized_prompts = self.tokenized_prompts
        a = 0
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = torch.unsqueeze(text_features, dim=0)
            if a == 0:
                text_features_resize = text_features
            else:
                text_features_resize = torch.cat((text_features_resize,text_features),0)
            a = a + 1
        
        return text_features_resize

class CLIPFeatureEncoder(nn.Module):
    def __init__(self, BACKBONE_NAME):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.transform = clip.load(BACKBONE_NAME, device=self.device)
        self.model_tagclip,self.transform_tagclip = clip_tagclip.load(BACKBONE_NAME, device=self.device)
        if BACKBONE_NAME == 'ViT-L/14':
            self.n_grid = 16
        elif BACKBONE_NAME == 'ViT-L/14@336px':
            self.n_grid = 24
        else:
            self.n_grid = 14
    
    # def create_mask(self, n_grid, bbox):
    #     mask = np.ones((n_grid, n_grid), dtype=int)

    #     x1, y1 = map(math.floor, bbox[:2])
    #     x2, y2 = map(math.ceil, bbox[2:])

    #     x1, y1 = max(x1, 0), max(y1, 0)
    #     x2, y2 = min(x2, n_grid), min(y2, n_grid)

    #     mask[y1:y2, x1:x2] = 0

    #     return mask

    # def create_mask(self, x,y, bbox):
    #     mask = np.ones((y, x), dtype=int)

    #     x1, y1 = map(math.floor, bbox[:2])
    #     x2, y2 = map(math.ceil, bbox[2:])

    #     x1, y1 = max(x1, 0), max(y1, 0)
    #     x2, y2 = min(x2, x), min(y2, y)

    #     mask[y1:y2, x1:x2] = 0

    #     return mask

    def create_mask(self,x, y, bbox):
    # 初始化mask，所有值先设为0（表示没有覆盖）
        mask = np.zeros((y, x), dtype=float)

        # bbox的坐标，可能包含小数
        x1, y1, x2, y2 = bbox
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, x), min(y2, y)

        # 遍历每个patch
        for i in range(y):
            for j in range(x):
                # 计算当前patch的边界
                patch_x1, patch_y1, patch_x2, patch_y2 = j, i, j + 1, i + 1
                
                # 计算交叉区域的边界
                inter_x1 = max(x1, patch_x1)
                inter_y1 = max(y1, patch_y1)
                inter_x2 = min(x2, patch_x2)
                inter_y2 = min(y2, patch_y2)
                
                # 如果有交叉区域，则计算交叉区域的面积
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    patch_area = 1  # 每个patch的面积假设为1（因为是单位大小）
                    # 更新mask中的值为交叉区域面积占patch面积的比例
                    mask[i, j] = inter_area / patch_area
        return mask

    # def forward(self, regions):
    #     self.model.eval()
    #     with torch.no_grad():
    #         features = []
    #         for region in regions:
    #             image_path = region[0]
    #             bbox = region[1]
    #             image = Image.open(image_path).convert("RGB")
    #             width, height = image.size
    #             flag = 0
    #             if bbox == None:
    #                 flag = 1
    #                 bbox = [0,0,width,height]
    #             image_resize = self.transform(image).unsqueeze(0).to(device=self.device)
    #             x1 = bbox[0] / width * self.n_grid
    #             x2 = bbox[2] / width * self.n_grid
    #             y1 = bbox[1] / height * self.n_grid
    #             y2 = bbox[3] / height * self.n_grid
    #             bbox_resize = [x1, y1, x2, y2]
    #             mask = self.create_mask(self.n_grid, bbox_resize)
    #             x = self.model.encode_image(image_resize, mask).squeeze().cpu().tolist()
    #             # if flag:
    #             #     feature = x1.squeeze().cpu().tolist()
    #             # else:
    #             #     feature = x2.squeeze().cpu().tolist()
    #             features.append(x)
    #         feature = np.array(features)
    #     return feature
    
    def forward(self, regions):
        self.model.eval()
        with torch.no_grad():
            features = []
            image_path = regions[0][0]
            image = Image.open(image_path).convert("RGB")
            # image_resize_yuansheng = self.transform(image).unsqueeze(0).to(device=self.device)
            
            width, height = image.size
            array_img = np.array(image)
            ori_height, ori_width = array_img.shape[:2]
            patch_size = 16
            preprocess = _transform_resize(336, 336)
            image_resize = preprocess(image).unsqueeze(0).to(self.device)
            _, x0 = self.model.encode_image(image_resize)
            h, w = image_resize.shape[-2], image_resize.shape[-1]
            # image_features, _ = self.model_tagclip.encode_image_tagclip(image_resize, h, w, attn_mask=1)
            _, image_features = self.model_tagclip.encode_image_tagclip(image_resize, h, w, attn_mask=1)
            for region in regions:
                bbox = region[1]
                flag = 0
                if bbox == None:
                    flag = 1
                    bbox = [0,0,width,height]
                x1 = bbox[0] / width * 24
                x2 = bbox[2] / width * 24
                y1 = bbox[1] / height * 24
                y2 = bbox[3] / height * 24
                bbox_resize = [x1, y1, x2, y2]
                # mask = self.create_mask(self.n_grid, bbox_resize)
                # mask = self.create_mask(int(np.ceil(ori_width/16)),int(np.ceil(ori_height/16)), bbox_resize)
                mask = self.create_mask(24,24, bbox_resize)
                # x1,x2 = self.model.encode_image(image_resize, mask)#.squeeze().cpu().tolist()
                mask = np.array(mask)
                mask = mask.flatten()
                # zero_positions = np.where(mask == 0)[0]
                # xx1 = copy.deepcopy(image_features[:,0,:])
                mask_tensor = torch.from_numpy(mask).float().cuda()
                # mask_tensor = torch.where(mask_tensor < 0.05, torch.zeros_like(mask_tensor), mask_tensor)
                # is_all_zeros = (mask_tensor == 0).all()
                # if is_all_zeros:
                #     sb
                mask_tensor = mask_tensor.unsqueeze(0)
                # xx2 = copy.deepcopy(image_features[:,1:,:])
                xx2 = copy.deepcopy(image_features)
                mask_tensor = mask_tensor.unsqueeze(-1)
                mask_tensor = mask_tensor.expand_as(xx2)
                weights_sum = mask_tensor.sum(dim=1, keepdim=True).clamp(min=1e-9)
                xx3_weighted = xx2 * mask_tensor
                xx4 = xx3_weighted.sum(dim=1) / weights_sum
                if flag:
                    feature = x0.squeeze().cpu().tolist()
                else:
                    feature = xx4.squeeze().cpu().tolist()
                features.append(feature)
            feature = np.array(features)
        return feature
