# Modified from [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)

import os

import torch
import torch.nn as nn
from clip import clip
import util.misc as util

from .coco_categories import COCO_CATEGORIES
from .lvis_v1_categories import LVIS_CATEGORIES
from .vidvrd_categories import VidVRD_CATEGORIES
from .vidor_categories import VidOR_CATEGORIES


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res

# coco class meta
coco_unseen_list = [4, 5, 11, 12, 15, 16, 21, 23, 27, 29, 32, 34, 45, 47, 54, 58, 63]


single_template = ["a photo of a {}."]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]




def load_clip_to_cpu(visual_backbone):
    backbone_name = visual_backbone
    url = clip._MODELS[backbone_name]

    root = os.path.expanduser("~/.cache/clip")
    if not util.is_dist_avail_and_initialized():
        clip._download(url, root)
    else:
        if util.is_main_process():
            print('clip download with barrier - distributed setup.')
            clip._download(url, root)
        torch.distributed.barrier()

    filename = os.path.basename(url)
    model_path = os.path.join(root, filename)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

# clip_name "ViT-B/32" (OV-DETR)
def build_text_embedding_coco(clip_name, bg=False):
    categories = COCO_CATEGORIES

    if bg:
        categories[len(categories)+1] = 'background'

    run_on_gpu = torch.cuda.is_available()

    clip_model = load_clip_to_cpu(clip_name)


    text_model = TextEncoder(clip_model)
    if run_on_gpu:
        text_model = text_model.cuda()

    for _, param in text_model.named_parameters():
        param.requires_grad = False

    templates = multiple_templates
    with torch.no_grad():
        zeroshot_weights = []

        for _, category in categories.items():

            texts = [
                template.format(processed_name(category, rm_dot=True), article=article(category))
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]

            texts = clip.tokenize(texts)  # tokenize

            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = text_model(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            zeroshot_weights.append(text_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        if run_on_gpu:
            zeroshot_weights = zeroshot_weights.cuda()
    zeroshot_weights = zeroshot_weights.t()
    all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]  # noqa
    if bg:
        all_ids.append(len(categories))
    all_ids = [i - 1 for i in all_ids]
    
    print('complete to build text embeddings.')
    return zeroshot_weights[all_ids].float()

def build_text_embedding_vidvrd(clip_name, bg=False):
    categories = VidVRD_CATEGORIES

    if bg:
        categories[len(categories)] = 'background'

    run_on_gpu = torch.cuda.is_available()

    clip_model = load_clip_to_cpu(clip_name)


    text_model = TextEncoder(clip_model)
    if run_on_gpu:
        text_model = text_model.cuda()

    for _, param in text_model.named_parameters():
        param.requires_grad = False

    templates = multiple_templates
    with torch.no_grad():
        zeroshot_weights = []

        for _, category in categories.items():

            texts = [
                template.format(processed_name(category, rm_dot=True), article=article(category))
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]

            texts = clip.tokenize(texts)  # tokenize

            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = text_model(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            zeroshot_weights.append(text_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        if run_on_gpu:
            zeroshot_weights = zeroshot_weights.cuda()
    zeroshot_weights = zeroshot_weights.t()
    all_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    if bg:
        all_ids.append(35)
    print('complete to build text embeddings.')
    return zeroshot_weights[all_ids].float()

def build_text_embedding_vidor(clip_name, bg=False):
    categories = VidOR_CATEGORIES

    if bg:
        categories[len(categories)] = 'background'

    run_on_gpu = torch.cuda.is_available()

    clip_model = load_clip_to_cpu(clip_name)


    text_model = TextEncoder(clip_model)
    if run_on_gpu:
        text_model = text_model.cuda()

    for _, param in text_model.named_parameters():
        param.requires_grad = False

    templates = multiple_templates
    with torch.no_grad():
        zeroshot_weights = []

        for _, category in categories.items():
            category = category.replace("_", " ")
            category = category.replace("/", " or ")
            texts = [
                template.format(processed_name(category, rm_dot=True), article=article(category))
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]

            texts = clip.tokenize(texts)  # tokenize

            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = text_model(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            zeroshot_weights.append(text_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        if run_on_gpu:
            zeroshot_weights = zeroshot_weights.cuda()
    zeroshot_weights = zeroshot_weights.t()
    all_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
    35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,
    70,71,72,73,74,75,76,77,78,79]
    if bg:
        all_ids.append(80)
    print('complete to build text embeddings.')
    return zeroshot_weights[all_ids].float()


def build_text_embedding_lvis(clip_name, bg=False):
    categories = LVIS_CATEGORIES

    if bg:
        categories.append(
            {
                'name': 'background'
            }
        )
    model, _ = clip.load(clip_name)
    templates = multiple_templates

    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        for category in categories:
            texts = [
                template.format(
                    processed_name(category["name"], rm_dot=True), article=article(category["name"])
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)

        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()

    all_text_embeddings = all_text_embeddings.t()
    return all_text_embeddings.float()

'''
def build_text_embedding_coco(clip_name):
    categories = COCO_CATEGORIES
    run_on_gpu = torch.cuda.is_available()

    clip_model = load_clip_to_cpu(clip_name)


    text_model = TextEncoder(clip_model)
    if run_on_gpu:
        text_model = text_model.cuda()

    for _, param in text_model.named_parameters():
        param.requires_grad = False
    templates = multiple_templates
    with torch.no_grad():
        zeroshot_weights = []

        for _, category in categories.items():
            texts = [
                template.format(processed_name(category, rm_dot=True), article=article(category))
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]

            texts = clip.tokenize(texts)  # tokenize

            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = text_model(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            zeroshot_weights.append(text_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        if run_on_gpu:
            zeroshot_weights = zeroshot_weights.cuda()
    zeroshot_weights = zeroshot_weights.t()
    all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]  # noqa
    all_ids = [i - 1 for i in all_ids]

    print('complete to build text embeddings.')
    return zeroshot_weights[all_ids].float()


def build_text_embedding_lvis(clip_name):
    categories = LVIS_CATEGORIES
    model, _ = clip.load(clip_name)
    templates = multiple_templates

    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        for category in categories:
            texts = [
                template.format(
                    processed_name(category["name"], rm_dot=True), article=article(category["name"])
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()

    all_text_embeddings = all_text_embeddings.t()
    return all_text_embeddings.float()
'''