import copy
import math
import time
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from util.clip_utils import build_text_embedding_coco, build_text_embedding_lvis, build_text_embedding_vidvrd, build_text_embedding_vidor
from methods.fpn_fusion import SimpleFeaturePyramid, LastLevelMaxPool
from .position_encoding import build_position_encoding
from methods.backbones.vit_det import local_deit_base
from methods.backbones.vit_det import local_deit_tiny as local_deit_tiny
from methods.backbones.clip_backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .matcher import build_matcher
from .post_process import OVPostProcess, PostProcessSegm
from .segmentation import DETRsegm
from .set_criterion import OVSetCriterion
import torchvision.transforms.functional as TF

from util.coco_categories import SEEN_CATEGORIES as COCO_SEEN_CATEGORIES
from util.lvis_v1_categories import SEEN_CATEGORIES as LVIS_SEEN_CATEGORIES
from util.vidvrd_categories import SEEN_CATEGORIES as VidVRD_SEEN_CATEGORIES
from util.vidor_categories import SEEN_CATEGORIES as VidOR_SEEN_CATEGORIES


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CLIP_Classifier(nn.Module):

    def __init__(self, in_dim=256, out_dim=512, clip_distil=False):
        super().__init__()

        self.clip_distil = clip_distil

        # cls head
        self.cls_layer = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.cls_layer.weight)
        nn.init.constant_(self.cls_layer.bias, 0)
        
        # det token -> clip embedding
        if self.clip_distil:
            self.feature_align = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(self.feature_align.weight)
            nn.init.constant_(self.feature_align.bias, 0)

    def forward(self, det_query, clip_query):

        bs = det_query.shape[0]

        # projection for clip distil
        projected_query = None
        if self.clip_distil:
            projected_query = self.feature_align(det_query)

        # cls head
        cls_query = self.cls_layer(det_query)
        clip_query = clip_query.unsqueeze(0).expand(bs, -1, -1)

        logit = (cls_query @ clip_query.permute(0, 2, 1))

        # return class prob, & prejected_query
        return logit, projected_query


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDETR(nn.Module):
    def __init__(self, backbone, transformer, pos_encoding, num_queries,
                 aux_loss=False, with_box_refine=False,
                 cross_scale_fusion=None):
        super().__init__()
        self.num_queries = num_queries
        self.pos_encoding = pos_encoding
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.backbone = backbone

        # two essential techniques used [default use]
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        # object tokens (object queries)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.pred_query_embed = nn.Embedding(1, hidden_dim * 2)

        # [PATCH] token channel reduction for the input to transformer decoder
        if cross_scale_fusion is None:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    # This is 1x1 conv -> so linear layer
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)

            # initialize the projection layer for [PATCH] tokens
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            self.fusion = None
        else:
            # the cross scale fusion module has its own reduction layers
            self.fusion = cross_scale_fusion

        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # the prediction is made for each decoding layers
        num_pred = transformer.decoder.num_layers

        # set up all required nn.Module for additional techniques
        if with_box_refine:
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        x = samples.tensors # RGB input
        mask = samples.mask # padding mask

        # return multi-scale [PATCH] tokens
        features = self.backbone(x) # deit는 attention masking 안 들어있음.

        # [PATCH] token projection - Simplified ViT-DET projection
        srcs = []
        if self.fusion is None:
            # Applying only projection for the four scale feature maps.
            for l, src in enumerate(features):
                srcs.append(self.input_proj[l](src))
        else:
            # Applying single FPN (only using the last stage feature maps)
            srcs = self.fusion(features[-1])

        # generate pos encoding and pad mask for attention
        pos, masks = [], []
        for src in srcs:
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            #pos.append(self.pos_encoding(src, _mask))

        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_token_class_unflat \
          = self.transformer(srcs, masks, query_embeds) # no pos encoding

        outputs_classes = []
        outputs_coords = []
        # perform predictions via the detection head
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            ## bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

        # stack all predictions made from each decoding layers
        outputs_coord = torch.stack(outputs_coords)

        # final prediction is made the last decoding layer
        out = {'pred_boxes': outputs_coord[-1]}

        # aux loss is defined by using the rest predictions
        if self.aux_loss and self.transformer.decoder.num_layers > 0:
            out['aux_outputs'] = self._set_aux_loss(outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class OVDETR(DeformableDETR):
    def __init__(
        self,
        backbone,
        transformer,
        pos_encoding,
        num_queries,
        aux_loss=True,
        with_box_refine=False,
        cross_scale_fusion=None,
        zeroshot_w=None,
        clip_feat_path=None,
        prob=0.75,
        distil_clip_embed=False,
        #
        seen_list=None,
        iou_aware=False,
        token_label=False,
    ):
        super().__init__(
            backbone,
            transformer,
            pos_encoding,
            num_queries,
            aux_loss,
            with_box_refine,
            cross_scale_fusion,
        )

        self.zeroshot_w = zeroshot_w.t()
        self.distil_clip_embed = distil_clip_embed
        self.patch2query = nn.Linear(self.zeroshot_w.shape[0], 256)
        self.patch2query_img = nn.Linear(self.zeroshot_w.shape[0], 256)
        for layer in [self.patch2query, self.patch2query_img]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

        # clip-based classification
        self.class_embed = CLIP_Classifier(256, self.zeroshot_w.shape[0], clip_distil=distil_clip_embed)
        self.seen_list = seen_list
        self.pred_embed = MLP(256,512,71,2)

        # additional losses for VIDT+
        self.iou_aware = iou_aware
        self.token_label = token_label

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
        else:
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])

        # all class ids. e.g., 65 for MS-COCO
        self.all_ids = torch.tensor(range(self.zeroshot_w.shape[-1]))
        self.clip_feat = torch.load(clip_feat_path)
        self.prob = prob

        # IoU Aware
        if self.iou_aware:
            self.iou_embed = MLP(256, 256, 1, 3)
            if with_box_refine:
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            else:
                self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])

    def forward(self, samples: NestedTensor, targets=None):
        if self.training:
            return self.forward_train(samples, targets)
        else:
            return self.forward_test(samples)

    def forward_train(self, samples, targets=None):

        mask = torch.zeros(samples.shape[0], 336, 336, dtype=torch.bool)
        features = samples
        w = h = int(math.sqrt(features.shape[1]))
        features = features.reshape(features.shape[0],w,h,features.shape[-1])
        features = features.permute(0, 3, 1, 2).float()
        srcs = self.fusion(features)

        # generate pos encoding and pad mask for attention
        pos, masks = [], []
        for src in srcs:
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            #pos.append(self.pos_encoding(src, _mask))
        #########################

        # new part.
        uniq_labels = torch.cat([t["labels"] for t in targets])
        uniq_labels = torch.unique(uniq_labels).to("cpu")
        uniq_labels = uniq_labels[torch.randperm(len(uniq_labels))]
        select_id = uniq_labels.tolist()

        if self.seen_list is not None:
            select_id = list(set(uniq_labels + self.seen_list))
        text_query = self.zeroshot_w[:, select_id].t()
        img_query = []
        for cat_id in select_id:
            # takes one random target object clip embedding.
            index = torch.randperm(len(self.clip_feat[cat_id]))[0:1]
            img_query.append(self.clip_feat[cat_id][index])
        # to tensor.
        img_query = torch.cat(img_query).to(text_query.device)
        # transform to unit vector.
        img_query = img_query / img_query.norm(dim=-1, keepdim=True)

        # if < 0.75 (75%) -> bool -> float -> [len, 1]
        mask = (torch.rand(len(text_query)) < self.prob).float().unsqueeze(1).to(text_query.device)
        # 75% text query + 25% img query
        clip_query_ori = (text_query * mask + img_query * (1 - mask)).detach()

        dtype = self.patch2query.weight.dtype
        # projection = patch2query
        text_query = self.patch2query(text_query.type(dtype))
        img_query = self.patch2query_img(img_query.type(dtype))

        clip_query = text_query * mask + img_query * (1 - mask)

        # class agnostic tokens.
        query_embeds = self.query_embed.weight
        pred_query_embed = self.pred_query_embed.weight
        query_embeds = torch.cat((query_embeds, pred_query_embed), dim=0)

        (hs, init_reference, inter_references, clip_query, enc_token_class_unflat), _ =\
            self.transformer(srcs, masks, query_embeds, text_query=clip_query) # text query -> new part
        pred_hs = hs[-1,:,-1,:]
        new_logit = torch.sigmoid(self.pred_embed(pred_hs))
        # pred_hs = pred_hs / pred_hs.norm(dim=-1, keepdim=True)
        # new_logit = torch.sigmoid(100*pred_hs@self.pred_text_feature.T)
        
        hs_ = hs[:,:,:-1,:]
        init_reference = init_reference[:,:-1,:]
        inter_references = inter_references[:,:,:-1,:]
        outputs_coords = []
        outputs_det_tokens = []
        outputs_embeds = []
        outputs_classes = []
        for lvl in range(hs_.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class, projected_hs = self.class_embed[lvl](hs_[lvl], clip_query_ori)

            tmp = self.bbox_embed[lvl](hs_[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_det_tokens.append(hs_[lvl])

            # new
            outputs_classes.append(outputs_class)
            outputs_embeds.append(projected_hs)

        outputs_coord = torch.stack(outputs_coords)
        outputs_det_tokens = torch.stack(outputs_det_tokens)
        outputs_class = torch.stack(outputs_classes)
        if self.distil_clip_embed:
            outputs_embed = torch.stack(outputs_embeds)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "select_id": select_id,
            "clip_query": clip_query_ori,
            "new_logit" : new_logit
        }

        if self.distil_clip_embed:
            out["pred_embed"] = outputs_embed[-1]

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
            if self.distil_clip_embed:
                for temp, embed, det_token in zip(out["aux_outputs"], outputs_embed[:-1], outputs_det_tokens[:-1]):
                    temp["select_id"] = select_id
                    temp["pred_embed"] = embed
                    temp["clip_query"] = clip_query_ori
            else:
                for temp, det_token in zip(out["aux_outputs"], outputs_det_tokens[:-1]):
                    temp["select_id"] = select_id
                    temp["clip_query"] = clip_query_ori

        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.iou_aware:
            outputs_ious = []
            for lvl in range(hs_.shape[0]):
                outputs_ious.append(self.iou_embed[lvl](hs_[lvl]))
            outputs_iou = torch.stack(outputs_ious)
            out['pred_ious'] = outputs_iou[-1]

            if self.aux_loss:
                for i, aux in enumerate(out['aux_outputs']):
                    aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}
        return out, select_id

    def forward_test(self, samples):
        mask = torch.zeros(samples.shape[0], 336, 336, dtype=torch.bool)
        features = samples
        w = h = int(math.sqrt(features.shape[1]))
        features = features.reshape(features.shape[0],w,h,features.shape[-1])
        features = features.permute(0, 3, 1, 2).float()


        # [PATCH] token projection - Simplified ViT-DET projection
        # srcs = []
        # if self.fusion is None:
        #     # Applying only projection for the four scale feature maps.
        #     for l, src in enumerate(features):
        #         srcs.append(self.input_proj[l](src))
        # else:
        #     # Applying single FPN (only using the last stage feature maps)
        #     srcs = self.fusion(features[-1])
        srcs = self.fusion(features)

        # generate pos encoding and pad mask for attention
        pos, masks = [], []
        for src in srcs:
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            #pos.append(self.pos_encoding(src, _mask))
        ##################################


        select_id = list(range(self.zeroshot_w.shape[-1]))  # 65 for MS-COCO
        
        query_embeds = self.query_embed.weight

        dtype = self.patch2query.weight.dtype
        clip_query_ori = self.zeroshot_w.t().type(dtype)

        clip_query = self.patch2query(clip_query_ori)

        (hs, init_reference, inter_references, _, enc_token_class_unflat), _ = \
            self.transformer(srcs, masks, query_embeds, text_query=clip_query)

        outputs_classes = []
        outputs_coords = []
        outputs_det_tokens = []
        outputs_embeds = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class, projected_hs = self.class_embed[lvl](hs[lvl], clip_query_ori)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_classes.append(outputs_class)
            outputs_det_tokens.append(hs[lvl])
            outputs_embeds.append(projected_hs)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_det_token = torch.stack(outputs_det_tokens)
        if self.distil_clip_embed:
            outputs_embed = torch.stack(outputs_embeds)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "select_id": select_id,
            "clip_query": clip_query_ori,
        }

        if self.distil_clip_embed:
            out["pred_embed"] = outputs_embed[-1]

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
            if self.distil_clip_embed:
                for temp, embed, det_token in zip(out["aux_outputs"], outputs_embed[:-1], outputs_det_tokens[:-1]):
                    temp["select_id"] = select_id
                    temp["pred_embed"] = embed
                    temp["clip_query"] = clip_query_ori
            else:
                for temp, det_token in zip(out["aux_outputs"], outputs_det_tokens[:-1]):
                    temp["select_id"] = select_id
                    temp["clip_query"] = clip_query_ori

        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.iou_aware:
            outputs_ious = []
            for lvl in range(hs.shape[0]):
                outputs_ious.append(self.iou_embed[lvl](hs[lvl]))
            outputs_iou = torch.stack(outputs_ious)
            out['pred_ious'] = outputs_iou[-1]

            if self.aux_loss:
                for i, aux in enumerate(out['aux_outputs']):
                    aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

        return out, select_id

def build(args):
    if args.dataset_file == "open_coco":
        seen_list = COCO_SEEN_CATEGORIES
    elif args.dataset_file == "open_lvis":
        seen_list = LVIS_SEEN_CATEGORIES
    elif args.dataset_file == "open_vidvrd":
        seen_list = VidVRD_SEEN_CATEGORIES
    elif args.dataset_file == "open_vidor":
        seen_list = VidOR_SEEN_CATEGORIES
    else:
        raise NotImplementedError

    device = torch.device(args.device)

    # backbone
    # if args.backbone_name == 'local_deit_base':
    #     backbone, hidden_dim = local_deit_base(pretrained=args.pre_trained)
    # else:
    #     raise ValueError(f'backbone {args.backbone_name} not supported')
    backbone = build_backbone(args)
    # return only last feature map.
    # backbone.finetune_det(out_indices=[2, 4, 10, 12])
    cross_scale_fusion = None
    if args.cross_scale_fusion:

        # ViT-DeT Simple FPN
        cross_scale_fusion = SimpleFeaturePyramid(
            in_feature=backbone.embed_dim,
            out_channels=args.reduced_dim,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            top_block=LastLevelMaxPool(),
            norm="LN",
            square_pad=1024,
        )
    pos_embed_layer = build_position_encoding(args)
    ################################

    transformer = build_deforamble_transformer(args)

    if args.dataset_file == "open_coco":
        # zeroshot_w = build_text_embedding_coco('ViT-B/32')
        zeroshot_w = build_text_embedding_coco(args.clip_backbone)
    elif args.dataset_file == "open_lvis":
        # zeroshot_w = build_text_embedding_lvis('ViT-B/32')
        zeroshot_w = build_text_embedding_lvis(args.clip_backbone)
    elif args.dataset_file == "open_vidvrd":
        # zeroshot_w = build_text_embedding_lvis('ViT-B/32')
        zeroshot_w = build_text_embedding_vidvrd(args.clip_backbone)
    elif args.dataset_file == "open_vidor":
        # zeroshot_w = build_text_embedding_lvis('ViT-B/32')
        zeroshot_w = build_text_embedding_vidor(args.clip_backbone)
    else:
        raise NotImplementedError
    model = OVDETR(
        backbone,
        transformer,
        pos_embed_layer,
        num_queries=args.det_token_num,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        cross_scale_fusion=cross_scale_fusion,
        zeroshot_w=zeroshot_w,
        clip_feat_path=args.clip_feat_path,
        prob=args.prob,
        distil_clip_embed=args.distil_clip_embed,
        seen_list=seen_list if args.all_train_token else None,
        #
        iou_aware=args.iou_aware,
        token_label=args.token_label,
    )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None), use_adapter=args.use_adapter)

    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    weight_dict["loss_embed"] = args.feature_loss_coef

    if args.iou_aware:
        weight_dict['loss_iouaware'] = args.iouaware_loss_coef
    
    if args.new_loss:
        weight_dict['loss_new'] = args.new_loss_coef

    if args.token_label:
        weight_dict['loss_token_focal'] = args.token_loss_coef
        weight_dict['loss_token_dice'] = args.token_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # 이것도 ablation 해야함. - embed가 필요한가? >> do ablation.
    if args.distil_clip_embed:
        print('clip embed distillation is enabled.')

    losses = ["labels", "boxes", "embed"] if args.distil_clip_embed else ["labels", "boxes"]
    if args.masks:
        # losses = ["labels", "boxes", "masks", "embed"] if args.distil_clip_embed else ["labels", "boxes", "masks"]
        losses = ['masks']

    if args.iou_aware:
        losses += ['iouaware']
    
    if args.new_loss:
        losses += ['newloss']

    criterion = OVSetCriterion(
        matcher,
        weight_dict,
        losses,
        focal_alpha=args.focal_alpha,
    )
    postprocessors = {}
    postprocessors["bbox"] = OVPostProcess(num_queries=args.det_token_num, dataset_file=args.dataset_file,
                                           clip_backbone=args.clip_backbone, temperature=args.temperature,
                                           size=(args.clip_h, args.clip_w),
                                           bg=args.bg)
    criterion.to(device)
    postprocessors = {key: postprocessor.to(device) for key, postprocessor in postprocessors.items()}

    if args.masks:
        postprocessors["segm"] = PostProcessSegm()

    return model, criterion, postprocessors
