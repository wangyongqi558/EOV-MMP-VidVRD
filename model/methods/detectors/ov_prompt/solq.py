"""
Deformable DETR model and criterion classes.
"""
import copy
import functools
import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from util.detectron2.layers import paste_masks_in_image
from util.detectron2.structures import BitMasks
from util.detectron2.utils.memory import retry_if_cuda_oom
import torchvision.transforms.functional as TF
from torch import nn

from methods.backbones.clip_deit import deit_base as clip_deit_base
from methods.backbones.clip_vit_det import local_deit_base as clip_local_deit_base
from methods.backbones.deit import deit_tiny, deit_small, deit_base, deit_base_distil
from methods.backbones.vit_det import local_deit_base
from methods.backbones.vit_det import local_deit_tiny as local_deit_tiny
from methods.fpn_fusion import SimpleFeaturePyramid, LastLevelMaxPool
from methods.segmentation import (PostProcessPanoptic,
                                  sigmoid_focal_loss, dice_loss)
from util import box_ops
from util.clip_utils import build_text_embedding_coco, build_text_embedding_lvis
from util.coco_categories import SEEN_CATEGORIES as COCO_SEEN_CATEGORIES
from util.coco_categories import UNSEEN_CATEGORIES as COCO_UNSEEN_CATEGORIES
from util.detectron2.structures import Boxes
from util.lvis_v1_categories import SEEN_CATEGORIES as LVIS_SEEN_CATEGORIES
from util.lvis_v1_categories import UNSEEN_CATEGORIES as LVIS_UNSEEN_CATEGORIES
from util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized)
from .dct import ProcessorDCT
from .deformable_transformer import build_deforamble_transformer
from .matcher import build_matcher
from .model import CLIP_Classifier
from .position_encoding import build_position_encoding
from .post_process import Aux_CLIP_Classifier

print = functools.partial(print, flush=True)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SOLQ(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, with_vector=False, processor_dct=None,
                 vector_hidden_dim=256, cross_scale_fusion=None, clip_feat_path=None, distil_clip_embed=False,
                 seen_list=None, iou_aware=False, token_label=False, zeroshot_w=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.zeroshot_w = zeroshot_w.t()

        self.patch2query = nn.Linear(self.zeroshot_w.shape[0], 256)
        self.patch2query_img = nn.Linear(self.zeroshot_w.shape[0], 256)
        self.distil_clip_embed = distil_clip_embed
        for layer in [self.patch2query, self.patch2query_img]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

        self.with_vector = with_vector
        self.processor_dct = processor_dct
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.processor_dct.n_keep, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if cross_scale_fusion is None:
            if num_feature_levels > 1:
                num_backbone_outs = len(backbone.num_channels)
                input_proj_list = []
                for _ in range(num_backbone_outs):
                    in_channels = backbone.num_channels[_]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                for _ in range(num_feature_levels - num_backbone_outs):
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                    in_channels = hidden_dim
                self.input_proj = nn.ModuleList(input_proj_list)
            else:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )])
        else:
            self.fusion = cross_scale_fusion

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.clip_feat = torch.load(clip_feat_path)
        self.all_ids = torch.tensor(range(self.zeroshot_w.shape[-1]))
        self.prob = 0.75

        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.class_embed = CLIP_Classifier(256, self.zeroshot_w.shape[0], clip_distil=distil_clip_embed)
        self.seen_list = seen_list

        # additional losses for VIDT+
        self.iou_aware = iou_aware
        self.token_label = token_label

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)
        if cross_scale_fusion is None:
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # IoU Aware
        if self.iou_aware:
            self.iou_embed = MLP(256, 256, 1, 3)
            if with_box_refine:
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            else:
                self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])

    def forward(self, samples: NestedTensor, targets=None, criterion=None):
        if self.training:
            return self.forward_train(samples, targets, criterion)
        else:
            return self.forward_test(samples)

    def forward_train(self, samples: NestedTensor, targets=None, criterion=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        # transform to nested tensor
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        x = samples.tensors
        mask = samples.mask
        features = self.backbone(x)

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
            # print(mask[None].float().shape)
            # mask = mask.unsqueeze(dim=1)
            # print(mask.unsqueeze(1).shape)
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            # pos.append(self.pos_encoding(src, _mask))
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

        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, _, _ = self.transformer(srcs,
        #                                                                                                           masks,
        #                                                                                                           pos,
        #                                                                                                           query_embeds)
        (hs, init_reference, inter_references, clip_query, enc_token_class_unflat), _ = \
            self.transformer(srcs, masks, query_embeds, text_query=clip_query)  # text query -> new part

        outputs_coords = []
        outputs_det_tokens = []
        outputs_embeds = []
        outputs_classes = []
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
            outputs_det_tokens.append(hs[lvl])

            # new
            outputs_classes.append(outputs_class)
            outputs_embeds.append(projected_hs)

        outputs_coord = torch.stack(outputs_coords)
        outputs_det_tokens = torch.stack(outputs_det_tokens)
        outputs_class = torch.stack(outputs_classes)
        if self.distil_clip_embed:
            outputs_embed = torch.stack(outputs_embeds)

        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               "select_id": select_id,
               "clip_query": clip_query_ori}

        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})

        if self.distil_clip_embed:
            out["pred_embed"] = outputs_embed[-1]

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord, outputs_vector)
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

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vector):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.distil_clip_embed:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_vectors': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_vector[:-1])]

    def forward_test(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        ##############################################
        x = samples.tensors  # RGB input
        mask = samples.mask  # padding mask

        # input normalization
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        # return multi-scale [PATCH] tokens
        features = self.backbone(x)  # deit는 attention masking 안 들어있음.

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
            # pos.append(self.pos_encoding(src, _mask))
        ##################################

        select_id = list(range(self.zeroshot_w.shape[-1]))  # 65 for MS-COCO

        query_embeds = self.query_embed.weight

        dtype = self.patch2query.weight.dtype
        clip_query_ori = self.zeroshot_w.t().type(dtype)

        clip_query = self.patch2query(clip_query_ori)

        (hs, init_reference, inter_references, _, enc_token_class_unflat), _ = \
            self.transformer(srcs, masks, query_embeds, text_query=clip_query)  # text query -> new part

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

        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "select_id": select_id,
            "clip_query": clip_query_ori,
        }

        if self.distil_clip_embed:
            out["pred_embed"] = outputs_embed[-1]

        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord, outputs_vector)
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

        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25,
                 with_vector=False,
                 processor_dct=None,
                 vector_loss_coef=0.7,
                 no_vector_loss_norm=False,
                 vector_start_stage=0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher, self.matcher_ori = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        self.vector_loss_coef = vector_loss_coef
        self.no_vector_loss_norm = no_vector_loss_norm
        self.vector_start_stage = vector_start_stage
        self.eos_coef = 0.1

        print(f'Training with {6 - self.vector_start_stage} vector stages.')

        print(f"Training with vector_loss_coef {self.vector_loss_coef}.")

        if not self.no_vector_loss_norm:
            print('Training with vector_loss_norm.')

    def loss_labels(self, outputs, targets, indices, num_boxes, select_id, log=False):

        assert 'pred_logits' in outputs

        src_logits = outputs['pred_logits']
        num_classes = src_logits.shape[-1]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # conversion : label to index.
        label_to_idx = {label: id for id, label in enumerate(select_id)}
        target_classes_o = torch.tensor([label_to_idx[label] for label in target_classes_o.cpu().numpy()]).to(
            target_classes_o)
        #
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        # loss_ce = asym_focal_loss(src_logits, target_classes_onehot, num_boxes,
        #                          gamma_pos=0.0, gamma_neg=4.0, clip=0.00) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        # if log:
        # TODO this should probably be a separate loss, not hacked in this one here
        losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, select_id):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, select_id):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_embed(self, outputs, targets, indices, num_boxes, select_id):
        idx = self._get_src_permutation_idx(indices)
        # < batch id , box id> -> ith target

        src_feature = outputs["pred_embed"][idx]

        select_id = torch.tensor(outputs["select_id"]).to(src_feature.device)  # [ 1 3 4] [ 3 3 3 3 1 1  3 3]
        clip_query = outputs["clip_query"]
        target_feature = []
        for t, (_, i) in zip(targets, indices):
            for c in t["labels"][i]:
                index = (select_id == c).nonzero(as_tuple=False)[0]
                target_feature.append(clip_query[index])
        target_feature = torch.cat(target_feature, dim=0)

        # l2 normalize the feature
        src_feature = nn.functional.normalize(src_feature, dim=1)
        loss_feature = F.mse_loss(src_feature, target_feature, reduction="none")
        losses = {"loss_embed": loss_feature.sum() / num_boxes}
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, select_id):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_vectors" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_vectors"]
        src_boxes = outputs['pred_boxes']
        # TODO use valid to mask invalid areas due to padding in loss
        target_boxes = torch.cat([t['xyxy_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)
        src_vectors = src_masks[src_idx]
        src_boxes = src_boxes[src_idx]
        target_masks = target_masks[tgt_idx]

        # crop gt_masks
        n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
        gt_masks = BitMasks(target_masks)
        gt_masks = gt_masks.crop_and_resize(target_boxes, gt_mask_len).to(device=src_masks.device).float()
        target_masks = gt_masks

        if target_masks.shape[0] == 0:
            losses = {
                "loss_vector": src_vectors.sum() * 0
            }
            return losses

        # perform dct transform
        target_vectors = []
        for i in range(target_masks.shape[0]):
            gt_mask_i = ((target_masks[i, :, :] >= 0.5) * 1).to(dtype=torch.uint8)
            gt_mask_i = gt_mask_i.cpu().numpy().astype(np.float32)
            coeffs = cv2.dct(gt_mask_i)
            coeffs = torch.from_numpy(coeffs).flatten()
            coeffs = coeffs[torch.tensor(self.processor_dct.zigzag_table)]
            gt_label = coeffs.unsqueeze(0)
            target_vectors.append(gt_label)

        target_vectors = torch.cat(target_vectors, dim=0).to(device=src_vectors.device)
        losses = {}
        if self.no_vector_loss_norm:
            losses['loss_vector'] = self.vector_loss_coef * F.l1_loss(src_vectors, target_vectors,
                                                                      reduction='none').sum() / num_boxes
        else:
            losses['loss_vector'] = self.vector_loss_coef * F.l1_loss(src_vectors, target_vectors, reduction='mean')
        return losses

    def loss_iouaware(self, outputs, targets, indices, num_boxes, select_id):
        assert 'pred_ious' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ious = outputs['pred_ious'][idx]  # logits
        src_ious = src_ious.squeeze(1)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        iou = torch.diag(box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))[0])

        losses = {}
        loss_iouaware = F.binary_cross_entropy_with_logits(src_ious, iou, reduction='none')
        losses['loss_iouaware'] = loss_iouaware.sum() / num_boxes
        return losses

    def loss_tokens(self, outputs, targets, num_boxes):
        enc_token_class_unflat = outputs['pred_logits']

        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()

        bs, n, h, w = target_masks.shape
        mask = torch.zeros((bs, h, w), dtype=torch.bool, device=target_masks.device)
        for j in range(n):
            target_masks[:, j] &= target_masks[:, j] ^ mask
            mask |= target_masks[:, j]
        target_classes_pad = torch.stack([F.pad(t['labels'], (0, n - len(t['labels']))) for t in targets])
        final_mask = torch.sum(target_masks * target_classes_pad[:, :, None, None], dim=1)  # (bs, h, w)

        num_classes = enc_token_class_unflat.shape[-1]
        final_mask_onehot = torch.zeros((bs, h, w, num_classes), dtype=torch.float32, device=target_masks.device)
        final_mask_onehot.scatter_(-1, final_mask.unsqueeze(-1), 1)  # (bs, h, w, 91)

        final_mask_onehot[..., 0] = 1 - final_mask_onehot[..., 0]  # change index 0 from background to foreground

        loss_token_focal = 0
        loss_token_dice = 0
        for i, enc_token_class in enumerate(enc_token_class_unflat):
            _, h, w, _ = enc_token_class.shape

            final_mask_soft = F.adaptive_avg_pool2d(final_mask_onehot.permute(0, 3, 1, 2), (h, w)).permute(0, 2, 3, 1)

            enc_token_class = enc_token_class.flatten(1, 2)
            final_mask_soft = final_mask_soft.flatten(1, 2)
            loss_token_focal += sigmoid_focal_loss(enc_token_class, final_mask_soft, num_boxes)
            loss_token_dice += dice_loss(enc_token_class, final_mask_soft, num_boxes)

        losses = {
            'loss_token_focal': loss_token_focal,
            'loss_token_dice': loss_token_dice,
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, select_id, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            "embed": self.loss_embed,
            'iouaware': self.loss_iouaware,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, select_id, **kwargs)

    def forward(self, outputs, targets):

        '''
        if not self.training:
            return {
                "loss_ce": outputs["pred_logits"].sum() * 0.0,
                "class_error": outputs["pred_logits"].sum() * 0.0,
            }
        '''

        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        masks = []
        for t in targets:
            mask = t["labels"] == -2
            for ind, v in enumerate(t["labels"]):
                if v in outputs["select_id"]:
                    mask[ind] = True
            masks.append(mask)
        num_boxes = sum(len(t["labels"][m]) for t, m in zip(targets, masks))
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Retrieve the matching between the outputs of the last layer and the targets
        select_id = outputs["select_id"]
        indices = self.matcher(outputs_without_aux, targets, select_id)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, select_id, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, select_id)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, select_id, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher_ori(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == "masks" or loss == "embed":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'enc_tokens' in outputs:
            l_dict = self.loss_tokens(outputs['enc_tokens'], targets, num_boxes)
            losses.update(l_dict)

        return losses


'''
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        select_id = outputs["select_id"]
        indices = self.matcher(outputs_without_aux, targets, select_id)
        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, select_id, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, select_id)
                for loss in self.losses:
                    if loss == 'masks' and i < self.vector_start_stage:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, select_id, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'enc_tokens' in outputs:
            l_dict = self.loss_tokens(outputs['enc_tokens'], targets, num_boxes)
            losses.update(l_dict)
        return losses
'''


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_queries=300, dataset_file='open_coco',
                 size=(420, 420),
                 clip_backbone=None,
                 temperature=0.06,
                 bg=True,
                 pruning_threshold=0.225,
                 processor_dct=None):
        super().__init__()
        self.num_queries = num_queries

        if dataset_file == "open_coco":
            self.seen_list = COCO_SEEN_CATEGORIES
            self.unseen_list = COCO_UNSEEN_CATEGORIES
        elif dataset_file == "open_lvis":
            self.seen_list = LVIS_SEEN_CATEGORIES
            self.unseen_list = LVIS_UNSEEN_CATEGORIES

        # aux clip cls
        if clip_backbone not in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']:
            self.aux_classifier = None
        else:
            self.aux_classifier = Aux_CLIP_Classifier(dataset_file, clip_backbone, size, temperature, bg=bg).to()
        self.pruning_threshold = pruning_threshold
        self.size = size

        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, samples, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, out_vector = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_vectors']
        # print(f"logits: {out_logits.shape}")
        # print(f"out_bbox: {out_bbox.shape}")
        # print(f"out_vector: {out_vector.shape}")
        # print(outputs['select_id'])
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        bs = out_bbox.shape[0]
        torch.save(out_bbox, "before_pick.pt")
        if self.aux_classifier is None:
            # prob = out_logits.sigmoid()
            # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
            # scores = topk_values
            # topk_boxes = topk_indexes // out_logits.shape[2]
            # labels = topk_indexes % out_logits.shape[2]
            # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
            # # if self.processor_dct is not None:
            # #     n_keep = self.processor_dct.n_keep
            # #     vectors = torch.gather(out_vector, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, n_keep))
            #
            # # and from relative [0, 1] to absolute [0, height] coordinates
            # img_h, img_w = target_sizes.unbind(1)
            # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            # boxes = boxes * scale_fct[:, None, :]
            # boxes_list = boxes
            # set_index = [[i for i in range(boxes.size(1))] for _ in range(boxes.size(0))]
            # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

            # seen and unseen ids
            unseen_list = self.unseen_list
            seen_list = self.seen_list
            revert_idx = np.argsort(seen_list + unseen_list)

            prob = out_logits.sigmoid()

            # prunning
            max_prob, _ = (prob).max(dim=-1)
            masking = torch.gt(max_prob, self.pruning_threshold)

            # zip the prunned boxes
            clip_bbox = box_ops.box_cxcywh_to_xyxy(out_bbox)
            img_h, img_w = samples.target_sizes.unbind(1)  # batch input resolution
            h_fct, w_fct = float(self.size[0]) / img_h, float(self.size[1]) / img_w
            scale_fct = torch.stack([img_w * w_fct, img_h * h_fct, img_w * w_fct, img_h * h_fct], dim=1)
            # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            clip_bbox = clip_bbox * scale_fct[:, None, :]

            # det prediction for seen prob
            seen_prob = prob[..., self.seen_list]
            unseen_prob = prob[..., self.unseen_list]

            roi_box = []
            eval_box = []
            bs_indice = []
            det_seen_prob = []
            det_unseen_prob = []
            increment = 0

            det_total_prob = []
            set_index = []
            for idx in range(bs):
                indice = (masking[idx] == True).nonzero(as_tuple=True)[0]
                if len(indice) == 0:
                    # if no selected box, then pick the best one.
                    max, _ = prob[idx].max(dim=-1)
                    _, indice = max.max(dim=0)
                    box = clip_bbox[idx][indice].unsqueeze(0)
                    det_seen_prob.append(seen_prob[idx][indice].unsqueeze(0))  # det prob for selected boxes
                    det_unseen_prob.append(unseen_prob[idx][indice].unsqueeze(0))  # det prob for selected boxes
                    roi_box.append(Boxes(box))  # roi for selected boxes (resized resolution)
                    bs_indice.append(
                        range(increment, box.shape[0] + increment))  # indices for selected boxes per sample
                    eval_box.append(out_bbox[idx][indice].unsqueeze(0))  # for evaluation (original resolution)
                    det_total_prob.append(prob[idx][indice].unsqueeze(0))
                    increment += box.shape[0]
                    set_index.append([indice])
                else:
                    box = clip_bbox[idx][indice]
                    det_seen_prob.append(seen_prob[idx][indice])  # det prob for selected boxes
                    det_unseen_prob.append(unseen_prob[idx][indice])  # det prob for selected boxes
                    roi_box.append(Boxes(box))  # roi for selected boxes (resized resolution)
                    bs_indice.append(
                        range(increment, box.shape[0] + increment))  # indices for selected boxes per sample
                    eval_box.append(out_bbox[idx][indice])  # for evaluation (original resolution)
                    det_total_prob.append(prob[idx][indice])
                    increment += box.shape[0]
                    set_index.append(indice)

            results = []
            topk_boxes = []
            boxes_list = []

            for det_seen_p, det_unseen_p, box, target_size, det_total_p,si \
                    in zip(det_seen_prob, det_unseen_prob,  eval_box, target_sizes,
                           det_total_prob,  set_index):
                seen_det_prob = det_seen_p
                seen_det_prob = torch.nn.functional.pad(seen_det_prob, (0, len(unseen_list), 0, 0), value=0.0)
                unseen_det_prob = det_unseen_p
                unseen_det_prob = torch.nn.functional.pad(unseen_det_prob, (len(seen_list), 0, 0, 0), value=0.0)

                seen_prob = seen_det_prob
                unseen_prob = unseen_det_prob
                _new_prob = seen_prob + unseen_prob
                _new_prob = _new_prob[..., revert_idx]

                num_sel_boxes = det_unseen_p.shape[0]
                max_sel = num_sel_boxes * out_logits.shape[2]
                num_sel = min(max_sel, 300)

                topk_values, topk_indexes = torch.topk(_new_prob.reshape(-1), num_sel, dim=0)

                scores = topk_values
                topk_box = topk_indexes // out_logits.shape[2]
                labels = topk_indexes % out_logits.shape[2]

                boxes = box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0))
                boxes = torch.gather(boxes, 1, topk_box.unsqueeze(-1).repeat(1, 1, 4)).squeeze(0)
                topk_boxes.append(topk_box)
                # and from relative [0, 1] to absolute [0, height] coordinates
                img_h, img_w = target_size.unbind(0)
                scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(boxes)
                boxes = boxes * scale_fct
                boxes_list.append(boxes)
                results.append({'scores': scores, 'labels': labels, 'boxes': boxes})

            boxes_list = torch.stack(boxes_list)
            topk_boxes = torch.stack(topk_boxes)
        else:
            # seen and unseen ids
            unseen_list = self.unseen_list
            seen_list = self.seen_list
            revert_idx = np.argsort(seen_list + unseen_list)

            prob = out_logits.sigmoid()

            # prunning
            max_prob, _ = (prob).max(dim=-1)
            masking = torch.gt(max_prob, self.pruning_threshold)

            # zip the prunned boxes
            clip_bbox = box_ops.box_cxcywh_to_xyxy(out_bbox)
            img_h, img_w = samples.target_sizes.unbind(1)  # batch input resolution
            h_fct, w_fct = float(self.size[0]) / img_h, float(self.size[1]) / img_w
            scale_fct = torch.stack([img_w * w_fct, img_h * h_fct, img_w * w_fct, img_h * h_fct], dim=1)

            clip_bbox = clip_bbox * scale_fct[:, None, :]

            # det prediction for seen prob
            seen_prob = prob[..., self.seen_list]
            unseen_prob = prob[..., self.unseen_list]

            roi_box = []
            eval_box = []
            bs_indice = []
            det_seen_prob = []
            det_unseen_prob = []
            increment = 0

            det_total_prob = []
            set_index = []
            for idx in range(bs):
                indice = (masking[idx] == True).nonzero(as_tuple=True)[0]
                if len(indice) == 0:
                    # if no selected box, then pick the best one.
                    max, _ = prob[idx].max(dim=-1)
                    _, indice = max.max(dim=0)
                    box = clip_bbox[idx][indice].unsqueeze(0)
                    det_seen_prob.append(seen_prob[idx][indice].unsqueeze(0))  # det prob for selected boxes
                    det_unseen_prob.append(unseen_prob[idx][indice].unsqueeze(0))  # det prob for selected boxes
                    roi_box.append(Boxes(box))  # roi for selected boxes (resized resolution)
                    bs_indice.append(
                        range(increment, box.shape[0] + increment))  # indices for selected boxes per sample
                    eval_box.append(out_bbox[idx][indice].unsqueeze(0))  # for evaluation (original resolution)
                    det_total_prob.append(prob[idx][indice].unsqueeze(0))
                    increment += box.shape[0]
                    set_index.append([indice])
                else:
                    box = clip_bbox[idx][indice]
                    det_seen_prob.append(seen_prob[idx][indice])  # det prob for selected boxes
                    det_unseen_prob.append(unseen_prob[idx][indice])  # det prob for selected boxes
                    roi_box.append(Boxes(box))  # roi for selected boxes (resized resolution)
                    bs_indice.append(
                        range(increment, box.shape[0] + increment))  # indices for selected boxes per sample
                    eval_box.append(out_bbox[idx][indice])  # for evaluation (original resolution)
                    det_total_prob.append(prob[idx][indice])
                    increment += box.shape[0]
                    set_index.append(indice)
            torch.save(eval_box, "after_pick.pt")
            assert False
            aux_prob = self.aux_classifier(samples, roi_box)[..., :-1]
            # clip aux prob for unseen classes
            aux_total_prob = [aux_prob[bs_index] for bs_index in bs_indice]
            aux_seen_prob = aux_prob[..., self.seen_list]
            aux_seen_prob = [aux_seen_prob[bs_index] for bs_index in bs_indice]
            aux_unseen_prob = aux_prob[..., self.unseen_list]
            aux_unseen_prob = [aux_unseen_prob[bs_index] for bs_index in bs_indice]

            results = []
            topk_boxes = []
            boxes_list = []

            for det_seen_p, det_unseen_p, aux_seen_p, aux_unseen_p, box, target_size, det_total_p, aux_total_p, si \
                    in zip(det_seen_prob, det_unseen_prob, aux_seen_prob, aux_unseen_prob, eval_box, target_sizes,
                           det_total_prob, aux_total_prob, set_index):
                seen_det_prob = det_seen_p
                seen_det_prob = torch.nn.functional.pad(seen_det_prob, (0, len(unseen_list), 0, 0), value=0.0)
                unseen_det_prob = det_unseen_p
                unseen_det_prob = torch.nn.functional.pad(unseen_det_prob, (len(seen_list), 0, 0, 0), value=0.0)
                seen_clip_prob = aux_seen_p
                seen_clip_prob = torch.nn.functional.pad(seen_clip_prob, (0, len(unseen_list), 0, 0), value=0.0)
                unseen_clip_prob = aux_unseen_p
                unseen_clip_prob = torch.nn.functional.pad(unseen_clip_prob, (len(seen_list), 0, 0, 0), value=0.0)

                alpha = 0.2 # seen
                beta = 0.4 # unseen
                seen_prob = (seen_det_prob * (1.0 - alpha)) + (seen_clip_prob * (alpha))
                unseen_prob = (unseen_det_prob * (1.0 - beta)) + (unseen_clip_prob * (beta))
                _new_prob = seen_prob + unseen_prob
                _new_prob = _new_prob[..., revert_idx]

                num_sel_boxes = det_unseen_p.shape[0]
                max_sel = num_sel_boxes * out_logits.shape[2]
                num_sel = min(max_sel, 300)

                topk_values, topk_indexes = torch.topk(_new_prob.reshape(-1), num_sel, dim=0)

                scores = topk_values
                topk_box = topk_indexes // out_logits.shape[2]
                labels = topk_indexes % out_logits.shape[2]

                boxes = box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0))
                boxes = torch.gather(boxes, 1, topk_box.unsqueeze(-1).repeat(1, 1, 4)).squeeze(0)
                topk_boxes.append(topk_box)
                # and from relative [0, 1] to absolute [0, height] coordinates
                img_h, img_w = target_size.unbind(0)
                scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(boxes)
                boxes = boxes * scale_fct
                boxes_list.append(boxes)
                results.append({'scores': scores, 'labels': labels, 'boxes': boxes})

            boxes_list = torch.stack(boxes_list)
            topk_boxes = torch.stack(topk_boxes)

        if self.processor_dct is not None:
            img_h, img_w = target_sizes.unbind(1)
            n_keep = self.processor_dct.n_keep
            # vectors = torch.gather(out_vector, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, n_keep))
            masks = []
            n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
            # b, r, c = vectors.shape

            for bi in range(bs):
                vectors = out_vector[bi][set_index[bi]]
                outputs_masks_per_image = []
                for ri in range(len(vectors)):
                    # here visual for training
                    idct = np.zeros((gt_mask_len ** 2))
                    idct[:n_keep] = vectors[ri].cpu().numpy()
                    idct = self.processor_dct.inverse_zigzag(idct, gt_mask_len, gt_mask_len)
                    re_mask = cv2.idct(idct)
                    max_v = np.max(re_mask)
                    min_v = np.min(re_mask)
                    re_mask = np.where(re_mask > (max_v + min_v) / 2., 1, 0)
                    re_mask = torch.from_numpy(re_mask)[None].float()
                    outputs_masks_per_image.append(re_mask)
                outputs_masks_per_image = torch.cat(outputs_masks_per_image, dim=0).to(out_vector.device)
                outputs_masks_per_image = torch.stack([outputs_masks_per_image[k] for k in topk_boxes[bi]])

                # here padding local mask to global mask
                outputs_masks_per_image = retry_if_cuda_oom(paste_masks_in_image)(
                    outputs_masks_per_image,  # N, 1, M, M
                    boxes_list[bi],
                    (img_h[bi], img_w[bi]),
                    threshold=0.5,
                )
                # outputs_masks_per_image = outputs_masks_per_image.unsqueeze(1).cpu()
                # outputs_masks_per_image = outputs_masks_per_image.cpu()
                masks.append(outputs_masks_per_image)


        if self.processor_dct is None:
            results = [{'scores': r['scores'], 'labels': r['labels'], 'boxes': r['boxes']} for r in results]
        else:
            results = [{'scores': r['scores'], 'labels': r['labels'], 'boxes': r['boxes'], 'masks': m} for r, m in
                       zip(results, masks)]

        return results, None, None


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5, processor_dct=None):
        super().__init__()
        self.threshold = threshold
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes, topk_boxes, aux_classifier):
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.dataset_file == "open_coco":
        seen_list = COCO_SEEN_CATEGORIES
    elif args.dataset_file == "open_lvis":
        seen_list = LVIS_SEEN_CATEGORIES
    else:
        raise NotImplementedError

    device = torch.device(args.device)

    # backbone
    if args.backbone_name == 'deit_tiny':
        backbone, hidden_dim = deit_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'deit_small':
        backbone, hidden_dim = deit_small(pretrained=args.pre_trained)
    elif args.backbone_name == 'deit_base':
        backbone, hidden_dim = deit_base(pretrained=args.pre_trained)
    elif args.backbone_name == 'deit_base_distil':
        backbone, hidden_dim = deit_base_distil(pretrained=args.pre_trained)
    elif args.backbone_name == 'local_deit_tiny':
        backbone, hidden_dim = local_deit_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'local_deit_base':
        backbone, hidden_dim = local_deit_base(pretrained=args.pre_trained)
    elif args.backbone_name == 'clip_deit_base':
        backbone, hidden_dim = clip_deit_base(pretrained=args.pre_trained)
    elif args.backbone_name == 'clip_local_deit_base':
        backbone, hidden_dim = clip_local_deit_base(pretrained=args.pre_trained)

    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    # return only last feature map.
    backbone.finetune_det(out_indices=[2, 4, 10, 12])

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
        zeroshot_w = build_text_embedding_coco('ViT-L/14@336px')
        # zeroshot_w = build_text_embedding_coco('ViT-B/32')
        # zeroshot_w = build_text_embedding_coco(args.clip_backbone)
        num_classes = 65
    elif args.dataset_file == "open_lvis":
        zeroshot_w = build_text_embedding_lvis('ViT-L/14@336px')
        # zeroshot_w = build_text_embedding_lvis('ViT-B/32')
        # zeroshot_w = build_text_embedding_lvis(args.clip_backbone)
        num_classes = 1203
    else:
        raise NotImplementedError

    device = torch.device(args.device)

    if args.with_vector:
        processor_dct = ProcessorDCT(args.n_keep, args.gt_mask_len)
    model = SOLQ(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        with_vector=args.with_vector,
        processor_dct=processor_dct if args.with_vector else None,
        vector_hidden_dim=args.vector_hidden_dim,
        cross_scale_fusion=cross_scale_fusion,
        distil_clip_embed=args.distil_clip_embed,
        clip_feat_path=args.clip_feat_path,
        seen_list=seen_list if args.all_train_token else None,
        iou_aware=args.iou_aware,
        token_label=args.token_label,
        zeroshot_w=zeroshot_w,
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.iou_aware:
        weight_dict['loss_iouaware'] = args.iouaware_loss_coef

    if args.token_label:
        weight_dict['loss_token_focal'] = args.token_loss_coef
        weight_dict['loss_token_dice'] = args.token_loss_coef

    if args.masks:
        weight_dict["loss_vector"] = 1

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.distil_clip_embed:
        print('clip embed distillation is enabled.')

    losses = ['labels', 'boxes', 'cardinality', 'embed'] if args.distil_clip_embed else ["labels", "boxes",
                                                                                         "cardinality"]
    if args.masks:
        losses += ["masks"]

    if args.iou_aware:
        losses += ['iouaware']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             with_vector=args.with_vector,
                             processor_dct=processor_dct if args.with_vector else None,
                             vector_loss_coef=args.vector_loss_coef,
                             no_vector_loss_norm=args.no_vector_loss_norm,
                             vector_start_stage=args.vector_start_stage)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_queries=args.det_token_num, dataset_file=args.dataset_file,
                                          clip_backbone=args.clip_backbone, temperature=args.temperature,
                                          size=(args.clip_h, args.clip_w),
                                          bg=args.bg, pruning_threshold=0.3,
                                          processor_dct=processor_dct if (args.with_vector and args.eval) else None)}

    if args.masks and args.eval:
        postprocessors['segm'] = PostProcessSegm(processor_dct=processor_dct if args.with_vector else None)
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
