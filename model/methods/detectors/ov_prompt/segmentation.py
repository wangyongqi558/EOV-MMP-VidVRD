"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms.functional as TF

from util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list


class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False, use_adapter=True):
        super().__init__()
        self.detr = detr
        self.use_adapter=use_adapter
        if freeze_detr:
            print("Freeze DETR")
            for p in self.parameters():
                # p.requires_grad = False
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [768, 768, 768], hidden_dim, self.use_adapter)

    def forward(self, samples: NestedTensor, targets=None, criterion=None):
        if self.training:
            return self.forward_train(samples, targets, criterion)
        else:
            return self.forward_test(samples)

    def forward_train(self, samples: NestedTensor, targets=None, criterion=None):
        with torch.no_grad():
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)

            x = samples.tensors
            mask = samples.mask
            features = self.detr.backbone(x)

            srcs = []
            if self.detr.fusion is None:
                # Applying only projection for the four scale feature maps.
                for l, src in enumerate(features):
                    srcs.append(self.detr.input_proj[l](src))
            else:
                # Applying single FPN (only using the last stage feature maps)
                srcs = self.detr.fusion(features[-1])

            # generate pos encoding and pad mask for attention
            pos, masks = [], []
            for src in srcs:
                _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                masks.append(_mask)
                # pos.append(self.pos_encoding(src, _mask))


            # new part.
            uniq_labels = torch.cat([t["labels"] for t in targets])
            uniq_labels = torch.unique(uniq_labels).to("cpu")
            uniq_labels = uniq_labels[torch.randperm(len(uniq_labels))]
            select_id = uniq_labels.tolist()

            if self.detr.seen_list is not None:
                select_id = list(set(uniq_labels + self.detr.seen_list))
            text_query = self.detr.zeroshot_w[:, select_id].t()

            img_query = []
            for cat_id in select_id:
                # takes one random target object clip embedding.
                index = torch.randperm(len(self.detr.clip_feat[cat_id]))[0:1]
                img_query.append(self.detr.clip_feat[cat_id][index])
            # to tensor.
            img_query = torch.cat(img_query).to(text_query.device)
            # transform to unit vector.
            img_query = img_query / img_query.norm(dim=-1, keepdim=True)

            # if < 0.75 (75%) -> bool -> float -> [len, 1]
            mask = (torch.rand(len(text_query)) < self.detr.prob).float().unsqueeze(1).to(text_query.device)
            # 75% text query + 25% img query
            clip_query_ori = (text_query * mask + img_query * (1 - mask)).detach()

            dtype = self.detr.patch2query.weight.dtype
            # projection = patch2query
            text_query = self.detr.patch2query(text_query.type(dtype))
            img_query = self.detr.patch2query_img(img_query.type(dtype))

            clip_query = text_query * mask + img_query * (1 - mask)

            # class agnostic tokens.
            query_embeds = self.detr.query_embed.weight

            (hs, init_reference, inter_references, clip_query, enc_token_class_unflat), memory = \
                self.detr.transformer(srcs, masks, query_embeds, text_query=clip_query)  # text query -> new part

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
                outputs_class, projected_hs = self.detr.class_embed[lvl](hs[lvl], clip_query_ori)

                tmp = self.detr.bbox_embed[lvl](hs[lvl])
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
            if self.detr.distil_clip_embed:
                outputs_embed = torch.stack(outputs_embeds)

            out = {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
                "select_id": select_id,
                "clip_query": clip_query_ori,
            }

            if self.detr.distil_clip_embed:
                out["pred_embed"] = outputs_embed[-1]

            if self.detr.aux_loss:
                out["aux_outputs"] = self.detr._set_aux_loss(outputs_class, outputs_coord)
                if self.detr.distil_clip_embed:
                    for temp, embed, det_token in zip(out["aux_outputs"], outputs_embed[:-1], outputs_det_tokens[:-1]):
                        temp["select_id"] = select_id
                        temp["pred_embed"] = embed
                        temp["clip_query"] = clip_query_ori
                else:
                    for temp, det_token in zip(out["aux_outputs"], outputs_det_tokens[:-1]):
                        temp["select_id"] = select_id
                        temp["clip_query"] = clip_query_ori

            # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
            if self.detr.iou_aware:
                outputs_ious = []
                for lvl in range(hs.shape[0]):
                    outputs_ious.append(self.detr.iou_embed[lvl](hs[lvl]))
                outputs_iou = torch.stack(outputs_ious)
                out['pred_ious'] = outputs_iou[-1]

                if self.detr.aux_loss:
                    for i, aux in enumerate(out['aux_outputs']):
                        aux['pred_ious'] = outputs_iou[i]

            # token label loss
            if self.detr.token_label:
                out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

            #
            # srcs = []
            # masks = []
            # for l, feat in enumerate(features):
            #     src, mask = feat.decompose()
            #     srcs.append(self.detr.input_proj[l](src))
            #     masks.append(mask)
            #     assert mask is not None
            # if self.detr.num_feature_levels > len(srcs):
            #     _len_srcs = len(srcs)
            #     for l in range(_len_srcs, self.detr.num_feature_levels):
            #         if l == _len_srcs:
            #             src = self.detr.input_proj[l](features[-1].tensors)
            #         else:
            #             src = self.detr.input_proj[l](srcs[-1])
            #         m = samples.mask
            #         mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            #         pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            #         srcs.append(src)
            #         masks.append(mask)
            #         pos.append(pos_l)

            # max_len = 20
            # uniq_labels = torch.cat([t["labels"] for t in targets])
            # uniq_labels = torch.unique(uniq_labels).to("cpu")
            # uniq_labels = uniq_labels[torch.randperm(len(uniq_labels))][:max_len]
            # select_id = uniq_labels.tolist()
            #
            # clip_query = self.detr.zeroshot_w[:, select_id].t()
            # clip_query = self.detr.patch2query(clip_query)
            #
            # query_embeds = None
            # if not self.detr.two_stage:
            #     query_embeds = self.detr.query_embed.weight
            # (
            #     hs,
            #     init_reference,
            #     inter_references,
            #     enc_outputs_class,
            #     enc_outputs_coord_unact,
            #     _,
            # ), memory = self.detr.transformer(srcs, masks, pos, query_embeds, text_query=clip_query)
            #
            # for lvl in [hs.shape[0] - 1]:
            #     if lvl == 0:
            #         reference = init_reference
            #     else:
            #         reference = inter_references[lvl - 1]
            #     reference = inverse_sigmoid(reference)
            #     outputs_class = self.detr.get_outputs_class(self.detr.class_embed[lvl], hs[lvl])
            #     tmp = self.detr.bbox_embed[lvl](hs[lvl])
            #     if reference.shape[-1] == 4:
            #         tmp += reference
            #     else:
            #         assert reference.shape[-1] == 2
            #         tmp[..., :2] += reference
            #     outputs_coord = tmp.sigmoid()
            # out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
            #
            # # FIXME h_boxes takes the last one computed, keep this in mind
            indices = criterion.matcher(out, targets, select_id)
            src_idx = criterion._get_src_permutation_idx(indices)
            hs_select = hs[-1][src_idx[0], src_idx[1], :]

        bbox_mask = self.bbox_attention(
            hs_select[
                None,
            ],
            memory[1],
            mask=masks[1],
        )
        # print(f"src[1]: {srcs[1].shape}")
        # print(f"bbox_mask: {bbox_mask.shape}")
        # print(f"features[2]: {features[2].shape}")
        # print(f"src index: {len(src_idx[0])}")
        max_length = 1500
        if bbox_mask.size(1) > max_length:
            # print(f">{max_length} bbox mask size: {bbox_mask.shape}")
            seg_masks = []
            iter = bbox_mask.size(1) // max_length if bbox_mask.size(1) % max_length == 0 else bbox_mask.size(1) // max_length + 1
            for i in range(iter):
                tmp_seg_masks = self.mask_head(
                    srcs[1], bbox_mask[:, max_length*i:max_length*(i+1), :, :, :], [features[1], features[0], features[0]]
                )
                seg_masks.append(tmp_seg_masks)
            seg_masks = torch.cat(seg_masks)
            del tmp_seg_masks
            # print(f">{max_length} SEG MASKs: {seg_masks.shape}")
        else:
            seg_masks = self.mask_head(
                srcs[1], bbox_mask, [features[1], features[0], features[0]]
            )
            # print(f"<{max_length} SEG MASKs: {seg_masks.shape}")
        # seg_masks = self.mask_head(
        #     srcs[1], bbox_mask, [features[2], features[1], features[0]]
        # )
        bs = features[-1].shape[0]
        outputs_seg_masks = seg_masks.view(
            bs, len(src_idx[0]), seg_masks.shape[-2], seg_masks.shape[-1]
        )
        out["pred_masks"] = outputs_seg_masks
        # out["select_id"] = outputs_class[-1].max(dim=-1)[1].squeeze()
        out["select_id"] = select_id
        del outputs_seg_masks
        del seg_masks
        return out

    def forward_test(self, samples: NestedTensor, targets=None, criterion=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        ##############################################
        x = samples.tensors  # RGB input
        mask = samples.mask  # padding mask

        # input normalization
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # return multi-scale [PATCH] tokens
        features = self.detr.backbone(x)  # deit는 attention masking 안 들어있음.

        # [PATCH] token projection - Simplified ViT-DET projection
        srcs = []
        if self.detr.fusion is None:
            # Applying only projection for the four scale feature maps.
            for l, src in enumerate(features):
                srcs.append(self.detr.input_proj[l](src))
        else:
            # Applying single FPN (only using the last stage feature maps)
            srcs = self.detr.fusion(features[-1])

        # generate pos encoding and pad mask for attention
        pos, masks = [], []
        for src in srcs:
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            # pos.append(self.pos_encoding(src, _mask))
        ##################################

        select_id = list(range(self.detr.zeroshot_w.shape[-1]))  # 65 for MS-COCO

        query_embeds = self.detr.query_embed.weight

        dtype = self.detr.patch2query.weight.dtype
        clip_query_ori = self.detr.zeroshot_w.t().type(dtype)

        clip_query = self.detr.patch2query(clip_query_ori)

        (hs, init_reference, inter_references, _, enc_token_class_unflat), memory = \
            self.detr.transformer(srcs, masks, query_embeds, text_query=clip_query)

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
            outputs_class, projected_hs = self.detr.class_embed[lvl](hs[lvl], clip_query_ori)

            tmp = self.detr.bbox_embed[lvl](hs[lvl])
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
        # outputs_det_token = torch.stack(outputs_det_tokens)
        bbox_mask = self.bbox_attention(hs[-1], memory[1], mask=masks[1])
        max_length = 500
        if bbox_mask.size(1) > max_length:
            # print(f">{max_length} bbox mask size: {bbox_mask.shape}")
            seg_masks = []
            iter = bbox_mask.size(1) // max_length if bbox_mask.size(1) % max_length == 0 else bbox_mask.size(
                1) // max_length + 1
            for i in range(iter):
                tmp_seg_masks = self.mask_head(
                    srcs[1], bbox_mask[:, max_length * i:max_length * (i + 1), :, :, :],
                    [features[1], features[0], features[0]]
                )
                seg_masks.append(tmp_seg_masks)
            del tmp_seg_masks
            seg_masks = torch.cat(seg_masks)
            # print(f">{max_length} SEG MASKs: {seg_masks.shape}")
        else:
            seg_masks = self.mask_head(
                srcs[1], bbox_mask, [features[1], features[0], features[0]]
            )
        del bbox_mask
        del srcs
        del features

            # print(f"<{max_length} SEG MASKs: {seg_masks.shape}")
        # outputs_seg_masks = self.mask_head(
        #     srcs[1], bbox_mask, [features[2], features[1], features[0]]
        # )

        # outputs_seg_masks = self.mask_head(
        #     srcs[1], bbox_mask, [features[1], features[0], features[0]]
        # )
        # outputs_seg_masks = seg_masks.view(
        #     bs, self.detr.num_queries * len(select_c), seg_masks.shape[-2], seg_masks.shape[-1]
        # )
        # outputs_seg_masks_list.append(outputs_seg_masks)
        # outputs_seg_masks = torch.cat(outputs_seg_masks_list, 1)

        if self.detr.distil_clip_embed:
            outputs_embed = torch.stack(outputs_embeds)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_masks": seg_masks,
            "select_id": outputs_class[-1].max(dim=-1)[1].squeeze(),
            "clip_query": clip_query_ori,
        }
        del seg_masks
        if self.detr.distil_clip_embed:
            out["pred_embed"] = outputs_embed[-1]

        if self.detr.aux_loss:
            out["aux_outputs"] = self.detr._set_aux_loss(outputs_class, outputs_coord)
            if self.detr.distil_clip_embed:
                for temp, embed, det_token in zip(out["aux_outputs"], outputs_embed[:-1], outputs_det_tokens[:-1]):
                    temp["select_id"] = select_id
                    temp["pred_embed"] = embed
                    temp["clip_query"] = clip_query_ori
            else:
                for temp, det_token in zip(out["aux_outputs"], outputs_det_tokens[:-1]):
                    temp["select_id"] = select_id
                    temp["clip_query"] = clip_query_ori

        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.detr.iou_aware:
            outputs_ious = []
            for lvl in range(hs.shape[0]):
                outputs_ious.append(self.detr.iou_embed[lvl](hs[lvl]))
            outputs_iou = torch.stack(outputs_ious)
            out['pred_ious'] = outputs_iou[-1]

            if self.detr.aux_loss:
                for i, aux in enumerate(out['aux_outputs']):
                    aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.detr.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

        return out


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, use_adapter):
        super().__init__()

        inter_dims = [
            dim,
            context_dim // 2, #128
            context_dim // 4, #64
            context_dim // 8, #32
            context_dim // 16, #16
            context_dim // 64, #4
        ]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim
        self.use_adapter = use_adapter

        if self.use_adapter:
            self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
            self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        if self.use_adapter:
            cur_fpn = self.adapter1(fpns[0])
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        # if self.use_adapter:
        #     cur_fpn = self.adapter2(fpns[1])
        #     if cur_fpn.size(0) != x.size(0):
        #         cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        #     x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        # if self.use_adapter:
        #     cur_fpn = self.adapter3(fpns[2])
        #     if cur_fpn.size(0) != x.size(0):
        #         cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        #     x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(
            k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1]
        )
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


'''
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Parameters:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes
'''

# DDETR Dice Loss
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Parameters:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes



def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid() # [B, D, L] but prob.
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def asym_focal_loss(inputs, targets, num_boxes, gamma_pos: float = 0, gamma_neg: float = 4, clip=0.05, eps=1e-8):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid() # [B, D, L] but prob.

    xs_pos, xs_neg = prob, 1 - prob

    # Asymmetric Clipping
    if clip is not None and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1)

    # Basic CE calculation
    los_pos = targets * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=eps))
    ce_loss = los_pos + los_neg

    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = prob * targets
        pt1 = (1 - prob) * (1 - targets)
        p_t = pt0 + pt1

        one_sided_gamma = gamma_pos * targets + gamma_neg * (1 - targets)
        one_sided_w = torch.pow(1 - p_t, one_sided_gamma)
        loss = ce_loss * one_sided_w

    loss = -loss

    return loss.mean(1).sum() / num_boxes


def softmax_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
