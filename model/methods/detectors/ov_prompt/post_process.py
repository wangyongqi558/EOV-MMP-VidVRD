import torch
import torch.nn as nn
import torch.nn.functional as F
from util import box_ops
import numpy as np
from util.coco_categories import SEEN_CATEGORIES as COCO_SEEN_CATEGORIES
from util.coco_categories import UNSEEN_CATEGORIES as COCO_UNSEEN_CATEGORIES
from util.lvis_v1_categories import SEEN_CATEGORIES as LVIS_SEEN_CATEGORIES
from util.lvis_v1_categories import UNSEEN_CATEGORIES as LVIS_UNSEEN_CATEGORIES
from util.vidvrd_categories import SEEN_CATEGORIES as VidVRD_SEEN_CATEGORIES
from util.vidvrd_categories import UNSEEN_CATEGORIES as VidVRD_UNSEEN_CATEGORIES
from util.vidor_categories import SEEN_CATEGORIES as VidOR_SEEN_CATEGORIES
from util.vidor_categories import UNSEEN_CATEGORIES as VidOR_UNSEEN_CATEGORIES

# CLIP AUX CLassifier
from util.detectron2.structures import Boxes
from util.clip_image_encoder import RoIAlignViTImageEncoder
import torchvision.transforms.functional as TF
from util.clip_utils import build_text_embedding_coco, build_text_embedding_lvis, build_text_embedding_vidvrd, build_text_embedding_vidor


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes, topk_boxes, aux_classifier):
        assert len(orig_target_sizes) == len(max_target_sizes)
        if aux_classifier is None:
            max_h, max_w = max_target_sizes.max(0)[0].tolist()
            outputs_masks = outputs["pred_masks"]
            outputs_masks = F.interpolate(
                outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
            )
            outputs_masks = outputs_masks.sigmoid() > self.threshold
            outputs_masks = outputs_masks.reshape((len(results), outputs_masks.size(0) // len(results), max_h, max_w))
        else:
            outputs_masks = []
            for elem, topk_box in zip(outputs["pred_masks"], topk_boxes): # list로 되어있음.
                max_h, max_w = max_target_sizes.max(0)[0].tolist()
                tmp_outputs_masks = elem
                tmp_outputs_masks = F.interpolate(
                    tmp_outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
                )
                tmp_outputs_masks = tmp_outputs_masks.sigmoid() > self.threshold

                tmp_outputs_masks = tmp_outputs_masks[topk_box, :, :, :]
                # tmp_outputs_masks = torch.gather(tmp_outputs_masks, 1, topk_box.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, tmp_outputs_masks.size(2), tmp_outputs_masks.size(3))).squeeze(0)
                # print(f"seg: {tmp_outputs_masks.shape}")

                outputs_masks.append(tmp_outputs_masks)
            # outputs_masks = torch.cat(outputs_masks, dim=0)

        for i, (cur_mask, t, tt) in enumerate(
            zip(outputs_masks, max_target_sizes, orig_target_sizes)
        ):
            # print(f"cur mask :{cur_mask.shape}")
            img_h, img_w = t[0], t[1]
            # print(f"imh imw: {img_h}, {img_w}")
            results[i]["masks"] = cur_mask[:, :img_h, :img_w]

            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()
            # print(results[i]["masks"].shape)
        return results
'''
class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"]
        outputs_masks = F.interpolate(
            outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
        )
        outputs_masks = outputs_masks.sigmoid() > self.threshold
        outputs_masks = outputs_masks.reshape((len(results), outputs_masks.size(0) // len(results), max_h, max_w))
        print(outputs_masks.shape)
        for i, (cur_mask, t, tt) in enumerate(
            zip(outputs_masks, max_target_sizes, orig_target_sizes)
        ):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results
'''
class Aux_CLIP_Classifier(nn.Module):

    def __init__(self, dataset_file, clip_backbone='ViT-L/14@336px', size=(420, 420),
                 temperature=0.06, bg=True):
        super().__init__()

        # if dataset_file == "open_coco":
        #     aux_zeroshot_w = build_text_embedding_coco(clip_backbone, bg=bg)
        # elif dataset_file == "open_lvis":
        #     aux_zeroshot_w = build_text_embedding_lvis(clip_backbone, bg=bg)
        # elif dataset_file == "open_vidvrd":
        #     aux_zeroshot_w = build_text_embedding_vidvrd(clip_backbone, bg=bg)

        # self.aux_zeroshot_w = aux_zeroshot_w.t()
        # self.aux_clip_image_backbone = RoIAlignViTImageEncoder(clip_backbone).cuda()
        # self.temperature = temperature
        # self.size = size

    def forward(self, samples, roi_boxes):

        a = 0
        for box in roi_boxes:
            a += len(box)
        scores = torch.zeros(a, 81)
    
        return scores


class OVPostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_queries=300, dataset_file='open_coco',
                 size=(420, 420),
                 clip_backbone=None,
                 temperature=0.06,
                 bg=True,
                 pruning_threshold=0.35):

        super().__init__()

        self.num_queries = num_queries

        if dataset_file == "open_coco":
            self.seen_list = COCO_SEEN_CATEGORIES
            self.unseen_list = COCO_UNSEEN_CATEGORIES
        elif dataset_file == "open_lvis":
            self.seen_list = LVIS_SEEN_CATEGORIES
            self.unseen_list = LVIS_UNSEEN_CATEGORIES
        elif dataset_file == "open_vidvrd":
            self.seen_list = sorted(VidVRD_SEEN_CATEGORIES)
            self.unseen_list = sorted(VidVRD_UNSEEN_CATEGORIES)
        elif dataset_file == "open_vidor":
            self.seen_list = sorted(VidOR_SEEN_CATEGORIES)
            self.unseen_list = sorted(VidOR_UNSEEN_CATEGORIES)

        # aux clip cls
        self.aux_classifier = Aux_CLIP_Classifier(dataset_file, clip_backbone, size, temperature, bg=bg)
        self.pruning_threshold = pruning_threshold
        self.size = size

    @torch.no_grad()
    def forward(self, outputs, target_sizes,selected_id,classifier):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        bs = out_bbox.shape[0]

        if classifier:
            prob = out_logits.sigmoid()

            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            new_tensor = torch.zeros_like(labels)
            for idx in range(len(labels[0])):
                new_tensor[0, idx] = selected_id[labels[0, idx]]
            labels = new_tensor
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]
            set_index = [[i for i in range(boxes.size(1))] for _ in range(boxes.size(0))]
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

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
            img_h, img_w = target_sizes.unbind(1) # batch input resolution
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

            #
            det_total_prob = []
            set_index = []
            for idx in range(bs):
                indice = (masking[idx] == True).nonzero(as_tuple=True)[0]
                if len(indice) == 0:
                    # if no selected box, then pick the best one.
                    max, _ = prob[idx].max(dim=-1)
                    _, indice = max.max(dim=0)
                    box = clip_bbox[idx][indice].unsqueeze(0)
                    det_seen_prob.append(seen_prob[idx][indice].unsqueeze(0)) # det prob for selected boxes
                    det_unseen_prob.append(unseen_prob[idx][indice].unsqueeze(0)) # det prob for selected boxes
                    roi_box.append(Boxes(box)) # roi for selected boxes (resized resolution)
                    bs_indice.append(range(increment, box.shape[0]+increment)) # indices for selected boxes per sample
                    eval_box.append(out_bbox[idx][indice].unsqueeze(0)) # for evaluation (original resolution)
                    det_total_prob.append(prob[idx][indice].unsqueeze(0))
                    increment += box.shape[0]
                    set_index.append([indice])
                else:
                    box = clip_bbox[idx][indice]
                    det_seen_prob.append(seen_prob[idx][indice]) # det prob for selected boxes
                    det_unseen_prob.append(unseen_prob[idx][indice]) # det prob for selected boxes
                    roi_box.append(Boxes(box)) # roi for selected boxes (resized resolution)
                    bs_indice.append(range(increment, box.shape[0]+increment)) # indices for selected boxes per sample
                    eval_box.append(out_bbox[idx][indice]) # for evaluation (original resolution)
                    det_total_prob.append(prob[idx][indice])
                    increment += box.shape[0]
                    set_index.append(indice)

            results = []
            topk_boxes = []
            for det_seen_p, det_unseen_p,  box, target_size, det_total_p \
                    in zip(det_seen_prob, det_unseen_prob,  eval_box, target_sizes, det_total_prob):
                seen_det_prob = det_seen_p
                seen_det_prob = torch.nn.functional.pad(seen_det_prob, (0, len(unseen_list), 0, 0), value=0.0)
                unseen_det_prob = det_unseen_p
                unseen_det_prob = torch.nn.functional.pad(unseen_det_prob, (len(seen_list), 0, 0, 0), value=0.0)
                # seen_clip_prob = aux_seen_p
                # seen_clip_prob = torch.nn.functional.pad(seen_clip_prob, (0, len(unseen_list), 0, 0), value=0.0)
                # unseen_clip_prob = aux_unseen_p
                # unseen_clip_prob = torch.nn.functional.pad(unseen_clip_prob, (len(seen_list), 0, 0, 0), value=0.0)

                alpha = 0.2 # seen
                beta = 0.5 # unseen
                seen_prob = seen_det_prob# * (1.0 - alpha)) + (seen_clip_prob * (alpha))
                unseen_prob = unseen_det_prob #* (1.0 - beta)) + (unseen_clip_prob * (beta))
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
                # boxes = box.unsqueeze(0)
                boxes = torch.gather(boxes, 1, topk_box.unsqueeze(-1).repeat(1, 1, 4)).squeeze(0)
                topk_boxes.append(topk_box)
                # and from relative [0, 1] to absolute [0, height] coordinates
                img_h, img_w = target_size.unbind(0)
                scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(boxes)
                final_boxes = boxes * scale_fct

                results.append({'scores': scores, 'labels': labels, 'boxes': final_boxes})

        return results, set_index, topk_boxes