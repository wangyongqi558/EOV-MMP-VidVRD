import torch
from torch import nn
from gen_labels import gen_frame_pre_label, gen_box_label, create_mask, compute_iou
from gen_labels import format_trajectories, gen_feats, filter, format_trajectories_test, gen_feats_test
import numpy as np
import warnings
from os.path import join
warnings.filterwarnings("ignore")
from deep_sort_app import run
from AFLink.AppFreeLink import *
import math
import json
import copy
import datetime

class End2End_Model(nn.Module):
    def __init__(self, args, modelA, modelB, modelC, criterionA, postprocessorA):
        super(End2End_Model, self).__init__()
        self.args = args
        self.modelA = modelA  
        self.modelB = modelB  
        self.modelC = modelC
        self.criterionA = criterionA
        self.postprocessorA = postprocessorA
        self.traj = {}
        self.train_ecc = json.load(open('/data/VidVRD_ECC_train.json'))
        self.test_ecc = json.load(open('/data/VidVRD_ECC_test.json'))

    def forward(self, input, sort_model):
        # print(datetime.datetime.now())
        if self.training:
            begin_fid = input['begin_fid'][0]
            end_fid = input['end_fid'][0]
            patch_ = input['patch_'].squeeze()
            patch_proj = input['patch_proj'].squeeze()
            object_instances = []
            for img_id in range(begin_fid,end_fid):
                targets = []
                target = {}
                pred_label = gen_frame_pre_label(img_id, input['gt_rel'])
                pred_label = torch.tensor(pred_label)
                target['pred_label'] = pred_label.cuda()
                xyxy_boxes,cxcywh_boxes,labels = gen_box_label(img_id, input['gt_traj'],input['image_size'][0],input['image_size'][1])
                target['boxes'] = torch.tensor(cxcywh_boxes).cuda()
                target['xyxy_boxes'] = torch.tensor(xyxy_boxes).cuda()
                target['labels'] = torch.tensor(labels).cuda()
                target['orig_size'] = torch.tensor([input['image_size'][1],input['image_size'][0]]).cuda()
                target['size'] = torch.tensor([input['image_size'][1],input['image_size'][0]]).cuda()
                targets.append(target)
                if len(labels)>0:
                    object_instances_orig = []
                    p_ = patch_[img_id:img_id+1,:,:].cuda()
                    output, seleted_id = self.modelA(p_,targets)
                    loss_dict = self.criterionA(output, targets)
                    weight_dict = self.criterionA.weight_dict
                    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    loss.backward()
                    orig_target_sizes = torch.tensor([[input['image_size'][1],input['image_size'][0]]]).cuda()
                    results, set_index, topk_boxes = self.postprocessorA["bbox"](output, orig_target_sizes, seleted_id, True)
                    results = results[0]
                    for i in range(len(results['scores'])):
                        if results['scores'][i] >= 0.35:
                            object_instance = {}
                            object_instance['frame_id'] = img_id
                            object_instance['score'] = results['scores'][i]
                            object_instance['category'] = results['labels'][i]
                            box = results['boxes'][i]
                            w = input['image_size'][0].cuda()
                            h = input['image_size'][1].cuda()
                            new_box = [box[0].item(),box[1].item(),box[2].item(),box[3].item()]
                            if box[0] < 0:
                                new_box[0] = 0
                            if box[1] < 0:
                                new_box[1] = 0
                            if box[2] > w:
                                new_box[2] = w
                            if box[3] > h:
                                new_box[3] = h
                            box = torch.tensor(new_box).cuda()
                            object_instance['box'] = box
                            x1 = box[0] / w * 24
                            x2 = box[2] / w * 24
                            y1 = box[1] / h * 24
                            y2 = box[3] / h * 24
                            bbox_resize = [x1, y1, x2, y2]
                            mask = create_mask(24,24, bbox_resize)
                            mask = np.array(mask)
                            mask = mask.flatten()
                            mask_tensor = torch.from_numpy(mask).float().cuda()
                            mask_tensor = mask_tensor.unsqueeze(0)

                            xx2 = patch_proj[img_id:img_id+1,:,:].cuda()
                            mask_tensor_ = mask_tensor.unsqueeze(-1)
                            mask_tensor_ = mask_tensor_.expand_as(xx2)
                            weights_sum = mask_tensor_.sum(dim=1, keepdim=True).clamp(min=1e-9)
                            xx3_weighted = xx2 * mask_tensor_
                            xx4 = xx3_weighted.sum(dim=1) / weights_sum
                            xx4 /= xx4.norm(dim=-1, keepdim=True)
                            obj_patch_ = xx4.squeeze(0)
                            object_instance['patch_proj'] = obj_patch_
                            object_instances.append(object_instance)
                            object_instances_orig.append(object_instance)
            with torch.no_grad():
                associate_result =  run(self.args,
                                        input['video_name'],
                                        object_instances,
                                        min_confidence=self.args.min_confidence,
                                        nms_max_overlap=self.args.nms_max_overlap,
                                        min_detection_height=self.args.min_detection_height,
                                        max_cosine_distance=self.args.max_cosine_distance,
                                        nn_budget=self.args.nn_budget,
                                        display=False,
                                        ecc=self.train_ecc
                                        )
            if len(associate_result) > 0:
                datasetss = LinkData('', '')
                linker = AFLink(
                    results=associate_result,
                    model=sort_model,
                    dataset=datasetss,
                    thrT=(0, 30),  # (-10, 30) for CenterTrack, FairMOT, TransTrack.
                    thrS=75,
                    thrP=0.05  # 0.10 for CenterTrack, FairMOT, TransTrack.
                )
                associate_result = linker.link()
                temp = []
                for track in associate_result:
                    track[4] = track[2] + track[4]
                    track[5] = track[3] + track[5]
                    box_t = [track[2],track[3],track[4],track[5]]
                    max_iou = 0.5
                    for r in object_instances:
                        if track[0] == r['frame_id']:
                            box_r = r['box']
                            iou = compute_iou(box_t, box_r)
                            if iou > max_iou:
                                max_iou = iou
                                track[6] = r['category']
                                track[7] = r['score']
                    temp.append(track)
                trajectories = format_trajectories(temp)
                temptemp = []
                tid = 0
                min_l = end_fid - begin_fid
                for trajectory in trajectories:
                    if len(trajectory['trajectory']) > 0.15*min_l and len(trajectory['trajectory'])>14:
                        trajectory['tid'] = tid
                        tid += 1
                        temptemp.append(trajectory)
                trajectories = temptemp
                items = gen_feats(input['video_name'],trajectories,'train',input['patch_proj'],
                                    input['global_proj'],input['image_size'][0],input['image_size'][1])
                if len(items) > 0:
                    for idx in items:
                        data = items[idx]
                        seq_lens = torch.tensor([len(data['sbj_label'])])
                        labels = {
                            'pre_label': data['pre_label'].cuda(),
                            'sbj_label': data['sbj_label'].cuda(),
                            'obj_label': data['obj_label'].cuda()
                        }
                        feats = {}
                        for k in data:
                            if 'feat' in k:
                                feats[k] = data[k].float()
                        loss = self.modelC(feats, seq_lens, labels)
                        loss.backward()
            return
        else:
            with torch.no_grad():
                begin_fid = input['begin_fid'][0]
                end_fid = input['end_fid'][0]
                patch_ = input['patch_'].squeeze()
                patch_proj = input['patch_proj'].squeeze()
                object_instances = []
                for img_id in range(begin_fid,end_fid):
                    object_instances_orig=[]
                    p_ = patch_[img_id:img_id+1,:,:].cuda()
                    pp = patch_proj[img_id:img_id+1,:,:].cuda()
                    output, seleted_id = self.modelA(p_)
                    orig_target_sizes = torch.tensor([[input['image_size'][1],input['image_size'][0]]]).cuda()
                    results, set_index, topk_boxes = self.postprocessorA["bbox"](output, orig_target_sizes, seleted_id, False)
                    results = results[0]
                    
                    detected_boxes = results['boxes']
                    detected_boxes = filter(detected_boxes)
                    for box in detected_boxes:
                        object_instance = {}
                        object_instance['frame_id'] = img_id
                        scores = [0] * 35
                        for b in range(len(results['boxes'])):
                            if torch.equal(box, results['boxes'][b]):
                                label = results['labels'][b]
                                score = results['scores'][b].item()
                                scores[label] = score
                        w = input['image_size'][0].cuda()
                        h = input['image_size'][1].cuda()
                        new_box = [box[0].item(),box[1].item(),box[2].item(),box[3].item()]
                        if box[0] < 0:
                            new_box[0] = 0
                        if box[1] < 0:
                            new_box[1] = 0
                        if box[2] > w:
                            new_box[2] = w
                        if box[3] > h:
                            new_box[3] = h
                        box = torch.tensor(new_box).cuda()
                        object_instance['box'] = box
                        x1 = box[0] / w * 24
                        x2 = box[2] / w * 24
                        y1 = box[1] / h * 24
                        y2 = box[3] / h * 24
                        bbox_resize = [x1, y1, x2, y2]
                        mask = create_mask(24,24, bbox_resize)
                        mask = np.array(mask)
                        mask = mask.flatten()
                        mask_tensor = torch.from_numpy(mask).float().cuda()
                        mask_tensor = mask_tensor.unsqueeze(0)

                        xx2 = pp
                        mask_tensor_ = mask_tensor.unsqueeze(-1)
                        mask_tensor_ = mask_tensor_.expand_as(xx2)
                        weights_sum = mask_tensor_.sum(dim=1, keepdim=True).clamp(min=1e-9)
                        xx3_weighted = xx2 * mask_tensor_
                        xx4 = xx3_weighted.sum(dim=1) / weights_sum
                        xx4 /= xx4.norm(dim=-1, keepdim=True)
                        xx4 = xx4.squeeze(0)
                        obj_patch_proj = xx4
                        object_instance['patch_proj'] = obj_patch_proj
                        scores_classifer = self.modelB(xx4)[0].detach()
                        new_order = [0,33,20,32,1,2,3,4,30,5,6,7,28,21,27,
                                    8,25,9,34,10,31,24,11,29,12,23,13,22,14,15,16,17,26,18,19]
                        similarity = torch.index_select(scores_classifer, 0, torch.tensor(new_order).cuda())
                        similarity = similarity.tolist()
                        for nb in [1,3,8,12,14,23]:
                            scores[nb] = similarity[nb]*0.6 + scores[nb]*0.4 
                        for nb in [0,2,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34]:
                            scores[nb] = similarity[nb]*0.3 + scores[nb]*0.7
                        object_instance['score'] = torch.tensor(scores)
                        object_instances.append(object_instance)    
                with torch.no_grad():
                    associate_result =  run(self.args,
                                            input['video_name'],
                                            object_instances,
                                            min_confidence=self.args.min_confidence,
                                            nms_max_overlap=self.args.nms_max_overlap,
                                            min_detection_height=self.args.min_detection_height,
                                            max_cosine_distance=self.args.max_cosine_distance,
                                            nn_budget=self.args.nn_budget,
                                            display=False,
                                            ecc=self.test_ecc
                                            )
                final_results = []
                if len(associate_result) > 0:
                    datasetss = LinkData('', '')
                    linker = AFLink(
                        results=associate_result,
                        model=sort_model,
                        dataset=datasetss,
                        thrT=(0, 30),  # (-10, 30) for CenterTrack, FairMOT, TransTrack.
                        thrS=75,
                        thrP=0.05  # 0.10 for CenterTrack, FairMOT, TransTrack.
                    )
                    associate_result = linker.link()

                    temp = []
                    for track in associate_result:
                        expanded_array = np.zeros(45)
                        expanded_array[:10] = track
                        track = expanded_array
                        track[4] = track[2] + track[4]
                        track[5] = track[3] + track[5]
                        box_t = [track[2],track[3],track[4],track[5]]
                        max_iou = 0.5
                        for r in object_instances:
                            if track[0] == r['frame_id']:
                                box_r = r['box']
                                iou = compute_iou(box_t, box_r)
                                if iou > max_iou:
                                    max_iou = iou
                                    track[6] = 1
                                    for i in range(35):
                                        track[i+10] = r['score'][i]
                        temp.append(track)
                    min_l = end_fid - begin_fid
                    trajectories = format_trajectories_test(temp)
                    temptemp = []
                    tid = 0
                    for trajectory in trajectories:
                        if len(trajectory['trajectory']) > 0.15*min_l:
                            trajectory['tid'] = tid
                            tid += 1
                            temptemp.append(trajectory)
                    trajectories = temptemp
                    self.traj[input['video_name'][0]] = trajectories
                    items = gen_feats_test(input['video_name'],trajectories,'test',input['patch_proj'],
                                        input['global_proj'],input['image_size'][0],input['image_size'][1])
                    if len(items) > 0:
                        for idx in items:
                            final_result = {}
                            data = items[idx]
                            seq_lens = torch.tensor([data['clip_feat'].shape[1]])
                            feats = {}
                            for k in data:
                                if 'feat' in k:
                                    feats[k] = data[k].float()
                            pre_preds, sbj_preds, obj_preds = self.modelC(feats, seq_lens)
                            final_result['pre_preds'] = pre_preds
                            final_result['sbj_preds'] = sbj_preds
                            final_result['obj_preds'] = obj_preds
                            final_result['seq_lens'] = seq_lens
                            final_result['video_name'] = input['video_name']
                            final_result['pair_data'] = data['pair_data']
                            final_results.append(final_result)           
        return final_results
