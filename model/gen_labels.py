import json
import torch
import os
import numpy as np
from collections import defaultdict
from os.path import join
from utils.utils import vru19_ext_loc_feat, gen_union_bbox, ext_bbox_loc_feat
import copy

ROOT = '../dataset/vidvrd'
CLIP_LEN = 30
class FeatExtractor:
    def __init__(self, feat_types):
        self.type2extractor = {
            'rel_feat':self._extract_rel_feat,
            'mot_feat':self._extract_mot_feat,
            'clip_feat':self._extract_clip_feat,
            'bbox_feat':self._extract_bbox_feat,
            }
        self.feat_types = feat_types
        self.data = None

    def load_frames(self, data):
        self.num_frames = len(os.listdir(join(ROOT, 'frames', data['video_id'])))
        # self.num_frames = len(os.listdir(join("../dataset/vidvrd", 'frames', data['video_id'])))
        self.video_height = data["height"]
        self.video_width = data["width"]
        self.frame_paths = {}
        for fid in range(self.num_frames):
            self.frame_paths[fid] = join(ROOT, 'frames', data['video_id'], '%06d.jpg'%(fid+1)) # picture No is named from 1
            # self.frame_paths[fid] = join("../dataset/vidvrd", 'frames', data['video_id'], '%06d.jpg'%(fid+1)) # picture No is named from 1

    def gen_feats(self, clip_data, patch_proj, global_proj,w,h):
        self.data = clip_data
        feats = {}
        for type in self.feat_types:
            if type != 'clip_feat':
                feats[type] = self.type2extractor[type]()
            else:
                feats[type] = self.type2extractor[type](patch_proj, global_proj,w,h)
        return feats
    
    def _extract_rel_feat(self):
        data = self.data
        assert data != None
        mid_fno = len(data['sbj_traj'])//2
        begin_feat  = vru19_ext_loc_feat(data['sbj_traj'][0], data['obj_traj'][0], self.video_height, self.video_width)
        mid_feat    = vru19_ext_loc_feat(data['sbj_traj'][mid_fno], data['obj_traj'][mid_fno], self.video_height, self.video_width)
        end_feat    = vru19_ext_loc_feat(data['sbj_traj'][-1], data['obj_traj'][-1], self.video_height, self.video_width)
        feat = np.concatenate((begin_feat, mid_feat, end_feat))
        return feat
    def _extract_mot_feat(self):
        data = self.data
        assert data != None
        mid_fno = len(data['sbj_traj'])//2
        begin_feat  = vru19_ext_loc_feat(data['sbj_traj'][0], data['obj_traj'][0], self.video_height, self.video_width)
        mid_feat    = vru19_ext_loc_feat(data['sbj_traj'][mid_fno], data['obj_traj'][mid_fno], self.video_height, self.video_width)
        end_feat    = vru19_ext_loc_feat(data['sbj_traj'][-1], data['obj_traj'][-1], self.video_height, self.video_width)
        bm_mot_feat = mid_feat-begin_feat
        me_mot_feat = end_feat-mid_feat
        be_mot_feat = end_feat-begin_feat
        feat = np.concatenate((bm_mot_feat, me_mot_feat, be_mot_feat))
        return feat

    def _extract_lan_feat(self):
        data = self.data
        assert data != None
        return np.concatenate((self.obj2vec[data['sbj_id']],self.obj2vec[data['obj_id']]))

    def _extract_clip_feat(self,patch_proj, global_proj,w,h):
        # print(patch_proj.shape)
        patch_proj = patch_proj[0]
        # print(patch_proj.shape)
        global_proj = global_proj[0]
        with torch.no_grad():
            data = self.data
            assert data != None

            mid_fno = len(data['sbj_traj'])//2
            mid_fid = data['begin_fid'] + mid_fno

            bboxes = [
                    data['sbj_traj'][mid_fno],
                    data['obj_traj'][mid_fno], 
                    gen_union_bbox(data['sbj_traj'][mid_fno], data['obj_traj'][mid_fno]),

                    None
                    ]

            feats = []
            for box in bboxes:
                if box != None:
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

                    xx2 = patch_proj[mid_fid:mid_fid+1,:,:].cuda()
                    mask_tensor_ = mask_tensor.unsqueeze(-1)
                    mask_tensor_ = mask_tensor_.expand_as(xx2)
                    weights_sum = mask_tensor_.sum(dim=1, keepdim=True).clamp(min=1e-9)
                    xx3_weighted = xx2 * mask_tensor_
                    xx4 = xx3_weighted.sum(dim=1) / weights_sum
                    # xx4 /= xx4.norm(dim=-1, keepdim=True)
                    obj_patch_ = xx4.squeeze(0)
                    # print(f"1111111111111111111111111111{obj_patch_.shape}")
                    feats.append(obj_patch_.squeeze())
                else:
                    feats.append(global_proj[mid_fid:mid_fid+1,:].squeeze().cuda())
            feat = torch.stack(feats,dim=0)
        return feat

    def _extract_bbox_feat(self):
        data = self.data
        assert data != None
        head_s, head_o, head_u, head_t = ext_bbox_loc_feat(
                                   data['sbj_traj'][0], 
                                   data['obj_traj'][0], 
                                   gen_union_bbox(data['sbj_traj'][0], data['obj_traj'][0]),
                                   self.video_height, self.video_width)
        tail_s, tail_o, tail_u, tail_t = ext_bbox_loc_feat(
                                   data['sbj_traj'][-1], 
                                   data['obj_traj'][-1], 
                                   gen_union_bbox(data['sbj_traj'][-1], data['obj_traj'][-1]),
                                   self.video_height, self.video_width)
        diff_s = tail_s - head_s
        diff_o = tail_o - head_o
        diff_u = tail_u - head_u
        diff_t = tail_t - head_t
        feat = np.asarray([
                np.concatenate((head_s, tail_s, diff_s)),
                np.concatenate((head_o, tail_o, diff_o)),
                np.concatenate((head_u, tail_u, diff_u)),
                np.concatenate((head_t, tail_t, diff_t))])
        return feat

def filter(detected_boxes):
    detected_boxes = unique_boxes(detected_boxes)
    return detected_boxes

def unique_boxes(boxes, threshold=0.8):
    """
    根据 IoU 去除重复的边界框。
    """
    if boxes.size(0) == 0:
        return boxes
    iou = calculate_iou(boxes, boxes)
    indices = torch.triu_indices(iou.shape[0], iou.shape[1], offset=1)
    duplicated = iou[indices[0], indices[1]] > threshold
    indices = indices.cuda()
    duplicated = duplicated.cuda()
    
    unique_indices = set(range(iou.shape[0])) - set(indices[0][duplicated].tolist())
    return boxes[list(unique_indices), :]

def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU。
    box1 和 box2 的形状分别为 [N,4] 和 [M,4]，其中 N 和 M 不必相同。
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    union_area = area1[:, None] + area2 - inter_area
    
    return inter_area / union_area

def gen_hit_tid(clip, gt_trajs):

    clip_begin_fid = clip['begin_fid']
    clip_end_fid = clip['end_fid']

    sbj_max_viou = -float('Inf')
    sbj_hit_tid = -1 
    obj_max_viou = -float('Inf')
    obj_hit_tid = -1 
    for gt_tid in gt_trajs:
        clip_gt_traj = []
        for fid in range(clip_begin_fid, clip_end_fid):
            if fid in gt_trajs[gt_tid]:
                clip_gt_traj.append(gt_trajs[gt_tid][fid])
            else:
                clip_gt_traj.append(None)
        sbj_viou = cal_viou(clip['sbj_traj'], clip_gt_traj)
        if sbj_viou >= 0.5 and sbj_viou > sbj_max_viou:
            sbj_max_viou = sbj_viou
            sbj_hit_tid = gt_tid
        
        obj_viou = cal_viou(clip['obj_traj'], clip_gt_traj)
        if obj_viou >= 0.5 and obj_viou > obj_max_viou:
            obj_max_viou = obj_viou
            obj_hit_tid = gt_tid
    return sbj_hit_tid, obj_hit_tid

def cal_viou(traj_1, traj_2):
    """ compute the voluminal Intersection over Union
    for two trajectories, each of which is represented
    by a duration [fstart, fend) and a list of bounding
    boxes (i.e. traj) within the duration.
    """
    v_overlap = 0
    for i in range(len(traj_1)):
        roi_1 = traj_1[i]
        roi_2 = traj_2[i]
        if roi_2 == None:
            v_overlap += 0
        else:
            left = max(roi_1[0], roi_2[0])
            top = max(roi_1[1], roi_2[1])
            right = min(roi_1[2], roi_2[2])
            bottom = min(roi_1[3], roi_2[3])
            v_overlap += max(0, right - left + 1) * max(0, bottom - top + 1)
    v1 = 0
    for i in range(len(traj_1)):
        v1 += (traj_1[i][2] - traj_1[i][0] + 1) * (traj_1[i][3] - traj_1[i][1] + 1)
    v2 = 0
    for i in range(len(traj_2)):
        if traj_2[i] == None:
            v2 += 0
        else:
            v2 += (traj_2[i][2] - traj_2[i][0] + 1) * (traj_2[i][3] - traj_2[i][1] + 1)
    return float(v_overlap) / (v1 + v2 - v_overlap)

def gen_frame_pre_label(frame_id, anno):
    split_path = '../dataset/vidvrd/data/openvoc_pred_class_spilt_info.json'
    with open(split_path, 'r') as f:
        split = json.load(f)
    cls2id = split['cls2id']
    cls2split = split['cls2split']
    ks = list(cls2split.keys())
    label_map = {}
    for cnt in range(len(ks)):
        if cls2split[ks[cnt]] == 'base':
            label_map[ks[cnt]] = cnt
    p = []
    label = [0] * 71
    for instance in anno:
        if frame_id >= instance['begin_fid'] and frame_id < instance['end_fid']:
            p.append(instance['predicate'])
    if len(p) > 0:
        for cls_name in p:
            if cls2split[cls_name[0]] == 'base':
                label[label_map[cls_name[0]]] = 1
    return label

def gen_box_label(frame_id, trajs, w, h):
    split_path = '../dataset/vidvrd/data/openvoc_obj_class_spilt_info.json'
    with open(split_path, "r") as f:
        split = json.load(f)
    cls2id = split['cls2id']
    xyxy_boxes = []
    cxcywh_boxes = []
    labels = []
    for traj in trajs:
        if str(frame_id) in traj['trajectory'].keys():
            box = traj['trajectory'][str(frame_id)]
            box = [box[0][0],box[1][0],box[2][0],box[3][0]]
            xyxy_box = [box[0]/w,box[1]/h,box[2]/w,box[3]/h]
            cxcywh_box = [(xyxy_box[0]+xyxy_box[2])/2,(xyxy_box[1]+xyxy_box[3])/2,
                          (xyxy_box[2]-xyxy_box[0]),(xyxy_box[3]-xyxy_box[1])]
            label = cls2id[traj['category'][0]]
            xyxy_boxes.append(xyxy_box)
            cxcywh_boxes.append(cxcywh_box)
            labels.append(label)
    return xyxy_boxes,cxcywh_boxes,labels

def create_mask(x, y, bbox):
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

def gen_label(clip, gt_relations):
    labels = []
    for relation in gt_relations:
        if (clip["sbj_tid"] == relation["subject_tid"]) and (clip["obj_tid"] == relation["object_tid"]):

            # <1> clipment has the relation label if having intersection with the relation duration
            # if (relation["begin_fid"] <= clip["begin_fid"] < relation["end_fid"]) or\
            #     (relation["begin_fid"] < clip["end_fid"] <= relation["end_fid"]):
            #     labels.append(relation["predicate"]) 

            # <2> clipment has the relation label if having intersection larger than threshold
            left = max(clip["begin_fid"], relation["begin_fid"])
            right = min(clip["end_fid"], relation["end_fid"])
            if right - left >= 10:
                labels.append(relation["predicate"])
    labels = list(set(labels))
    return labels  

def compute_iou(box1, box2):
    """
    计算两个边界框的交并比（IoU）。

    参数:
    box1, box2 -- 两个边界框，各自表示为 (x1, y1, x2, y2)

    返回:
    iou -- 两个边界框的交并比
    """

    # 计算交集的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集的面积
    if (x2_inter < x1_inter) or (y2_inter < y1_inter):
        inter_area = 0  # 如果没有交集
    else:
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算每个边界框的面积
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集的面积
    union_area = area_box1 + area_box2 - inter_area
    
    # 计算IoU
    iou = inter_area / union_area
    
    return iou

def add_initial_frames(track_frames):
    # Find the first frame
    first_frame_id = min(track_frames.keys())
    first_frame_bbox = track_frames[first_frame_id]
    
    # Add two frames before the first frame
    track_frames[first_frame_id - 2] = first_frame_bbox
    track_frames[first_frame_id - 1] = first_frame_bbox
    return track_frames

def interpolate_and_adjust_frames(track_frames):
    sorted_frames = sorted(track_frames.keys())
    x_lefts = [track_frames[f][0] for f in sorted_frames]
    y_uppers = [track_frames[f][1] for f in sorted_frames]
    x_rights = [track_frames[f][2] for f in sorted_frames]
    y_bottoms = [track_frames[f][3] for f in sorted_frames]
    
    frame_ids = np.arange(sorted_frames[0], sorted_frames[-1] + 1)
    if len(sorted_frames) < len(frame_ids)*0.65:
        return {}
    else:
        sorted_frames_copy = sorted_frames
        while True:
            # 计算相邻帧之间的差值
            differences = np.diff(sorted_frames)

            # 计算最大值和最小值之间的总差
            total_difference = sorted_frames[-1] - sorted_frames[0]

            # 设置阈值为总差的 20%
            threshold = 0.15 * total_difference

            # 查找所有间隙并按大小排序
            gap_indices = np.argsort(-differences)  # 负号用于降序排序
            large_gaps = gap_indices[differences[gap_indices] > threshold]

            if large_gaps.size == 0:
                # 没有大于阈值的间隙，退出循环
                break

            # 处理最大的间隙，去掉较少的一段
            max_gap_index = large_gaps[0]
            if max_gap_index + 1 > len(sorted_frames) - max_gap_index - 1:
                # 如果第一个间隙前的帧数更多
                sorted_frames = sorted_frames[:max_gap_index + 1]
            else:
                # 如果第一个间隙后的帧数更多
                sorted_frames = sorted_frames[max_gap_index + 1:]

        # 根据最终的 sorted_frames 生成 frame_ids
        frame_ids = np.arange(sorted_frames[0], sorted_frames[-1] + 1)
        sorted_frames = sorted_frames_copy
    
    interp_x_lefts = np.interp(frame_ids, sorted_frames, x_lefts)
    interp_y_uppers = np.interp(frame_ids, sorted_frames, y_uppers)
    interp_x_rights = np.interp(frame_ids, sorted_frames, x_rights)
    interp_y_bottoms = np.interp(frame_ids, sorted_frames, y_bottoms)
    
    interpolated_track = {}
    for frame_id, bbox in zip(frame_ids, zip(interp_x_lefts, interp_y_uppers, interp_x_rights, interp_y_bottoms)):
        interpolated_track[int(frame_id)] = round_and_positive(bbox)
    return interpolated_track

def round_and_positive(bbox):
    # Round values and ensure non-negative
    return [max(0, round(coord)) for coord in bbox]

def format_trajectories(temp):
    obj_split = json.load(open(
            '../dataset/vidvrd/data/openvoc_obj_class_spilt_info.json', "r"))
    trajectories = {}
    traj_scores = {}
    for line in temp:
        frame_id, track_id, x_left, y_upper, x_right, y_bottom, category, score, _ ,_= line
        if track_id not in trajectories:
            trajectories[track_id] = {}
            traj_scores[track_id] = {}
        trajectories[track_id][frame_id] = [x_left, y_upper, x_right, y_bottom]
        if score > 0:
            if category not in traj_scores[track_id]:
                traj_scores[track_id][category] = []
            traj_scores[track_id][category].append(score)

    for track_id, track_frames in trajectories.items():
        track_frames = add_initial_frames(track_frames)
        if len(track_frames) > 1:
            trajectories[track_id] = interpolate_and_adjust_frames(track_frames)
        else:
            trajectories[track_id] = {frame_id: round_and_positive(bbox) for frame_id, bbox in track_frames.items()}
        scores = traj_scores[track_id]
        f = 0
        for s in scores:
            f += len(scores[s])
        cat = -1
        max_score = 0
        for s in scores:
            ss = sum(scores[s])/f
            if ss > max_score:
                max_score = ss
                cat = s
        if str(int(cat)) == '-1':
            cat = 1
        trajectories[track_id]['category'] = obj_split['id2cls'][str(int(cat))]
        trajectories[track_id]['score'] = max_score

    formatted_trajectories = []
    tid = 0
    for track in trajectories.values():
        if len(track)>=12:
            track['tid'] = tid
            tttt = {}
            keys_to_delete = [key for key in track if key not in ['category','score','tid']]
            for key in keys_to_delete:
                tttt[key] = track[key]
                del track[key]
            track['trajectory'] = tttt
            track['begin_fid'] = min(tttt.keys())
            track['end_fid'] = max(tttt.keys()) +1 
            formatted_trajectories.append(track)
            tid += 1

    return formatted_trajectories

def gen_gt_trajs(vid_anno):
    gt_trajs = defaultdict(dict)
    for fid, frame in enumerate(vid_anno["trajectories"]):
        for bbox_anno in frame:
            tid = bbox_anno["tid"]
            bbox = bbox_anno["bbox"]
            bbox = [bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"]]
            gt_trajs[tid][fid] = bbox
    return gt_trajs

def gen_feats(video_name, trajectories, split, patch_proj, global_proj,w,h):
    object2id = json.load(open(join(ROOT, 'data', 'object2id.json'),'r'))
    predicate2id = json.load(open(join(ROOT, 'data','predicate2id.json'),'r'))
    path = '../dataset/vidvrd/anno/' + split
    vid_anno = json.load(open(path+'/'+video_name[0]+'.json','r'))
    vid_gt_trajs = gen_gt_trajs(vid_anno)
    feat_extractor = FeatExtractor(['rel_feat', 'mot_feat', 'clip_feat', 'bbox_feat'])
    feat_extractor.load_frames(vid_anno)
    vid_name = video_name[0]
    vid_trajs = trajectories
    # print(vid_gt_trajs)
    # print(vid_trajs)
    # dsfasdfasdfasd
    
    pair_id = 0
    items = {}
    for s_traj_id, s_traj in enumerate(vid_trajs):
        for o_traj_id, o_traj in enumerate(vid_trajs):
            if s_traj_id == o_traj_id: continue
            begin_fid = max(s_traj['begin_fid'], o_traj['begin_fid'])
            end_fid   = min(s_traj['end_fid'], o_traj['end_fid'])
            if (end_fid - begin_fid) < 10: continue

            clip_num = int((end_fid-begin_fid) / CLIP_LEN)
            tail_len = (end_fid-begin_fid) % CLIP_LEN
            if tail_len >= 10:
                clip_num += 1
            elif 0 < tail_len <10:
                end_fid = end_fid - tail_len
            
            s_bboxes = []
            o_bboxes = []
            for fid in range(begin_fid, end_fid):
                s_bboxes.append(s_traj['trajectory'][fid])
                o_bboxes.append(o_traj['trajectory'][fid]) 

            pair_labels = []

            pair_name = vid_name + "_%06d.pkl"%pair_id
            pair_id += 1
            pair_feats = defaultdict(list)
            
            for clip_id in range(clip_num):
                clip = {} 
                clip["sbj_id"] = object2id[s_traj['category']]
                clip["obj_id"] = object2id[o_traj['category']]
                clip["begin_fid"] = begin_fid + clip_id*CLIP_LEN
                if clip_id == clip_num-1:
                    clip["sbj_traj"] = s_bboxes[clip_id*CLIP_LEN:]
                    clip["obj_traj"] = o_bboxes[clip_id*CLIP_LEN:]
                    clip["end_fid"] = end_fid
                else:
                    clip["sbj_traj"] = s_bboxes[clip_id*CLIP_LEN:(clip_id+1)*CLIP_LEN]
                    clip["obj_traj"] = o_bboxes[clip_id*CLIP_LEN:(clip_id+1)*CLIP_LEN]
                    clip["end_fid"] = begin_fid + (clip_id+1)*CLIP_LEN
                
                
                clip["sbj_tid"], clip["obj_tid"] = gen_hit_tid(clip, vid_gt_trajs)
                pair_labels.append([predicate2id[p] for p in gen_label(clip, vid_anno["relation_instances"])])
                
                feats = feat_extractor.gen_feats(clip,patch_proj, global_proj,w,h)
                for type_ in ['rel_feat', 'mot_feat', 'clip_feat', 'bbox_feat']:
                    pair_feats[type_].append(feats[type_])
                # print(pair_feats['clip_feat'][0].shape)
                # fawefawefawefe1
            
            pair_labels = [object2id[s_traj['category']], object2id[o_traj['category']], pair_labels]
            data = [pair_feats, pair_labels]
            pair_feats = data[0]
            item = {}
            for type_ in pair_feats:
                if type_ != 'clip_feat':
                    item[type_] = torch.tensor(pair_feats[type_]).unsqueeze(dim=0)
                else:
                    item[type_] = torch.stack(pair_feats[type_],dim=0).unsqueeze(dim=0).float()
            pair_data = data[1][2]
            pair_label = np.zeros((len(pair_data), 132),)
            for clip_idx, clip_label in enumerate(pair_data):
                tmp_label = np.zeros(132,)
                if len(clip_label) > 0:
                    tmp_label[clip_label] = 1
                pair_label[clip_idx] = tmp_label
            item['pre_label'] = torch.tensor(pair_label).unsqueeze(dim=0)
            item['sbj_label'] = torch.tensor([data[1][0]])
            item['obj_label'] = torch.tensor([data[1][1]])
            items[pair_id] = item
    return items



def format_trajectories_test(temp):
    obj_split = json.load(open(
            '../dataset/vidvrd/data/openvoc_obj_class_spilt_info.json', "r"))
    trajectories = {}
    traj_scores = {}
    for line in temp:
        frame_id=line[0]
        track_id=line[1]
        x_left=line[2]
        y_upper=line[3]
        x_right=line[4]
        y_bottom=line[5]
        score=line[10:]
        if track_id not in trajectories:
            trajectories[track_id] = {}
            traj_scores[track_id] = []
        trajectories[track_id][frame_id] = [x_left, y_upper, x_right, y_bottom]
        if max(score) > 0:
            traj_scores[track_id].append(score)
    for track_id, track_frames in trajectories.items():
        track_frames = add_initial_frames(track_frames)
        if len(track_frames) > 1:
            trajectories[track_id] = interpolate_and_adjust_frames(track_frames)
        else:
            trajectories[track_id] = {frame_id: round_and_positive(bbox) for frame_id, bbox in track_frames.items()}
        scores = traj_scores[track_id]
        sssss = np.zeros(35)
        f = 0
        cat = -1
        max_score = 0
        for s in scores:
            sssss += s
            f+=1
        max_score = max(sssss)/f
        cat = np.argmax(sssss)
        trajectories[track_id]['category'] = obj_split['id2cls'][str(int(cat))]
        trajectories[track_id]['score'] = max_score
    formatted_trajectories = []
    tid = 0
    for track in trajectories.values():
        if len(track)>=12:
            track['tid'] = tid
            tttt = {}
            keys_to_delete = [key for key in track if key not in ['category','score','tid']]
            for key in keys_to_delete:
                tttt[key] = track[key]
                del track[key]
            track['trajectory'] = tttt
            track['begin_fid'] = min(tttt.keys())
            track['end_fid'] = max(tttt.keys()) +1 
            formatted_trajectories.append(track)
            tid += 1

    return formatted_trajectories


def gen_feats_test(video_name, trajectories, split, patch_proj, global_proj,w,h):
    object2id = json.load(open(join(ROOT, 'data', 'object2id.json'),'r'))
    predicate2id = json.load(open(join(ROOT, 'data','predicate2id.json'),'r'))
    path = '../dataset/vidvrd/anno/' + split
    vid_anno = json.load(open(path+'/'+video_name[0]+'.json','r'))
    feat_extractor = FeatExtractor(['rel_feat', 'mot_feat', 'clip_feat', 'bbox_feat'])
    feat_extractor.load_frames(vid_anno)
    vid_name = video_name[0]
    vid_trajs = trajectories
    
    pair_id = 0
    items = {}
    for s_traj_id, s_traj in enumerate(vid_trajs):
        for o_traj_id, o_traj in enumerate(vid_trajs):
            if s_traj_id == o_traj_id: continue
            begin_fid = max(s_traj['begin_fid'], o_traj['begin_fid'])
            end_fid   = min(s_traj['end_fid'], o_traj['end_fid'])
            if (end_fid - begin_fid) < 10: continue

            clip_num = int((end_fid-begin_fid) / CLIP_LEN)
            tail_len = (end_fid-begin_fid) % CLIP_LEN
            if tail_len >= 10:
                clip_num += 1
            elif 0 < tail_len <10:
                end_fid = end_fid - tail_len
            
            s_bboxes = []
            o_bboxes = []
            for fid in range(begin_fid, end_fid):
                fid = int(fid)
                s_bboxes.append(s_traj['trajectory'][(fid)])
                o_bboxes.append(o_traj['trajectory'][(fid)]) 

            pair_data = {
                    'sbj_scr': s_traj['score'],
                    'obj_scr': o_traj['score'],
                    'sbj_cls': s_traj['category'],
                    'obj_cls': o_traj['category'],
                    'sbj_traj': s_bboxes,
                    'obj_traj': o_bboxes,
                    'duration': [begin_fid, end_fid]
                    }


            pair_id += 1
            pair_feats = defaultdict(list)
            
            for clip_id in range(clip_num):
                clip = {} 
                clip["sbj_id"] = object2id[s_traj['category']]
                clip["obj_id"] = object2id[o_traj['category']]
                clip["begin_fid"] = begin_fid + clip_id*CLIP_LEN
                if clip_id == clip_num-1:
                    clip["sbj_traj"] = s_bboxes[clip_id*CLIP_LEN:]
                    clip["obj_traj"] = o_bboxes[clip_id*CLIP_LEN:]
                    clip["end_fid"] = end_fid
                else:
                    clip["sbj_traj"] = s_bboxes[clip_id*CLIP_LEN:(clip_id+1)*CLIP_LEN]
                    clip["obj_traj"] = o_bboxes[clip_id*CLIP_LEN:(clip_id+1)*CLIP_LEN]
                    clip["end_fid"] = begin_fid + (clip_id+1)*CLIP_LEN
                
                
                feats = feat_extractor.gen_feats(clip,patch_proj, global_proj,w,h)
                for type_ in ['rel_feat', 'mot_feat', 'clip_feat', 'bbox_feat']:
                    pair_feats[type_].append(feats[type_])
            
            data = [pair_feats, pair_data]
            pair_feats = data[0]
            item = {}
            for type_ in pair_feats:
                if type_ != 'clip_feat':
                    item[type_] = torch.tensor(pair_feats[type_]).unsqueeze(dim=0)
                else:
                    item[type_] = torch.stack(pair_feats[type_],dim=0).unsqueeze(dim=0).float()
            item['vid'] = video_name[0]
            item['pair_data'] = [data[1]]
            items[pair_id] = item
    return items
