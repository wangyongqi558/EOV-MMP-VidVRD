import os
from os.path import join
import json
from collections import defaultdict

import numpy as np

from utils.parser_func import parse_args

def format_gt_trajs(data):
    tid2class = {}
    for obj in data["subject/objects"]:
        tid2class[obj['tid']] = object2id[obj['category']]
    trajs = data["trajectories"]

    traj_queues = defaultdict(dict)
    traj_list = []
    for frame_id, frame_objs in enumerate(trajs):
        unconnected_tids = list(traj_queues.keys())
        for obj in frame_objs: 
            if obj["tid"] in unconnected_tids:
                unconnected_tids.remove(obj["tid"])
            else:
                traj_queues[obj["tid"]]["traj"] = []
                traj_queues[obj["tid"]]["begin_fid"] = frame_id
            traj_queues[obj["tid"]]["traj"].append([obj["bbox"]["xmin"],obj["bbox"]["ymin"],obj["bbox"]["xmax"],obj["bbox"]["ymax"]])
        for tid in unconnected_tids:
            traj_queues[tid]["end_fid"] = frame_id
            traj_list.append({
                "class":tid2class[tid], 
                "score":1.0, 
                "traj":traj_queues[tid]["traj"],
                "begin_fid":traj_queues[tid]["begin_fid"],
                "end_fid":traj_queues[tid]["end_fid"]})
            traj_queues.pop(tid)
    tids = list(traj_queues.keys())
    for tid in tids:
            traj_queues[tid]["end_fid"] = len(trajs)
            traj_list.append({
                "class":tid2class[tid], 
                "score":1.0, 
                "traj":traj_queues[tid]["traj"],
                "begin_fid":traj_queues[tid]["begin_fid"],
                "end_fid":traj_queues[tid]["end_fid"]})
            traj_queues.pop(tid)

    return traj_list

def format_vru21_trajs(data):
    



def gen_pairs(trajs):
    traj_queues = defaultdict(dict)
    traj_list = []
    for frame_id, frame_objs in enumerate(trajs):
        unconnected_tids = list(traj_queues.keys())
        for obj in frame_objs: 
            if obj["tid"] in unconnected_tids:
                unconnected_tids.remove(obj["tid"])
            else:
                traj_queues[obj["tid"]]["traj"] = []
                traj_queues[obj["tid"]]["begin_fid"] = frame_id
            traj_queues[obj["tid"]]["traj"].append([obj["bbox"]["xmin"],obj["bbox"]["ymin"],obj["bbox"]["xmax"],obj["bbox"]["ymax"]])
        for tid in unconnected_tids:
            traj_queues[tid]["end_fid"] = frame_id
            traj_list.append({"tid":tid,"traj":traj_queues[tid]["traj"],"begin_fid":traj_queues[tid]["begin_fid"],"end_fid":traj_queues[tid]["end_fid"]})
            traj_queues.pop(tid)
    tids = list(traj_queues.keys())
    for tid in tids:
            traj_queues[tid]["end_fid"] = len(trajs)
            traj_list.append({"tid":tid,"traj":traj_queues[tid]["traj"],"begin_fid":traj_queues[tid]["begin_fid"],"end_fid":traj_queues[tid]["end_fid"]})
            traj_queues.pop(tid)

    trajs = traj_list
    pairs = []
    for sbj in trajs:
        for obj in trajs:
            if sbj["tid"] == obj["tid"]:
                continue
            if sbj["end_fid"] < obj["begin_fid"] or obj["end_fid"] < sbj["begin_fid"]:
                continue
            begin_fid = max(sbj["begin_fid"], obj["begin_fid"])
            end_fid = min(sbj["end_fid"], obj["end_fid"])
            if end_fid - begin_fid < 10:
                continue
            pairs.append({
                "sbj_tid":sbj["tid"],
                "sbj_traj":[sbj["traj"][i] for i in range(begin_fid-sbj["begin_fid"],end_fid-sbj["begin_fid"])],
                "obj_tid":obj["tid"],
                "obj_traj":[obj["traj"][i] for i in range(begin_fid-obj["begin_fid"],end_fid-obj["begin_fid"])],
                "begin_fid":begin_fid,
                "end_fid":end_fid
                })

    return pairs
    

if __name__ == '__main__':

    args = parse_args()
    ROOT = join("..","dataset",args.dataset)
    object2id = json.load(open(join(ROOT, 'data', 'object2id.json'),'r'))


    # format trajs from gt train annotations
    split = 'train'
    trajs = {}
    pkg_list = os.listdir(join(ROOT, 'anno', split))
    for pkg_name in pkg_list:
        vid_list = os.listdir(join(ROOT, 'anno', split, pkg_name))
        for vid_name in vid_list:
            data = json.load(open(join(ROOT, 'anno', split, pkg_name, vid_name),'r'))
            trajs[vid_name] = format_gt_trajs(data)
    with open('.json','w') as f:
        json.dump(,f)
    
    # format trajs from vru19 val object detection results
        
    # format trajs from vru21 val object detection results
    trajs = {}
    data_paths = os.listdir(join(ROOT, 'data', 'val_object_trajectories_vru21'))
    for data_path in data_paths:
        data = np.load(open(join(ROOT, 'data', 'val_object_trajectories_vru21', data_path)))
        vid_name = data_path.split('_')[-1].split('.')[0]
        trajs[vid_name] = format_vru21_trajs(data)





