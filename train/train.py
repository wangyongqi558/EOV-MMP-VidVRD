import numpy as np
from collections import defaultdict
from os.path import join
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# import visdom
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dataset"))
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from dataset import Dataset_new
from utils.parser_func import parse_args
from arguments import get_args_parser
from utils.video_relation_detection import evaluate
from utils.utils import get_feat_types, AverageMeter, get_logger, print_results
from utils.post_process import process_pred, association, format_
from utils.video_relation_detection_openvoc import eval_relation_detection_openvoc
import multiprocessing as mp
from methods import build_model
from end2end_model import End2End_Model
from model import Classifier
import warnings
from os.path import join
warnings.filterwarnings("ignore")
# from deep_sort_app import run
from AFLink.AppFreeLink import *

OUTPUT_LOG_DIR = '../output/log'
OUTPUT_CKPT_DIR = '../output/ckpt'
MODEL_CKPT_DIR = '../output/ckpt'


def seed_everything(seed = 3407):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_everything(3407)
    args = parse_args()

    from model_zoo.model_tuing_plus_repro_copy_new_cross_dataset import Model

    feat_types = get_feat_types(args)
    feat_config = "_"
    for type_ in feat_types:
        feat_config += type_.split("_")[0] + "_"
    env_config = 'baseline_fbce_' +\
        args.dataset+ \
        "_bs"+str(args.batch_size)+ \
        "_lr"+str(args.lr)+ \
        "_dim"+str(args.clip_emb_dim)+ \
        "_"+str(args.temp_model)+ \
        feat_config+args.ps
    # vis = visdom.Visdom(env=env_config)

    logger = get_logger(join(OUTPUT_LOG_DIR, env_config + '_train.log'))
    logger.info('Experiment Config: {}'.format(args))

    train_dataset = Dataset_new(args, "train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_det = Dataset_new(args, "val")
    val_loader_det = DataLoader(val_dataset_det, batch_size=args.batch_size, shuffle=False)

    object_detection_model, object_detection_criterion, object_detection_postprocessors = build_model(args)
    checkpoint_path = join(MODEL_CKPT_DIR, 'checkpoint_vidvrd0059_new_1e-5.pth')
    checkpoint = torch.load(checkpoint_path)
    object_detection_model.load_state_dict(checkpoint['model'])

    relationship_classification_model = Model(args).cuda()
    checkpoint_path = join(MODEL_CKPT_DIR, 'baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth')
    ckpt = torch.load(checkpoint_path)
    pretrained_dict = ckpt['state_dict']
    model_dict = relationship_classification_model.state_dict()
    model_dict.update(pretrained_dict)
    relationship_classification_model.load_state_dict(model_dict)

    object_classifer = Classifier(args).cuda()
    checkpoint_path = join(MODEL_CKPT_DIR, 'vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth')
    checkpoint = torch.load(checkpoint_path)
    object_classifer.load_state_dict(checkpoint['state_dict'])

    model = End2End_Model(args, object_detection_model, object_classifer, relationship_classification_model,
                          object_detection_criterion, object_detection_postprocessors).cuda()

    for name, param in model.named_parameters():
        if "text_encoder" in name:
            param.requires_grad_(False)
        elif 'backbone' in name:
            param.requires_grad_(False)
        elif 'pre_classifier' in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20,25], gamma=0.1)
    
    sort_model = PostLinker()
    sort_model.load_state_dict(torch.load(args.path_AFLink))
    epoch_loss = AverageMeter()

    best_mmap = 0
    for epoch in range(args.start_epoch+1, args.max_epoch+1):      
 
        model.train()
        batch_loss = AverageMeter()
        for idx, data in enumerate(tqdm(train_loader)):
            model(data, sort_model)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        model.eval()
        map_list = []
        
        model.modelC.tgt_split = 'all'
        pred_rels = defaultdict(list)
        with torch.no_grad():
            for data in tqdm(val_loader_det):
                final_results = model(data,sort_model)
                for final_result in final_results:
                    pre_preds=final_result['pre_preds'] 
                    sbj_preds=final_result['sbj_preds'] 
                    obj_preds=final_result['obj_preds'] 
                    seq_lens=final_result['seq_lens'] 
                    vids=final_result['video_name'] 
                    pair_data = final_result['pair_data']
                    for seq_id, seq_len in enumerate(seq_lens):
                        clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                        pred_rels[vids[seq_id]].extend(association(clip_rels))
        for vid in pred_rels:
            pred_rels[vid] = format_(args, pred_rels[vid])
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels, rt_hit_infos=True)
        logger.info("SGDet and All split     | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        map_list.append(mean_ap*100)
    
    
        model.modelC.tgt_split = 'novel'
        pred_rels = defaultdict(list)
        with torch.no_grad():
            for data in tqdm(val_loader_det):
                final_results = model(data,sort_model)
                for final_result in final_results:
                    pre_preds=final_result['pre_preds'] 
                    sbj_preds=final_result['sbj_preds'] 
                    obj_preds=final_result['obj_preds'] 
                    seq_lens=final_result['seq_lens'] 
                    vids=final_result['video_name'] 
                    pair_data = final_result['pair_data']
                    for seq_id, seq_len in enumerate(seq_lens):
                        clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                        pred_rels[vids[seq_id]].extend(association(clip_rels))
        for vid in pred_rels:
            pred_rels[vid] = format_(args, pred_rels[vid])
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels, rt_hit_infos = True)
        logger.info("SGDet and Novel split   | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        map_list.append(mean_ap*100)
        
        mmap = sum(map_list)/2

        logger.info(f"Mean mAP: {mmap}, Best Mean mAP: {best_mmap}")
        if mmap > best_mmap:
            best_mmap = mmap
            state = {
                'map': map_list,
                'epoch': epoch,
                'config': args,
                'state_dict': model.state_dict()}
            ckpt_path = join(OUTPUT_CKPT_DIR, env_config + ".pth")
            torch.save(state, ckpt_path)
    
    print("================================Final Results======================================")
    logger.info('Best Epoch: {}, mAP List: {}'.format(state['epoch'], str(state['map'])))



