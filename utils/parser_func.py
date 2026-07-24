import argparse
import ast
from math import sqrt

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Build a video relation detection network')

    parser.add_argument('--dataset', dest='dataset',
                        help='The name of dataset to be used',
                        type=str, default='vidvrd')
    parser.add_argument('--clip_len', dest='clip_len',
                        help='Atomatic clip length in training and test',
                        type=int, default=30)
    parser.add_argument('--use_unlabeld_pair', dest='use_unlabeld_pair',
                        help='Whether to use unlabeled trajectory pairs for training',
                        action='store_true')                                                    
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size in training',
                        type=int, default=1)
    parser.add_argument('--batch_size_eval', dest='batch_size_eval',
                        help='Batch size in evluation',
                        type=int, default=32)
    parser.add_argument('--lr', dest='lr',
                        help='Initial learning rate',
                        type=float, default=0.01)
    parser.add_argument('--dropout_r', dest='dropout_r',
                        help='Dropout value',
                        type=float, default=0.5)
    parser.add_argument('--num_layers', dest='num_layers',
                        help='Lstm Number of Layers',
                        type=int, default=2)
    parser.add_argument('--momentum', dest='momentum',
                        help='The momentum of SGD optimizer',
                        type=float, default=0.9)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='Model weight decay value',
                        type=float, default=0.0001)
    parser.add_argument('--max_epoch', dest='max_epoch',
                        help='The epoch to stop',
                        type=int, default=40)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='The epoch to run from',
                        type=int, default=0)
    parser.add_argument('--resume', dest='resume',
                        help='Whether to resume the training',
                        action='store_true')
    parser.add_argument('--ckpt_path', dest='ckpt_path',
                        help='The checkpoint saved path',
                        type=str, default="")
    parser.add_argument('--print_freq', dest='print_freq',
                        help='The batch frequence of printing info',
                        type=int, default=100)
    parser.add_argument('--clip_top_n', dest='clip_top_n',
                        help='The top n predictions of clip to be saved',
                        type=int, default=20)
    parser.add_argument('--max_per_video', dest='max_per_video',
                        help='Max number of relations for each video to be saved',
                        type=int, default=200)
    parser.add_argument('--ps', dest='ps',
                        help='The P.S. information for this training process',
                        type=str, default="")
    parser.add_argument('--train_traj', dest='train_traj',
                        help='The trajectories for training split',
                        type=str, default="gt")
    parser.add_argument('--val_traj', dest='val_traj',
                        help='The trajectories source for validation split',
                        type=str, default="gt")
    parser.add_argument('--test_traj', dest='test_traj',
                        help='The trajectories source for testing split',
                        type=str, default="gt")
    parser.add_argument('--use_prior', dest='use_prior',
                        help='Wether to use prior or not',
                        action='store_true')
    parser.add_argument('--temp_model', dest='temp_model',
                        help='The temporal model used to encoding context',
                        default=None)
    parser.add_argument('--obj_loss_weight', dest='obj_loss_weight',
                        help='The loss weight factor for object loss',
                        type=float, default=0.1)
    parser.add_argument('--int_loss_weight', dest='int_loss_weight',
                        help='The loss weight factor for interactive loss',
                        type=float, default=0.1)

    parser.add_argument('--rel_feat', dest='rel_feat',
                        help='Use relative location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--mask_feat', dest='mask_feat',
                        help='Use mask location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--lan_feat', dest='lan_feat',
                        help='Use language feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--v2d_feat', dest='v2d_feat',
                        help='Use visual 2d feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--mot_feat', dest='mot_feat',
                        help='Use motion location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--v3d_feat', dest='v3d_feat',
                        help='Use visual 3d feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--clip_feat', dest='clip_feat',
                        help='Use clip visual feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--intern_feat', dest='intern_feat',
                        help='Use intern visual feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--bbox_feat', dest='bbox_feat',
                        help='Use bbox location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--ptm_mode', dest='ptm_mode',
                        help='Use vision only or vision-text model to train',
                        type=str, default='vision_only')
    parser.add_argument('--src_split', dest='src_split',
                        help='Use what data for training',
                        type=str, default='all')
    parser.add_argument('--tgt_split', dest='tgt_split',
                        help='Use what data for evaluation',
                        type=str, default='all')

    parser.add_argument('--rel_emb_dim', dest='rel_emb_dim',
                        help='The dimension of relative location feature',
                        type=int, default=256)
    parser.add_argument('--mask_emb_dim', dest='mask_emb_dim',
                        help='The dimension of mask location feature',
                        type=int, default=256)
    parser.add_argument('--lan_emb_dim', dest='lan_emb_dim',
                        help='The dimension of language feature',
                        type=int, default=256)
    parser.add_argument('--v2d_emb_dim', dest='v2d_emb_dim',
                        help='The dimension of visual 2d feature',
                        type=int, default=512)
    parser.add_argument('--mot_emb_dim', dest='mot_emb_dim',
                        help='The dimension of mition location feature',
                        type=int, default=256)
    parser.add_argument('--v3d_emb_dim', dest='v3d_emb_dim',
                        help='The dimension of visual 3d feature',
                        type=int, default=512)
    parser.add_argument('--clip_hidden_dim', dest='clip_hidden_dim',
                        help='The dimension of clip embedding hidden layer',
                        type=int, default=1024)
    parser.add_argument('--clip_emb_dim', dest='clip_emb_dim',
                        help='The dimension of clip embedding output layer',
                        type=int, default=512)
    parser.add_argument('--temp_out_dim', dest='temp_out_dim',
                        help='The dimension of temporal unit hidden layer',
                        type=int, default=512)     
    
    #Prompt-OV
    parser.add_argument('--img_size', default=336, type=int)
    parser.add_argument('--eval_size', default=336, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm') 

    parser.add_argument('--set_cost_class', default=3, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class") 
    parser.add_argument('--dataset_file', default='open_vidvrd')
    parser.add_argument('--vidvrd_path', default='../dataset/vidvrd', type=str)
    parser.add_argument('--remove_difficult', action='store_true')     
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')    
    parser.add_argument('--eval', default=True, type=lambda x: (str(x).lower() == 'true'), help='eval mode')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--pos_dim', default=256, type=int, help="Size of the embeeding for pos")
    parser.add_argument('--reduced_dim', default=256, type=int, help="Size of the embeddings for head")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, # Deform-DETR: 1024, DETR: 2048
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout_d', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=3, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--method', default='ov_prompt', type=str, help='method names in {vidt, vidt_wo_neck}')
    parser.add_argument("--det_token_num", default=300, type=int, help="Number of det token in the body backbone")

    # * Auxiliary Techniques
    parser.add_argument('--aux_loss', default=True, type=lambda x: (str(x).lower() == 'true'), help='auxiliary decoding loss')
    parser.add_argument('--with_box_refine', default=True, type=lambda x: (str(x).lower() == 'true'), help='iterative box refinement')
    parser.add_argument('--cross_scale_fusion', default=True, type=lambda x: (str(x).lower() == 'true'), help='use of scale fusion')
    parser.add_argument('--n_iter_to_acc', default=1, type=int, help='gradient accumulation step size')

    # * ov detr [default setup]
    parser.add_argument('--cache_mode', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='enable cache mode for data loading')
    parser.add_argument("--label_map", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='enable label map')
    parser.add_argument("--dilation", default=False, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument("--enc_n_points", default=4, type=int)
    parser.add_argument("--two_stage", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--max_len", default=15, type=int)
    parser.add_argument("--feature_loss_coef", default=2, type=float) # -> 32 64.
    parser.add_argument("--clip_feat_path", default="../dataset/vidvrd/data/clip_L14_feat_vidvrd.pkl", type=str,)
    parser.add_argument("--prob", default=0.75, type=float)
    parser.add_argument("--amp", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument("--distil_clip_embed", default=True, type=lambda x: (str(x).lower() == 'true'))

    # segmentor
    parser.add_argument('--masks', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--with_vector', default=False, type=lambda x: (str(x).lower() == 'true'))

    ## New losses
    # iou-aware
    parser.add_argument('--iou_aware', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use of iou-aware loss')
    parser.add_argument('--iouaware_loss_coef', default=2, type=float)
    parser.add_argument('--new_loss', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use of new loss')
    parser.add_argument('--new_loss_coef', default=3, type=float)
    # token label
    parser.add_argument('--return_masks', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--token_label', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='use of token label loss')
    parser.add_argument('--token_loss_coef', default=2, type=float)
    parser.add_argument('--all_train_token', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--clip_backbone', default='ViT-L/14@336px', type=str)
    parser.add_argument('--temperature', default=0.01, type=float)
    parser.add_argument('--clip_h', default=336, type=int)
    parser.add_argument('--clip_w', default=336, type=int)
    parser.add_argument('--bg', default=True, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--use_adapter', default=True, type=bool)
    parser.add_argument('--vector_hidden_dim', default=1024, type=int)
    parser.add_argument('--vector_loss_coef', default=3, type=float)
    parser.add_argument('--n_keep', default=256, type=int,
                        help="Number of coeffs to be remained")
    parser.add_argument('--gt_mask_len', default=128, type=int,
                        help="Size of target mask")
    parser.add_argument('--no_vector_loss_norm', default=False, action='store_true')
    parser.add_argument('--activation', default='relu', type=str, help="Activation function to use")
    parser.add_argument('--vector_start_stage', default=0, type=int)
    parser.add_argument('--loss_type', default='l1', type=str)
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--train_split', dest='train_split',
                        help='train_split',
                        type=str, default="base")
    parser.add_argument('--test_split', dest='test_split',
                        help='test_split',
                        type=str, default="all")
    parser.add_argument('--N_CTX', dest='N_CTX',
                        help='number of context',
                        type=int, default=16)
    parser.add_argument(
            '--BoT',dest='BoT',help='Replacing the original feature extractor with BoT',
            type=bool, default=True)
    parser.add_argument(
            '--ECC',dest='ECC',
            type=bool,
            default=True,
            help='CMC model'
        )
            
    parser.add_argument(
            '--NSA',dest='NSA',type=bool,
            default=True,
            help='NSA Kalman filter'
        )
    parser.add_argument(
            '--EMA',dest='EMA',type=bool,
            default=True,
            help='EMA feature updating mechanism'
        )
    parser.add_argument(
            '--MC',dest='MC',type=bool,
            default=True,
            help='Matching with both appearance and motion cost'
        )
    parser.add_argument(
            '--woC',dest='woC',type=bool,
            default=True,
            help='Replace the matching cascade with vanilla matching'
        )
    parser.add_argument(
            '--AFLink',dest='AFLink',type=bool,
            default=True,
            help='Appearance-Free Link'
        )
    parser.add_argument(
            '--GSI',dest='GSI',
            action='store_true',
            help='Gaussian-smoothed Interpolation'
        )
    parser.add_argument(
            '--path_AFLink',dest='path_AFLink',type=str,
            default='../output/ckpt/AFLink_epoch20.pth'
        )
    parser.add_argument(
            '--EMA_alpha',dest='EMA_alpha',type=float,
            default=0.9
        )
    parser.add_argument(
            '--MC_lambda',dest='MC_lambda',type=float,
            default=0.98
        )
    parser.add_argument(
            '--min_confidence',dest='min_confidence',type=float,
            default=0.2
        )
    parser.add_argument(
            '--nms_max_overlap',dest='nms_max_overlap',type=float,
            default=1.0
        )
    parser.add_argument(
            '--min_detection_height',dest='min_detection_height',type=float,
            default=0
        )
    parser.add_argument(
            '--max_cosine_distance',dest='max_cosine_distance',type=float,
            default=0.35
        )
    parser.add_argument(
            '--nn_budget',dest='nn_budget',type=int,
            default=100
        )
    


         
    
    args = parser.parse_args()
    return args 
