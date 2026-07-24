import argparse

def str2bool(v, bool):

    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Set OV-DETR', add_help=False)
    #
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--lr_backbone_names", default=["backbone.0"], type=str, nargs="+")
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    #

    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--img_size', default=336, type=int)
    parser.add_argument('--eval_size', default=336, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Learning rate schedule parameters
    parser.add_argument("--sgd", action="store_true")
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                         help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                         help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                         help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * model setting
    parser.add_argument('--backbone_name', default='local_deit_base', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained', default='imagenet', type=str,
                        help="set imagenet pretrained model path if not train yolos from scatch")

    # * Matcher set_cost_class set_cost_bbox set_cost_giou
    parser.add_argument('--set_cost_class', default=3, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # * Dataset
    parser.add_argument('--dataset_file', default='open_vidvrd')
    parser.add_argument('--coco_path', default='../dataset/COCO2017_Seg', type=str)
    parser.add_argument('--lvis_path', default='../dataset/COCO2017_Seg', type=str)
    parser.add_argument('--vidvrd_path', default='../dataset', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    # * Device and Log
    parser.add_argument('--output_dir', default='../output/log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=True, type=lambda x: (str(x).lower() == 'true'), help='eval mode')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # * Training setup
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3457', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--num_workers', default=2, type=int)

    # * Pos encodig
    parser.add_argument('--position_embedding', default='sine', type=str)

    # * Transformer
    parser.add_argument('--pos_dim', default=256, type=int, help="Size of the embeeding for pos")
    parser.add_argument('--reduced_dim', default=256, type=int, help="Size of the embeddings for head")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, # Deform-DETR: 1024, DETR: 2048
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # * Deformable Attention
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=3, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')


    ####### ViDT Params
    parser.add_argument('--method', default='ov_prompt', type=str, help='method names in {vidt, vidt_wo_neck}')
    parser.add_argument("--det_token_num", default=300, type=int, help="Number of det token in the body backbone")

    # * Auxiliary Techniques
    parser.add_argument('--aux_loss', default=True, type=lambda x: (str(x).lower() == 'true'), help='auxiliary decoding loss')
    parser.add_argument('--with_box_refine', default=True, type=lambda x: (str(x).lower() == 'true'), help='iterative box refinement')


    # cross-scale fusion
    parser.add_argument('--cross_scale_fusion', default=True, type=lambda x: (str(x).lower() == 'true'), help='use of scale fusion')
    #######

    # * Logs
    parser.add_argument('--n_iter_to_acc', default=1, type=int, help='gradient accumulation step size')
    parser.add_argument('--print_freq', default=10, type=int, help='number of iteration to print training logs')

    # * backbone freeze
    parser.add_argument('--freeze', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='freezing backbone')

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
    parser.add_argument("--clip_feat_path", default="../output/data/clip_L14_336px_feat_vidvrd.pkl", type=str,)
    parser.add_argument("--prob", default=0.75, type=float)
    parser.add_argument("--amp", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval_period", default=1, type=int)

    # our
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
    parser.add_argument('--new_loss_coef', default=2, type=float)
    # token label
    parser.add_argument('--return_masks', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--token_label', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='use of token label loss')
    parser.add_argument('--token_loss_coef', default=2, type=float)

    # new params
    parser.add_argument('--all_train_token', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--clip_backbone', default='ViT-L/14@336px', type=str)
    parser.add_argument('--temperature', default=0.01, type=float)
    parser.add_argument('--clip_h', default=336, type=int)
    parser.add_argument('--clip_w', default=336, type=int)
    parser.add_argument('--bg', default=True, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--use_adapter', default=True, type=bool)

    # SOLQ
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


    return parser
