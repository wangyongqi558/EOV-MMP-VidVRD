import numpy as np
import torch
from pkg_resources import packaging

from transformers import CLIPModel, CLIPConfig, CLIPVisionModel
from methods.backbones.clip_vit_det import local_deit_base


config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")


_dict = model.state_dict()
print(_dict.keys())
_new_dict = {}

for layer in range(12):
    # qkv
    q_weight = 'vision_model.encoder.layers.' + str(layer) + '.self_attn.q_proj.weight'
    k_weight = 'vision_model.encoder.layers.' + str(layer) + '.self_attn.k_proj.weight'
    v_weight = 'vision_model.encoder.layers.' + str(layer) + '.self_attn.v_proj.weight'
    q_bias = 'vision_model.encoder.layers.' + str(layer) + '.self_attn.q_proj.bias'
    k_bias = 'vision_model.encoder.layers.' + str(layer) + '.self_attn.k_proj.bias'
    v_bias = 'vision_model.encoder.layers.' + str(layer) + '.self_attn.v_proj.bias'

    _new_dict['blocks.' + str(layer) + '.attn.qkv.weight'] \
        = torch.cat([_dict[q_weight], _dict[k_weight], _dict[v_weight]], dim=0)
    _new_dict['blocks.' + str(layer) + '.attn.qkv.bias'] \
        = torch.cat([_dict[q_bias], _dict[k_bias], _dict[v_bias]], dim=0)

    # others
    proj_weight = 'vision_model.encoder.layers.' + str(layer) + '.self_attn.out_proj.weight'
    proj_bias = 'vision_model.encoder.layers.' + str(layer) + '.self_attn.out_proj.bias'
    norm1_weight = 'vision_model.encoder.layers.' + str(layer) + '.layer_norm1.weight'
    norm1_bias = 'vision_model.encoder.layers.' + str(layer) + '.layer_norm1.bias'
    fc1_weight = 'vision_model.encoder.layers.' + str(layer) + '.mlp.fc1.weight'
    fc1_bias = 'vision_model.encoder.layers.' + str(layer) + '.mlp.fc1.bias'
    fc2_weight = 'vision_model.encoder.layers.' + str(layer) + '.mlp.fc2.weight'
    fc2_bias = 'vision_model.encoder.layers.' + str(layer) + '.mlp.fc2.bias'
    norm2_weight = 'vision_model.encoder.layers.' + str(layer) + '.layer_norm2.weight'
    norm2_bias = 'vision_model.encoder.layers.' + str(layer) + '.layer_norm2.bias'

    _new_dict['blocks.' + str(layer) + '.attn.proj.weight'] = _dict[proj_weight]
    _new_dict['blocks.' + str(layer) + '.attn.proj.bias'] = _dict[proj_bias]
    _new_dict['blocks.' + str(layer) + '.norm1.weight'] = _dict[norm1_weight]
    _new_dict['blocks.' + str(layer) + '.norm1.bias'] = _dict[norm1_bias]
    _new_dict['blocks.' + str(layer) + '.mlp.fc1.weight'] = _dict[fc1_weight]
    _new_dict['blocks.' + str(layer) + '.mlp.fc1.bias'] = _dict[fc1_bias]
    _new_dict['blocks.' + str(layer) + '.mlp.fc2.weight'] = _dict[fc2_weight]
    _new_dict['blocks.' + str(layer) + '.mlp.fc2.bias'] = _dict[fc2_bias]
    _new_dict['blocks.' + str(layer) + '.norm2.weight'] = _dict[norm2_weight]
    _new_dict['blocks.' + str(layer) + '.norm2.bias'] = _dict[norm2_bias]

    # global
    _new_dict['cls_token'] = _dict['vision_model.embeddings.class_embedding'].unsqueeze(0).unsqueeze(0)
    _new_dict['pos_embed'] = _dict['vision_model.embeddings.position_embedding.weight'].unsqueeze(0)
    _new_dict['patch_embed.proj.weight'] = _dict['vision_model.embeddings.patch_embedding.weight']
    _new_dict['pre_layernorm.weight'] = _dict['vision_model.pre_layrnorm.weight']
    _new_dict['pre_layernorm.bias'] = _dict['vision_model.pre_layrnorm.bias']
    _new_dict['norm.weight'] = _dict['vision_model.post_layernorm.weight']
    _new_dict['norm.bias'] = _dict['vision_model.post_layernorm.bias']

timm_model, _ = local_deit_base(pretrained='scratch')
timm_model.load_state_dict(_new_dict, strict=False)

output_dict = {'model': _new_dict}
torch.save(output_dict, 'checkpoints/timm-clip-vit-base-patch16.pth')

sys.exit(1)
import clip
print(clip.available_models())

model, preprecess = clip.load('ViT-B/32')
print(model)
stat_dict = model.state_dict()
for key, value in Rstat_dict.items():
    print(key)
