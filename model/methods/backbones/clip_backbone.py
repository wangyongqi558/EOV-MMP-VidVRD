from clip_tagclip import clip
import torch
import torch.nn as nn

class CLIPFeatureEncoder(nn.Module):
    def __init__(self, BACKBONE_NAME):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_tagclip, _ = clip.load("ViT-L/14@336px", device=self.device)
        self.embed_dim = 1024

    def forward(self, image_resize):
        self.model_tagclip.eval()
        with torch.no_grad():
            h, w = image_resize.shape[-2], image_resize.shape[-1]
            image_features, _ = self.model_tagclip.encode_image_tagclip(image_resize, h, w, attn_mask=1)

            # feature = self.model.encode_image(image_resize)
        
        return image_features[:,1:,:]

def build_backbone(args):
    model = CLIPFeatureEncoder(args.clip_backbone)
    return model