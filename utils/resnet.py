import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class ResNet(nn.Module):
    
    def __init__(self):
        super(ResNet, self).__init__()

        fasterrcnn = fasterrcnn_resnet50_fpn(pretrained=True)
        self.transform = fasterrcnn.transform
        self.backbone = fasterrcnn.backbone
        self.roi_pool = fasterrcnn.roi_heads.box_roi_pool
        self.roi_head = fasterrcnn.roi_heads.box_head

    def resize_boxes(self, boxes, original_size, new_size):
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def forward(self, images, bboxes):
        
        images = images.cuda()
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, _ = self.transform(images)
        for i in range(len(bboxes)):
            bboxes[i] = bboxes[i].cuda()
            bboxes[i] = self.resize_boxes(bboxes[i], original_image_sizes[i], images.image_sizes[i])
        
        features = self.backbone(images.tensors)
        features = self.roi_pool(features, bboxes, images.image_sizes)
        features = self.roi_head(features)
        return features