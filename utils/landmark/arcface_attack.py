import torch.nn.functional as F
from torchvision import transforms

class AttackArcFace:
    def __init__(self):
        super(AttackArcFace, self).__init__()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def preprocess(self, image):
        img_crop = F.interpolate(image, (224, 224), mode='bilinear')
        
        img_norm50 = self.normalize(img_crop)
        img_norm100 = self.normalize(image)
        
        # mode : nearest
        img_resize50 = F.interpolate(img_norm50, (112, 112))
        img_resize100 = F.interpolate(img_norm100, (112, 112))
        
        return img_resize50, img_resize100