import torchvision.transforms as T
from torchvision.transforms import functional as F
import random

class TrainTransforms:
    def __init__(self, resize=(800, 800), rot_prob=0.5, max_rotation=360):
        self.resize = resize
        self.rot_prob = rot_prob
        self.max_rotation = max_rotation
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __call__(self, image, target=None):
        image = F.resize(image, self.resize)

        if random.random() < self.rot_prob:
            angle = random.uniform(0, self.max_rotation)
            image = F.rotate(image, angle)
            if target is not None and 'boxes' in target:
                boxes = target['boxes']
                target['boxes'] = boxes  # placeholder

        if random.random() < 0.5:
            image = F.hflip(image)
            if target is not None and 'boxes' in target:
                w, _ = image.size
                boxes = target['boxes']
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes

        image = F.to_tensor(image)
        image = self.normalize(image)

        if target is not None:
            return image, target
        return image

class EvalTransforms:
    def __init__(self, resize=(800, 800)):
        self.resize = resize
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __call__(self, image, target=None):
        image = F.resize(image, self.resize)
        image = F.to_tensor(image)
        image = self.normalize(image)
        if target is not None:
            return image, target
        return image
