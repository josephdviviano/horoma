from torchvision import transforms
from torchvision.transforms import functional
import numpy as np


class RandomQuarterTurn:
    """
    Rotate the image by a multiple of a quarter turn.
    """

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = np.random.choice([0, 90, 180, 270])

        return functional.rotate(img, angle)


class HoromaTransformsCR:
    """
    Performs all transforms at once for resnet.
    """

    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((28, 28)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.56121268, 0.20801756, 0.2602411], std=[0.22911494, 0.10410614, 0.11500103]),
        ])

    def __call__(self, img):
        return self.transforms(img)


class HoromaTransforms:
    """
    Performs all transforms at once.
    """

    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomQuarterTurn(),
            transforms.ToTensor()
        ])

    def __call__(self, img):
        return self.transforms(img)