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
