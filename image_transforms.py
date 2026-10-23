import torchvision
from torchvision import transforms

def make_transforms(height,width,training: bool = False) -> transforms.Compose:
    image_transforms = []
    if training:
        image_transforms.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    image_transforms.extend(
        [
            transforms.Resize((height, width)),
            #TODO test with transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(image_transforms)