from __future__ import annotations

import h5py
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MediumImagenetHDF5Dataset(Dataset):
    def __init__(
        self,
        img_size,
        split: str = "train",
        filepath: str = "/data/nmep/cv/medium-imagenet-96.hdf5",
        augment: bool = True,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.augment = augment
        self.input_size = img_size
        self.transform = self._get_transforms()
        self.file = h5py.File(filepath, "r")

    def __getitem__(self, index):
        image = self.file[f"images-{self.split}"][index]
        if self.split != "test":
            label = self.file[f"labels-{self.split}"][index]
        else:
            label = -1
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.file[f"images-{self.split}"])

    def _get_transforms(self):
        transform = []
        transform.append(lambda x: torch.tensor(x / 256).to(torch.float))
        if self.split == "train" and self.augment: # bug: need to augment before normalization
            transform.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                ]
            )
        normalization = torch.Tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform.append(transforms.Normalize(normalization[0], normalization[1]))
        transform.append(transforms.Resize([self.input_size] * 2))
        # if self.split == "train" and self.augment:
        #     transform.extend(
        #         [
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        #         ]
        #     )
        return transforms.Compose(transform)


class CIFAR10Dataset(Dataset):
    def __init__(self, img_size=32, train=True, aug_scheme="baseline"):
        self.train = train
        self.img_size = img_size
        self.aug_scheme = aug_scheme

        self.transform = self._get_transforms()
        self.dataset = CIFAR10(root="/data/nmep/cv/cifar10", train=self.train, download=True)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

    def _get_transforms(self):
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        resize = transforms.Resize([self.img_size] * 2)

        # No aug on val set
        if not self.train:
            return transforms.Compose([transforms.ToTensor(), normalize, resize])
        # Aug on train set
        if self.aug_scheme == "baseline":
            pil_aug = [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
            ]
            tensor_aug = []
        elif self.aug_scheme == "crop_erase":
            pil_aug = [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
            ]
            tensor_aug = [transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))]
        elif self.aug_scheme == "trivialaugment":
            pil_aug = [
                transforms.TrivialAugmentWide(),
                transforms.RandomHorizontalFlip(),
            ]
            tensor_aug = []
        else:
            raise NotImplementedError(f"Unknown augmentation scheme: {self.aug_scheme}")

        # PIL augmentation -> tensor -> normalize -> resize -> tensor augmentation
        return transforms.Compose(pil_aug + [transforms.ToTensor(), normalize, resize] + tensor_aug)
