import os
import json
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
PURPLE = (0x44, 0x01, 0x54)
YELLOW = (0xFD, 0xE7, 0x25)

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, image_transform, mask_transform, support=1, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False,
                 ensemble: bool = False, purple: bool = False):
        self.fold = fold
        self.support = support
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOCdevkit/VOC2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)
        query_name, class_sample = self.img_metadata[idx]
        support_names = []

        while len(support_names) < self.support:
            candidate = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if candidate != query_name and candidate not in support_names:
                support_names.append(candidate)

        query_img = self.read_img(query_name)
        query_cmask = self.read_mask(query_name)

        if self.image_transform:
            query_img = self.image_transform(query_img)
        query_mask, _ = self.extract_ignore_idx(query_cmask, class_sample, purple=self.purple)
        if self.mask_transform:
            query_mask = self.mask_transform(query_mask)

        support_imgs, support_masks = [], []
        for support_name in support_names:
            s_img = self.read_img(support_name)
            s_mask = self.read_mask(support_name)

            if self.image_transform:
                s_img = self.image_transform(s_img)
            s_mask, _ = self.extract_ignore_idx(s_mask, class_sample, purple=self.purple)
            if self.mask_transform:
                s_mask = self.mask_transform(s_mask)

            support_imgs.append(s_img)
            support_masks.append(s_mask)
            
        target_mask = query_mask

        return {
            "support_images": support_imgs,
            "support_masks": support_masks,
            "query_image": query_img,
            "query_mask": query_mask,
            "support_ids": support_names,
            "query_id": query_name
        }

    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.0)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x, y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary

    def read_mask(self, img_name):
        return Image.open(os.path.join(self.ann_path, img_name) + '.png')

    def read_img(self, img_name):
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self):
        def read_metadata(split, fold_id):
            cwd = os.path.dirname(os.path.abspath(__file__))
            fold_file = os.path.join(cwd, f'splits/pascal/{split}/fold{fold_id}.txt')
            with open(fold_file, 'r') as f:
                lines = f.read().strip().split('\n')
            return [[line.split('__')[0], int(line.split('__')[1]) - 1] for line in lines]

        metadata = read_metadata('val', self.fold)
        print('Total (val) images are : %d' % len(metadata))
        return metadata

    def build_img_metadata_classwise(self):
        classwise = {i: [] for i in range(self.nclass)}
        for name, class_id in self.img_metadata:
            classwise[class_id].append(name)
        return classwise

    def write_jsonl(self, out_path):
        """Writes a JSONL file with path references for inputs and targets."""
        data = []
        for idx in range(len(self)):
            query_name, class_sample = self.img_metadata[idx]
            support_names = []

            while len(support_names) < 2:
                candidate = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
                if candidate != query_name and candidate not in support_names:
                    support_names.append(candidate)

            input_paths = []
            for s_name in support_names:
                input_paths.append(os.path.join(self.img_path, s_name + ".jpg"))
                input_paths.append(os.path.join(self.ann_path, s_name + ".png"))

            input_paths.append(os.path.join(self.img_path, query_name + ".jpg"))
            target_paths = [os.path.join(self.ann_path, query_name + ".png")]

            data.append({"input": input_paths, "target": target_paths})

        with open(out_path, 'w') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')
        print(f"Saved JSONL file with {len(data)} examples to: {out_path}")
