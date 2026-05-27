""" Module handling the Image Dataset used for evaluation of Neuralizer on downstream tasks""" 

import os
import csv
from PIL import Image
from torch.utils.data import Dataset

class Imagedataset(Dataset):
    def __init__(self, csv_path, image_dir, label_dir, image_transform, support=2):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.support = support  # number of support pairs to use
        self.entries = []

        # Load CSV rows
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) < support + 1:
                    continue
                self.entries.append(row)

    def __len__(self):
        return len(self.entries)
        
    def __getitem__(self, idx):
        row = self.entries[idx]
        support_names = row[:self.support * 2]  # Each support has 2 files
        query_names = row[-2:]  # Query also has 2 files (image,label)

        support_imgs = []
        support_lbls = []

        for i in range(0, len(support_names), 2):
            image_path = os.path.join(self.image_dir, support_names[i])
            label_path = os.path.join(self.label_dir, support_names[i + 1])

            support_img = Image.open(image_path).convert("RGB")
            support_lbl = Image.open(label_path).convert("RGB")

            if self.image_transform:
                support_img = self.image_transform(support_img)
                support_lbl = self.image_transform(support_lbl)

            support_imgs.append(support_img)
            support_lbls.append(support_lbl)

        # Query
        query_image_path = os.path.join(self.image_dir, query_names[0])
        query_label_path = os.path.join(self.label_dir, query_names[1])

        query_img = Image.open(query_image_path).convert("RGB")
        query_lbl = Image.open(query_label_path).convert("RGB")

        if self.image_transform:
            query_img = self.image_transform(query_img)
            query_lbl = self.image_transform(query_lbl)


        return {
            "support_images": support_imgs,
            "support_labels": support_lbls,
            "query_image": query_img,
            "query_label": query_lbl,
            "support_ids": support_names,
            "query_id": query_names
        }
