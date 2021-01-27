# Script for defining dataset class
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import transforms
from PIL import Image
import pandas as pd


#Hardcoded for now
class_list=["__background__", "tumor"]
# Add class list for different datasets in case of multi label setting
labels_dataset={"aiims":class_list}

data_mode="Train"
aiimsImg_path="./data/AIIMS_Mammograms/%s" % data_mode
aiimsAnn_path="./data/AIIMS_Mammograms/anns/%s_labels.csv" % data_mode
valAiimsImg_path="./data/AIIMS_Mammograms/Val"
valAiimsAnn_path="./data/AIIMS_Mammograms/anns/Val_labels.csv"
# Add AIIMS and Public Dataset Class for Mammogram Data

class AIIMS(Dataset):
    def __init__(self, img_path=aiimsImg_path, ann_path=aiimsAnn_path, transform=None, trainIds=[], mode="unlabel"):
        self.imgPath = img_path
        self.annPath = ann_path
        self.transform = transform
        self.classes =  labels_dataset["aiims"]
        # self.trainIds = trainIds
        self.mode = mode
        self.df = pd.read_csv(self.annPath)
        self.image_ids = self.df["image_id"].unique().tolist()
        # self.image_ids.sort()

        if self.mode=="unlabel":
            self.image_ids = list(set(self.image_ids)-set(trainIds))
        else:
            self.image_ids = trainIds


    def __getitem__(self, idx):

        img_id = self.image_ids[idx]
        # print(image_name)
        anns = self.df[self.df["image_id"]==img_id]
        image_name = anns["filename"].iloc[0]
        img = Image.open(os.path.join(self.imgPath, image_name)).convert("RGB")

        
        labels=[]
        boxes=[]
        area=[]
        # iscrowd=[]

        for index, row in anns.iterrows():
            box_coords = row["xmin":"ymax"].to_numpy()
            if box_coords[0]>=box_coords[2] or box_coords[1]>=box_coords[3]:
                continue
            labels.append(self.classes.index(row["class"]))
            b_area = (box_coords[2]-box_coords[0])*(box_coords[3]-box_coords[1])
            area.append(b_area)
            boxes.append(box_coords)

        boxes = np.vstack(boxes).astype(np.int32)
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        img_id = torch.tensor([img_id])

        target={}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target




    def __len__(self):
        return len(self.image_ids)

class CBIS_DDSM(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

class inBreast(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def get_transform(train:bool=True):
    T = []
    T.append(transforms.ToTensor())
    if train:
        T.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(T)

