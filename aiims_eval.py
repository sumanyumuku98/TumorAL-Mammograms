"""
This script only currently supports fROC calculation in 
AIIMS test data where both positive and negative
images are present.

Support for fROC on public data needs to be added.

"""
import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
from PIL import Image
from detection.faster_rcnn import FasterRCNN
from detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from engine import evaluate
from dataset import AIIMS, testAiimsImg_path, testAiimsAnn_path, get_transform
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import utils
import transforms
import argparse
import pandas as pd 
import glob
from torch.backends import cudnn
cudnn.benchmark=True

ignore_filenames=["RAD129592_20151208_5_FILE04257452.png"]
preprocess=torchvision.transforms.ToTensor()

parser = argparse.ArgumentParser(description="For generating fROC on AIIMS Test Data")
parser.add_argument("--img_dir", type=str)
# parser.add_argument("--ann_path", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--save_dir", type=str, required=True)

args = parser.parse_args()
save_dir = args.save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

all_test_files = glob.glob(os.path.join(args.img_dir, "*.png"))
all_test_names = [item.split("/")[-1] for item in all_test_files]

testDataSet = AIIMS(img_path=testAiimsImg_path, ann_path=testAiimsAnn_path, transform=get_transform(False))
class_list = testDataSet.classes
print("Length of Tumor Positive images: %d" % len(testDataSet))

pos_filenames = testDataSet.getPosFileNames()

neg_filenames= list(set(all_test_names)- set(pos_filenames))
neg_filenames = list(set(neg_filenames)- set(ignore_filenames))
print("Length of Tumor Negative images: %d" % len(neg_filenames))

# Initialize Model
print("Loading checkpoint...")
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(class_list))

state_dict = torch.load(args.checkpoint)
model.load_state_dict(state_dict["model_state_dict"])
model.to(device)
model.eval()
print("Checkpoint loaded")

testLoader= DataLoader(testDataSet, batch_size=1, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

names, logits, _ = evaluate(model, testLoader, device, save_dir, class_list, True)
 # .txt files for positive images have been saved

 # Now lets do it for negative images
neg_dir = os.path.join(save_dir, "neg_results")
if not os.path.exists(neg_dir):
    os.makedirs(neg_dir)

for negImgName in tqdm(neg_filenames):
    img_path = os.path.join(args.img_dir, negImgName)
    img = Image.open(img_path)
    img_name = negImgName[:-4]
    tensor_img = preprocess(img).to(device)
    detections, _ = model([tensor_img])

    fin_boxes = detections[0]["boxes"].detach().cpu()
    fin_scores = detections[0]["scores"].detach().cpu()
    fin_labels = detections[0]["labels"].cpu()

    neg_txt_file = os.path.join(neg_dir, img_name+".txt")

    with open(neg_txt_file, "w") as f:
        for label, score, box in zip(fin_labels, fin_scores, fin_boxes):
            cls_name = class_list[label.item()]
            score_ = score.item()
            bbox_ = box.tolist()
            
            str_ = "%s %.4f %d %d %d %d\n" % (cls_name, score_, bbox_[0], bbox_[1], bbox_[2], bbox_[3])
            f.write(str_)


# os.system("python compute_froc_with_neg_data.py --posDet ")








