import os
from PIL import Image
from Vizer.vizer.draw import draw_boxes, draw_boxes_fixedColor
import argparse
import pandas as pd 
import glob
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from detection.faster_rcnn import FasterRCNN
from detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from glob import glob


# Hardcoded for now will be changed later
class_list=["__background__", "tumor"]

preprocess=transforms.ToTensor()

parser = argparse.ArgumentParser(description="Plotting bounding boxes over tumor region in mammograms")
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--ann_path", type=str)
parser.add_argument("--ann_format", type=str, choices=["GT", "Pred", "Both"])
parser.add_argument("--ckpt", type=str)
parser.add_argument("--result_dir", type=str, default="./mammogram_TumorVis")
parser.add_argument("--thresh", type=float, default=0.9)


args = parser.parse_args()

save_dir = args.result_dir

# Make result dir if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device= "cuda" if torch.cuda.is_available() else "cpu"

# Read Annotations from csv

if args.ann_format=="GT":
    df = pd.read_csv(args.ann_path)

    unique_filenames = df["filename"].unique().tolist()

    for file_ in tqdm(unique_filenames):
        boxes = []
        class_names = []
        img_path = os.path.join(args.img_dir, file_)
        img = Image.open(img_path)

        img_df = df[df["filename"]==file_]

        for index, row in img_df.iterrows():
            class_ = row["class"]
            box_ = row["xmin":"ymax"].to_numpy()
            class_names.append(class_list.index(class_))
            boxes.append(box_)
        
        boxes = np.stack(boxes)
        img2 = draw_boxes(img, boxes=boxes, labels=class_names, class_name_map=class_list)
        save_path = os.path.join(save_dir, file_)
        img_pil = Image.fromarray(img2)
        img_pil.save(save_path)

elif args.ann_format=="Pred":
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(class_list))
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    images_list = glob(os.path.join(args.img_dir, "*.png")) # Change this based on image format
    for img_path in tqdm(images_list):
        img = Image.open(img_path)
        img_name = img_path.split("/")[-1]

        tensor_img = preprocess(img).to(device)
        detections, logits = model([tensor_img])
        # keep = nms(detections[0]["boxes"].detach().cpu(), detections[0]["scores"].detach().cpu(), 0.5)
        np_logits = logits.detach().cpu().numpy()

        fin_boxes = detections[0]["boxes"].detach().cpu().numpy()
        fin_scores = detections[0]["scores"].detach().cpu().numpy()
        fin_labels = detections[0]["labels"].cpu().numpy()

        keep = fin_scores > args.thresh
        new_boxes = fin_boxes[keep]
        new_scores=fin_scores[keep]
        new_labels= fin_labels[keep]

        new_img2 = draw_boxes(img, boxes=new_boxes, labels=new_labels, scores=new_scores, class_name_map=class_list)
        final_img = Image.fromarray(new_img2)
        final_img.save(os.path.join(save_dir, img_name))
else:
    # First plot detection then GT or vice-versa
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(class_list))
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    images_list = glob(os.path.join(args.img_dir, "*.png")) # Change this based on image format

    df = pd.read_csv(args.ann_path)

    unique_filenames = df["filename"].unique().tolist()

    for img_path in tqdm(images_list):
        img = Image.open(img_path)
        img_name = img_path.split("/")[-1]
        
        img2=None
        GT_boxes=[]
        GT_labels=[]
        if img_name in unique_filenames:
            # print("Doing GT")
            img_df = df[df["filename"]==img_name]

            for index, row in img_df.iterrows():
                class_ = row["class"]
                box_ = row["xmin":"ymax"].to_numpy()
                GT_boxes.append(box_)
                GT_labels.append(class_list.index(class_))
            
            GT_boxes= np.stack(GT_boxes)
            
        if len(GT_boxes)!=0 and len(GT_labels)!=0:
            img2 = draw_boxes_fixedColor(img, boxes=GT_boxes, labels=GT_labels, class_name_map=class_list)
            img2 = Image.fromarray(img2)


        tensor_img = preprocess(img).to(device)
        detections, logits = model([tensor_img])
        # keep = nms(detections[0]["boxes"].detach().cpu(), detections[0]["scores"].detach().cpu(), 0.5)
        np_logits = logits.detach().cpu().numpy()

        fin_boxes = detections[0]["boxes"].detach().cpu().numpy()
        fin_scores = detections[0]["scores"].detach().cpu().numpy()
        fin_labels = detections[0]["labels"].cpu().numpy()

        keep = fin_scores > args.thresh
        new_boxes = fin_boxes[keep]
        new_scores=fin_scores[keep]
        new_labels= fin_labels[keep]

        if img2 is None:
            new_img2 = draw_boxes_fixedColor(img, boxes=new_boxes, labels=new_labels, scores=new_scores, class_name_map=class_list, colour=(255,0,0))
        else:
            new_img2 = draw_boxes_fixedColor(img2, boxes=new_boxes, labels=new_labels, scores=new_scores, class_name_map=class_list, colour=(255,0,0))
        final_img = Image.fromarray(new_img2)
        final_img.save(os.path.join(save_dir, img_name))

print("Bounding boxes have been plotted on the images!!")



