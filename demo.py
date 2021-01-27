import os
from PIL import Image
from vizer.draw import draw_boxes
import argparse
import pandas as pd 
import glob
import numpy as np
from tqdm import tqdm

# Hardcoded for now will be changed later
class_list=["__background__", "tumor"]


parser = argparse.ArgumentParser(description="Plotting bounding boxes over tumor region in mammograms")
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--ann_path", type=str, required=True)
# parser.add_argument("--ann_format", type=str, choices=["csv"])
parser.add_argument("--result_dir", type=str, default="./mammogram_TumorVis")

args = parser.parse_args()

save_dir = args.result_dir

# Make result dir if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Read Annotations from csv
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

print("Bounding boxes have been plotted on the images!!")



