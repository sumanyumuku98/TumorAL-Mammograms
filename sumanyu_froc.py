"""
This script is used to calculate FROC and ROC curve.
    1) FROC is sensitivity Vs Average FP per image and is only calculated on Positive images
    to quantify tumor localization ability of model.
    2) ROC is calculated on both positive and negative images to quantify the ability of model
    to differentiate between tumor positive and tumor negative images.
    3) AUCROC is finally saved for classification.
    4) Average sensitivity is also reported on 6 predefined false positive rates
"""

import os
import numpy as np 
import argparse
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import roc_curve, roc_auc_score

parser = argparse.ArgumentParser(description="Path of detections and ground truths")
parser.add_argument("--posDet", type=str, required=True, help="Detection results on tumor positive images")
parser.add_argument("--negDet", type=str, required=True, help="Detection results on tumor negative images")
parser.add_argument("--GTpath", type=str, required=True, help="Ground Truth results on tumor positive images")
parser.add_argument("--save_dir", type=str, default="./fROC_plots")

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

def closest_index(lst, K):
    lst_ = np.asarray(lst) 
    idx = (np.abs(lst_ - K)).argmin() 
    return idx

def find_center(bbox):
    cx = 1.0*(bbox[0] + bbox[2]) / 2
    cy = 1.0*(bbox[1] + bbox[3]) / 2
    return cx,cy

def check_presence_box(cx,cy, gt_bbox):
    presence=[]
    for gt in gt_bbox:
        x1, y1, x2, y2 = gt
        if (cx >= x1 and cx <= x2) and (cy >= y1 and cy<= y2):
            presence.append(True)
        else:
            presence.append(False)
    return presence


# Lets first calculate ROC
scores=[]
labels=[]

GT_files = glob(os.path.join(args.GTpath, "*.txt"))
GT_files.sort()
pos_files = glob(os.path.join(args.posDet, "*.txt"))
pos_files.sort()

assert len(GT_files)==len(pos_files)

for det_file, label_file in zip(pos_files, GT_files):
    pos_name = det_file.split("/")[-1]
    label_name = label_file.split("/")[-1]

    assert pos_name==label_name
    labels.append(1)
    temp_score_list=[]
    with open(det_file, "r") as f:
        dets = f.readlines()
    
    if len(dets)==0:
        scores.append(0.0)
        continue

    for ind in range(len(dets)):
        words = dets[ind].split()
        temp_score_list.append(float(words[1]))

    scores.append(max(temp_score_list))

neg_files = glob(os.path.join(args.negDet, "*.txt"))

for neg_file in neg_files:
    labels.append(0)
    temp_score_list=[]
    with open(neg_file, "r") as f:
        negs = f.readlines()
    
    if len(negs)==0:
        scores.append(0.0)
        continue

    for ind in range(len(negs)):
        words = negs[ind].split()
        temp_score_list.append(float(words[1]))

    scores.append(max(temp_score_list))

fpr, tpr, thresholds = roc_curve(labels, scores)
auc = roc_auc_score(labels, scores)

print("AUC of the Model: %.3f" % auc)

plt.plot(fpr, tpr, "go-")
plt.title("ROC Curve (AUC: %.3f)" % auc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.legend("AUC Score: %.3f" % auc)
plt.savefig(os.path.join(args.save_dir, "roc_curve.png"), bbox_inches="tight", pad_inches=0)

# Lets plot FROC now

number_of_images=len(pos_files)
number_of_GT_boxes=0

det_boxes=[]
GT_boxes=[]

for det_file, label_file in zip(pos_files, GT_files):
    pos_name = det_file.split("/")[-1]
    label_name = label_file.split("/")[-1]

    assert pos_name==label_name

    with open(det_file, "r") as f:
        dets = f.readlines()
    
    pm = np.zeros([len(dets),5])
    for ind in range(len(dets)):
        words = dets[ind].split()
        xmin = int(float(words[2]))
        ymin = int(float(words[3]))
        xmax = int(float(words[4]))
        ymax = int(float(words[5]))
        pm[ind, :] = [xmin, ymin, xmax, ymax, float(words[1])]

    det_boxes.append(pm)


    with open(label_file, "r") as f:
        GTs = f.readlines()
    
    ground = np.zeros([len(GTs),4])
    for ind in range(len(GTs)):
        words = GTs[ind].split()
        xmin = int(float(words[1]))
        ymin = int(float(words[2]))
        xmax = int(float(words[3]))
        ymax = int(float(words[4]))
        ground[ind,:] = [xmin, ymin, xmax, ymax]

    number_of_GT_boxes+=ground.shape[0]
    GT_boxes.append(ground)


sensitivity_list=[]
fp_per_image=[]
sensitivity_levels=[0.25*pow(2,i) for i in range(6)]

threshold = list(np.arange(0.0, 1.001, 0.01))

for th in threshold:
    tp_count=0
    fp_count=0

    for pred_bbox, gt_bbox in zip(det_boxes, GT_boxes):
        # Assuming length of pred box is not equal to 0
        if len(pred_bbox)==0:
            continue
        GT_found_tp = np.zeros(gt_bbox.shape[0])
        pred_status=[]
        for bbox in pred_bbox:
            if bbox[-1]>th:
                cx, cy = find_center(bbox)
                presence = check_presence_box(cx, cy, gt_bbox)
                pred_status.append(np.array(presence, dtype=np.int8))
        
        if len(pred_status)==0:
            continue
        pred_status = np.stack(pred_status)
        for arr in pred_status:
            if not np.any(arr):
                fp_count+=1
            else:
                indx = np.where(arr==1)[0]
                for i in indx:
                    if GT_found_tp[i]==1:
                        continue
                    else:
                        GT_found_tp[i]=1
                        tp_count+=1

        
    s = (tp_count*1.0)/number_of_GT_boxes
    avg_fp = (fp_count*1.0)/number_of_images
    sensitivity_list.append(s)
    fp_per_image.append(avg_fp)


average_sensitivity=[]
for l in sensitivity_levels:
    idx = closest_index(fp_per_image, l)
    average_sensitivity.append(sensitivity_list[idx])

avg_Sensitivity = np.average(average_sensitivity)

print("Average Sensitivity: %.3f" % avg_Sensitivity)

plt.plot(fp_per_image, sensitivity_list, "ro-")
plt.title("FROC Curve (Avg S: %.3f)" % avg_Sensitivity)
plt.xlabel("Average False Positives Per Image")
plt.ylabel("Sensitivity")
plt.savefig(os.path.join(args.save_dir, "froc_curve.png"), bbox_inches="tight", pad_inches=0)

