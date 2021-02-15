# Write training script to train on any amount of data
import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
from PIL import Image
from detection.faster_rcnn import FasterRCNN
from detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from dataset import AIIMS, get_transform, valAiimsAnn_path, valAiimsImg_path
from engine import train_one_epoch, evaluate
import utils
import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import gc
from torch.backends import cudnn
import random
from random import sample
cudnn.benchmark=True

parser = argparse.ArgumentParser(description="For training on Mammogram data for tumor detection")
parser.add_argument("--gpu", action='store_true')
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="./TumorAL_results")
parser.add_argument("--name", type=str, default="full_data")
parser.add_argument("--train_size", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=23)
# parser.add_argument("--save_freq", type=int, default=10)
parser.add_argument("--print_freq", type=int, default=20)
# parser.add_argument("--img_path", type=str, default="")

# Seed used till now 23=> full data 45=> initial_ALrun1
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.gpu or torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

bs = args.bs
num_epochs= args.epochs

save_dir = os.path.join(args.save_dir, args.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Doing All Data related stuff here
print("Initializing Data...")
trainDataSet = AIIMS(transform=get_transform(True))
print("Size of Total Dataset: %d" %len(trainDataSet))

num_classes = len(trainDataSet.classes)

valDataSet = AIIMS(img_path=valAiimsImg_path, ann_path=valAiimsAnn_path, transform=get_transform(False))
print("Length of Val Dataset: %d" % len(valDataSet))

trainLength = len(trainDataSet)
sample_size = int(args.train_size * trainLength)
train_indices = sample(list(range(trainLength)), sample_size)
print("Length of labelled Pool: %d" % len(train_indices))


trainImageIds=[]
trainSubSet = Subset(trainDataSet,train_indices)
for i in range(len(trainSubSet)):
    _, target = trainSubSet.__getitem__(i)
    trainImageIds.append(target["image_id"].item())

with open(os.path.join(save_dir,"trainImageIds.txt"), "w") as f:
    for item in trainImageIds:
        f.write("%d\n" % item)

trainLoader = DataLoader(trainSubSet, batch_size=bs, shuffle=True, num_workers=6, collate_fn=utils.collate_fn)
valLoader = DataLoader(valDataSet, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

# Now initialize Model, optimizer
print("Initializing Model...")
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)
model.train()

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params, lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.5)

print("Training for %d epochs" % num_epochs)
global_val_loss = float("inf")

for epoch in tqdm(range(num_epochs)):
    train_one_epoch(model, optimizer, trainLoader, device, epoch+1, print_freq=args.print_freq)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lr_scheduler.step()
    # Write validation part here if val loss is lower then only save model weights
    print("Validating the current model..")
    with torch.no_grad():
        val_epoch_loss=0.0
        for images, targets in valLoader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            val_epoch_loss+=loss_value

            del images, targets, loss_dict

        print("Validation Loss for Epoch %d is %.3f" % (epoch+1, val_epoch_loss))

        if val_epoch_loss<global_val_loss:
            global_val_loss=val_epoch_loss
            model_path = os.path.join(save_dir, "model_weights.pt")
            torch.save({"epoch":epoch+1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict()},
                        model_path)


print("Model Training is complete!!!!!!")








