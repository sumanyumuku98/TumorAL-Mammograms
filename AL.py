import os
import sys
import torch
import numpy as np
import torchvision
import torch.nn as nn
from PIL import Image
from detection.faster_rcnn import FasterRCNN
from detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from dataset import AIIMS, get_transform, valAiimsAnn_path, valAiimsImg_path
from misc import *
from engine import train_one_epoch, evaluate
import utils
from torch.utils.data import DataLoader, Subset
from torch.backends import cudnn
from tqdm import tqdm
import argparse
import random
from random import sample
import json
cudnn.benchmark=True

parser = argparse.ArgumentParser(description="Arguments for Tumor dataset")
parser.add_argument("--gpu", action='store_true')
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="./Tumor_iterations")
parser.add_argument("--name", type=str, default="random")
parser.add_argument("--budget", type=float, default=0.1)
parser.add_argument("--AL", type=str, choices=["CS", "occlusion", "random", "MCD", "entropy"], default="random")
parser.add_argument("--initial_ckpt", type=str, required=True)
parser.add_argument("--initial_ids", type=str, required=True)
parser.add_argument("--cycles", type=int, default=5)
# parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--print_freq", type=int, default=20)
parser.add_argument("--k", type=int, default=5)

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

save_dir = os.path.join(args.save_dir, args.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

status_file = os.path.join(save_dir, "last_status.json")

if not os.path.exists(status_file):
    print("No init config found")
    print("Initializing config...")
    status_dict={}
    status_dict["AL_iter"]=0
    status_dict["last_checkpoint"] = args.initial_ckpt
    status_dict["last_trainIds"] = args.initial_ids
    with open(args.initial_ids, "r") as f:
        initIds = f.readlines()

    initIds=[int(x) for x in initIds]

    complete_data = AIIMS(transform=get_transform(True))

    full_length = len(complete_data)
    sample_budget = int(args.budget * full_length)
    stop_length = len(initIds) + args.cycles*sample_budget
    status_dict["budget"] = sample_budget
    status_dict["stop_len"] = stop_length
    status_dict["AL"] = args.AL
    status_dict["bs"] = args.bs
    status_dict["epochs"] = args.epochs
    status_dict["num_classes"] = len(complete_data.classes)

    with open(status_file, "w") as f:
        json.dump(status_dict, f, indent=2)


with open(status_file, "r") as f:
    init_status = json.load(f)

print("<=======Current Config=======>")
print(init_status)
print("<============================>")
checkpoint_file = init_status["last_checkpoint"]
trainId_file = init_status["last_trainIds"]

with open(trainId_file, "r") as f:
    trainImageIds = f.readlines()
trainImageIds = [int(x) for x in trainImageIds]

sample_budget = int(init_status["budget"])
stop_length = int(init_status["stop_len"])
al_iteration = int(init_status["AL_iter"])
algo_use = init_status["AL"]
bs = int(init_status["bs"])
num_classes = int(init_status["num_classes"])
num_epochs = int(init_status["epochs"])


print("Length of initial pool: %d" % len(trainImageIds))
print("Sampling Budget: %d" % sample_budget)
print("Training will stop when length of labelled pool is: %d" % stop_length)


### Define function to train model for iteration
"""
Steps:
    1) First sample data from unlabelled pool using specified_algo
    2) Train model on selected pool
"""

def AL_iteration(checkpoint_file, train_ids_file, budget, stop_length, al_iteration, device="cpu"):

    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    ckpt=checkpoint_file
    print("Reading from checkpoint: %s" % ckpt)

    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)

    with open(train_ids_file, "r") as f:
        trainImageIds= f.readlines()

    trainImageIds = [int(x) for x in trainImageIds]
    unlabelData = AIIMS(transform=get_transform(False), trainIds=trainImageIds)

    print("Length of unlabelled pool: %d" % len(unlabelData))

    all_classes = unlabelData.classes
    unlabelLoader = DataLoader(unlabelData, batch_size= 1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    if len(trainImageIds)==stop_length:
        print("Cycles specified in Config has been completed!!")
        sys.exit(0)
    else:

        if algo_use=="random":
            selected_ids= sample(unlabelData.image_ids, budget)
        elif algo_use=="occlusion":
            ## get Ids using Hide N Seek
            image_paths = unlabelData.getImagePaths()
            selected_ids = occlusion_dppHelper(model, unlabelLoader, image_paths, budget, device, args.k)
            # selected_ids = hide_n_seek_helper(model, unlabelLoader, image_paths, budget, device, k=args.k)
        elif algo_use=="CS":
            ## get Ids using CS
            fullData = AIIMS(transform=get_transform(False))
            fullLoader = DataLoader(fullData, batch_size= 1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
            selected_ids = coreset_dppHelper(model, fullLoader, budget, device, trainImageIds, args.k)

            # selected_ids = coreSet_Helper(model, fullLoader, budget, all_classes, save_dir, device, trainImageIds)
        elif algo_use=="entropy":
            ## Get ids using max entropy
            selected_ids = maxEntropy_Helper(model, unlabelLoader, budget, all_classes, save_dir, device)
        elif algo_use=="MCD":
            ## GEt ids using MCD
            selected_ids = MCD_Helper(model, unlabelLoader, budget, device)

        del model, state_dict

        print("Selected Pool of Length: %d" % len(selected_ids))

        trainImageIds+=selected_ids

        selection_file = os.path.join(save_dir, "%s_ALselection_iter_%d.txt" % (algo_use, al_iteration))

        # with open(selection_file, "w") as f:
        #     for item in selected_ids:
        #         f.write("%d\n" % item)

        train_file = os.path.join(save_dir, "trainImageIds_%d.txt" % al_iteration)

        # with open(train_file, "w") as f:
        #     for item in trainImageIds:
        #         f.write("%d\n" % item)

        init_status["last_trainIds"]=train_file

        trainData=AIIMS(transform=get_transform(True), trainIds=selected_ids, mode="train")
        valData=AIIMS(img_path=valAiimsImg_path, ann_path=valAiimsAnn_path, transform=get_transform(False))

        print("Iteratively finetuning on data of length: %d" % len(trainData))
        print("Will also validate on data of length %d" % len(valData))
        trainLoader = DataLoader(trainData, batch_size=bs, shuffle=True, num_workers=6, collate_fn=utils.collate_fn)
        valLoader = DataLoader(valData, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)


        model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        print("Loading weights from %s for training" % ckpt)

        state_dict = torch.load(ckpt)
        model.load_state_dict(state_dict["model_state_dict"])
        model.to(device)

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
                    model_path = os.path.join(save_dir, "{}_ALmodel_iter_{}.pt".format(algo_use, al_iteration))
                    torch.save({"epoch":epoch+1,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict":optimizer.state_dict()},
                                model_path)
                    print("Model saved at %s" % model_path)
                    init_status["last_checkpoint"]=model_path
            
    print("Training in AL cycle: %d is completed" % al_iteration)

    with open(selection_file, "w") as f:
        for item in selected_ids:
            f.write("%d\n" % item)

    
    with open(train_file, "w") as f:
        for item in trainImageIds:
            f.write("%d\n" % item)

    # Save updated dict
    print("Updating status dict...")
    with open(status_file, "w") as f:
        json.dump(init_status, f, indent=2)
    

if __name__=="__main__":
    init_status["AL_iter"]=al_iteration+1
    AL_iteration(checkpoint_file, trainId_file, sample_budget, stop_length, al_iteration+1, device)