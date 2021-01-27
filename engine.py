import math
import sys
import time
import torch

# import torchvision.models.detection.mask_rcnn

# from coco_utils import get_coco_api_from_dataset
# from coco_eval import CocoEvaluator
import utils
import gc
import os
import numpy as np


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        del images, targets, loss_dict
        gc.collect()

    if device=="cuda":
        torch.cuda.empty_cache()

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, save_dir, class_names=[], save_in_txt=False):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    detection_path = os.path.join(save_dir, "detections")
    GT_path = os.path.join(save_dir, "groundTruths")

    # inspect_class= class_names[1]
    # print("Class under inspection is: ", inspect_class)


    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_logits=[]
    all_detections=[]
    Names=[]
    # print("Evaluating on image IDs of length: %d " % len(img_ids))
    # plotting_features=[]
    # coco = get_coco_api_from_dataset(data_loader.dataset)
    # iou_types = _get_iou_types(model)
    # coco_evaluator = CocoEvaluator(coco, iou_types, label_mapping, catIds, img_ids)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        # tempNames = [item.split("/")[-1] for item in paths]
        Names.append(targets[0]["image_id"].item())

        torch.cuda.synchronize()
        model_time = time.time()
        outputs, logits = model(images)
        # print(outputs)

        # min_regions=1e8
        # for item in logits:
        #     min_regions = min(min_regions, item.shape[0])
        # newLogits = []
        # for item in logitImages:
        #     rows = item.shape[0]
        #     indices = np.random.choice(rows, size=min_regions, replace=False)
        #     array = item[indices,:]
        #     newLogits.append(array)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        # print(logits.shape)
        all_logits.append(logits)
        # plotting_features.append(features)
        # print(logits[0].shape)

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
        gt = targets[0]
        image_id = gt["image_id"].item()
        detections = res[image_id]

        all_detections.append(detections)

        if save_in_txt:
            # Save Detections Results
            if not os.path.exists(detection_path):
                os.mkdir(detection_path)
    
            if not os.path.exists(GT_path):
                os.mkdir(GT_path)

            # Lets save detection results
            dt_boxes = detections["boxes"]
            dt_scores = detections["scores"]
            dt_labels = detections["labels"]

            dt_txt_file = os.path.join(detection_path,"%d.txt" % image_id)

            with open(dt_txt_file, "w") as f:
                for label, score, box in zip(dt_labels, dt_scores, dt_boxes):
                    cls_name = class_names[label.item()]

                    # Do not save results of inspect class
                    # if cls_name==inspect_class:
                    #     continue
                    score_ = score.item()
                    bbox_ = box.tolist()
                    str_ = "%s %.4f %d %d %d %d\n" % (cls_name, score_, bbox_[0], bbox_[1], bbox_[2], bbox_[3])
                    f.write(str_)

            # Lets save ground truth results
        
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            gt_txt_file = os.path.join(GT_path, "%d.txt" % image_id)

            with open(gt_txt_file, "w") as f:
                for label, box in zip(gt_labels, gt_boxes):
                    cls_name = class_names[label.item()]

                    # Do not save results of inspect class

                    # if cls_name==inspect_class:
                    #     continue
                    bbox_ = box.tolist()

                    str_ = "%s %d %d %d %d\n" % (cls_name, bbox_[0], bbox_[1], bbox_[2], bbox_[3])
                    f.write(str_)
        

        evaluator_time = time.time()
        # coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        del images

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # coco_evaluator.synchronize_between_processes()
    # all_logits = torch.stack(all_logits)
    # accumulate predictions from all images
    # coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return Names, all_logits, all_detections
