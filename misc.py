# Import libraries here.
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from scipy.stats import entropy
from collections import defaultdict
from random import sample
from sklearn.metrics import pairwise_distances
import utils
from engine import evaluate
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

preprocess = transforms.ToTensor()


def ir_numpy(array):
    return np.std(array, axis=-1)/np.mean(array, axis=-1)


def binary_cooccurence(cls_logits:np.ndarray):
    # Assuming that background class has been removed
    cooccur = np.zeros(cls_logits.shape[-1])
    # probs = softmax(cls_logits, axis=-1)
    # max_probs = np.max(probs, axis=-1)
    # argmax_labels = np.argmax(probs, axis=-1)
    labels = np.unique(np.argmax(cls_logits, axis=-1))
    # assert max_probs.shape == argmax_labels.shape
    
    # ind = max_probs > thresh
    # labels = np.unique(argmax_labels[ind])
    cooccur[labels] = 1.0
    return cooccur[1:]


def k_center_modified(points_:np.ndarray, imageIds:list, dm, k:int):
    """
    Args:
        points_: Unlabelled Pool
        dm: distance measure in our case imbalance ratio
        k: number of points to select
    returns:
        selected: Selected Pool of points
    """
    selected = []
    selected_img_ids=[]
    upool = list(zip(points_, imageIds))
    ind1 = np.random.choice(len(upool))  # Choose first point randomly

    p1 = upool[ind1]
    selected.append(p1[0])                        # Add the first point to selected pool
    selected_img_ids.append(p1[1])
    # points_ = np.delete(points_, ind1, 0)      # Delete the point from unlabelled pool
    del upool[ind1]
    k-=1
    while k!=0:
        ir_list=[]             ### Maintain an i.r list for unlablled points.
        for p2 in upool:             ## iterate in unlabeled pool
            temp_list = selected.copy()           
            temp_list.append(p2[0])
            ir_list.append(dm(np.sum(temp_list, axis=0)))   # imbalance ratio of unlabelled point when it is added to the selected pool. 
#         print(k, ir_list)
        
        min_ind = np.argmin(np.array(ir_list))  # Choose the point which has the minimum imbalance ratio.
        p3 = upool[min_ind]                   
        selected.append(p3[0])                     # Add that point to the selected pool
        selected_img_ids.append(p3[1])
        # points_ = np.delete(points_, min_ind, 0)
        del upool[min_ind]
        k-=1
    return selected, selected_img_ids

def maxEntropy(cls_logits:list, image_ids:list, budget:int):
    entropy_list=[]
    for arr in cls_logits:
        probs = softmax(arr, axis=-1)
        ent = entropy(probs, axis=-1)
        entropy_list.append(ent.sum())
    
    ind = np.argsort(entropy_list)
    reverse_ind = ind[::-1].tolist()
#     print(reverse_ind)

    sorted_rev_ids = [image_ids[idx] for idx in reverse_ind]
    return sorted_rev_ids[:budget]


class kCenterGreedy():
    def __init__(self, X, metric='euclidean', already_selected=[]):
        self.features =X
        self.metric = metric
        self.min_distances = None
        self.n_obs = X.shape[0]
        self.already_selected = already_selected
    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
        if cluster_centers:
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features,x,metric='euclidean')
#             print(dist)
#             print(dist.shape)
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
            

    def select_batch(self, N):
        try:
            print('Calculating distances...')
            self.update_distances(self.already_selected, only_new=False, reset_dist=True)
        except:
            print('Using flat_X as features.')
            self.update_distances(self.already_selected, only_new=True, reset_dist=False)
        new_batch = []
        
        for idx in range(N):
            if not self.already_selected:
#                 print("random selection being done")
                ind = np.random.choice(np.arange(self.n_obs))
            else:
#                 print("No random selection being done")
                ind = np.argmax(self.min_distances)
            assert ind not in self.already_selected
            self.update_distances([ind], only_new=True, reset_dist=False)
            self.already_selected.append(ind)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'% max(self.min_distances))
#         self.already_selected = already_selected
        return new_batch

def coreset(points_:list, image_ids:list, budget:int, already_selected:list=[]):
    """
    points : image feature, if 100 images, 100x1000x12
    image_id : image Ids
    distance_fn: euclidean
    budget
    return : list of selected ids. 
    """

    points_ = np.stack(points_)
    points_ = softmax(points_,axis=-1) # converting logits to softmax
    a,b,c = points_.shape
        
#     already_selected = already_selected
    points_ = np.reshape(points_,(a,b*c))
    kc = kCenterGreedy(points_, already_selected=already_selected) # returns centers for the clusters
    select = np.sort(kc.select_batch(budget)) # select contains the index of selected features
    
    selection = []   
    for s in select:
        selection.append(image_ids[s]) # image ids for selected features
    return selection



def irSet_Helper(model, unlabelLoader, budget, all_classes, save_dir, device):
    unlabel_ids, cls_logits, _ = evaluate(model, unlabelLoader, device, save_dir, all_classes, False)
    cls_logits = [item.cpu().numpy() for item in cls_logits]
    np_logits=[]
    for arr in cls_logits:
        ind = np.argmax(arr, axis=-1)!=0
        new_arr= arr[ind]
        arr = new_arr[:,1:]
        np_logits.append(arr)
    
    weighted_list = [binary_cooccurence(item) for item in np_logits]
    weighted_arr = np.stack(weighted_list)

    if device=="cuda":
        torch.cuda.empty_cache()

    _, selected_ids = k_center_modified(weighted_arr, unlabel_ids, ir_numpy, budget)
    return selected_ids

# IRset based on model detetctions instead of logits

def irSet_Helper2(model, unlabelLoader, budget, all_classes, save_dir, device, thresh=0.20):
    unlabel_ids, _, all_detections = evaluate(model, unlabelLoader, device, save_dir, all_classes, False)
    
    labels = [item["labels"].numpy() for item in all_detections]
    scores = [item["scores"].numpy() for item in all_detections]

    final_labels=[]
    for label_arr, score_arr in zip(labels, scores):
        arr = np.zeros(len(all_classes),)
        # print("Label arr initial shape:", label_arr.shape)
        score_ind = score_arr>thresh
        new_label = label_arr[score_ind]
        # print("Label arr final shape:", label_arr.shape)

        unique_labels = np.unique(new_label)
        arr[unique_labels]=1.0
        final_labels.append(arr[2:])
    
    # weighted_list = [binary_cooccurence(item) for item in np_logits]
    weighted_arr = np.stack(final_labels)
    # print(weighted_arr.shape)

    if device=="cuda":
        torch.cuda.empty_cache()

    _, selected_ids = k_center_modified(weighted_arr, unlabel_ids, ir_numpy, budget)
    return selected_ids



def coreSet_Helper(model, fullLoader, budget, all_classes, save_dir, device, trainImageIds):
    full_ids, cls_logits, _ = evaluate(model, fullLoader, device, save_dir, all_classes, False)
    cls_logits = [item.cpu().numpy() for item in cls_logits]
    ### This has been added as for some images the final proposed regions is not 1000 as usually is with resnet Backbone
    processed_logits=[]
    processed_ids=[]
    for id_,arr in zip(full_ids, cls_logits):
        if arr.shape==(1000,len(all_classes)):
            processed_logits.append(arr)
            processed_ids.append(id_)

    ###########################################

    already_selected=[]
    for id_ in trainImageIds:
        try:
            already_selected.append(processed_ids.index(id_))
        except:
            continue

    if device=="cuda":
        torch.cuda.empty_cache()
    
    selected_ids = coreset(processed_logits, processed_ids, budget, already_selected)
    return selected_ids


def maxEntropy_Helper(model, unlabelLoader, budget, all_classes, save_dir, device):
    unlabel_ids, cls_logits, _ = evaluate(model, unlabelLoader, device, save_dir, all_classes, False)
    cls_logits = [item.cpu().numpy() for item in cls_logits]
    if device=="cuda":
        torch.cuda.empty_cache()
    
    selected_ids = maxEntropy(cls_logits, unlabel_ids, budget)
    return selected_ids


def MCD(multi_logits_list:list, image_ids:list, budget:int):
    """
    multi_logits_list: each item in this list has the dimension (T*regions*classes). In MCD
                       each input is forward passed to the model T times so we get T logits.
    """
    information_gain=[]
    for arr in multi_logits_list:
        probs = softmax(arr, axis=-1)
        average_ = np.average(probs, axis=-1)
        avg_entropy = entropy(average_, axis=-1).sum()

        every_entropy = entropy(probs, axis=-1)
        ent_average = np.average(every_entropy, axis=0).sum()

        info_gain = avg_entropy - ent_average
        information_gain.append(info_gain)

    ind = np.argsort(information_gain)
    reverse_ind = ind[::-1].tolist()
    # print(reverse_ind)

    sorted_rev_ids = [image_ids[idx] for idx in reverse_ind]
    return sorted_rev_ids[:budget]



def MCD_eval(m):
    if type(m)== nn.Dropout:
        m.train()
    else:
        m.eval() 


def MCD_Helper(model, unlabelLoader, budget, device, T=5):
    model.apply(MCD_eval)

    model.to(device)
    unlabel_ids=[]
    multi_logits_list=[]
    print("Evaluating images...")
    for images, targets in tqdm(unlabelLoader):
        images = list(img.to(device) for img in images)
        unlabel_ids.append(targets[0]["image_id"].item())
        temp_logits=[]
        for i in range(T):
            _, logit = model(images)
            # print(logit.shape)
            temp_logits.append(logit.detach().cpu().numpy())
            del logit
        
        del images

        final_logit = np.stack(temp_logits)
        multi_logits_list.append(final_logit)

    selection = MCD(multi_logits_list, unlabel_ids, budget)

    return selection


def hide_n_seek(original_image_list:list, multiple_runs_list:list, image_ids:list, budget:int, k:int):
    # Assuming that the preds are not empty for these images specified.
    uncertainity_list=[]
    for original_det, multiple_dets in zip(original_image_list, multiple_runs_list):
        agg_uncertainity=0.0
        for i in range(k):
            orig_score = original_det[i]
            occlude_scores = multiple_dets[i]
            if occlude_scores:
                agg_uncertainity+=(max(occlude_scores)-orig_score)
            else:
                agg_uncertainity+=(orig_score*(-1.0))
        avg_uncertainity = float(agg_uncertainity)/k
        uncertainity_list.append(avg_uncertainity)

    ind = np.argsort(uncertainity_list)
    reverse_ind = ind[::-1].tolist()

    sorted_rev_ids = [image_ids[idx] for idx in reverse_ind]
    return sorted_rev_ids[:budget]




def hide_n_seek_helper(model, unlabelLoader, imgPath_list, budget, device, k):
    model.eval()
    det_scores_list=[]
    multiple_runs_list=[]
    selected_ids_list=[]
    unlabeled_ids=[]

    print("Evaluating Images...")
    for i, (images, target) in enumerate(tqdm(unlabelLoader)):
        images = list(img.to(device) for img in images)
        detections, _ = model(images)
        if not detections:
            id_ = target[0]["image_id"].item()
            selected_ids_list.append(id_)
        else:
            original_scores = detections[0]["scores"].detach().cpu().numpy()
            original_boxes = detections[0]["boxes"].detach().cpu().numpy()

            det_scores_list.append(original_scores.tolist())
            unlabeled_ids.append(target[0]["image_id"].item())
            occlude_list=[]

            for j in range(k):
                img_ = Image.open(imgPath_list[i])
                img1 = ImageDraw.Draw(img_)
                img1.rectangle(original_boxes[j].tolist(), fill="#000000")
                img_pro = preprocess(img_).to(device)
                occlude_dets, _ = model([img_pro])
                occlude_list.append(occlude_dets[0]["scores"].detach().cpu().numpy().tolist())
                del occlude_dets, img_pro

            multiple_runs_list.append(occlude_list)
        del images

    # print("Length of images where there was no preds: %d" % len(selected_ids_list))
    new_budget = budget - len(selected_ids_list)
    new_selection = hide_n_seek(det_scores_list, multiple_runs_list, unlabeled_ids, new_budget, k)
    final_selection = selected_ids_list+new_selection

    return final_selection


# class kCenterGreedy():
#     def __init__(self, X, metric='euclidean'):
#         self.features =X
#         self.metric = metric
#         self.min_distances = None
#         self.n_obs = X.shape[0]
#         self.already_selected = []
#     def update_distances(self, cluster_centers, only_new=True, reset_dist=False):

#         if reset_dist:
#             self.min_distances = None
#         if only_new:
#             cluster_centers = [d for d in cluster_centers
#                          if d not in self.already_selected]
#         if cluster_centers:
#             x = self.features[cluster_centers]
#             dist = pairwise_distances(self.features,x,metric='euclidean')
#             if self.min_distances is None:
#                 self.min_distances = np.min(dist, axis=1).reshape(-1,1)
#             else:
#                 self.min_distances = np.minimum(self.min_distances, dist)

#     def select_batch(self, already_selected, N):
#         try:
#             print('Calculating distances...')
#             self.update_distances(already_selected, only_new=False, reset_dist=True)
#         except:
#             print('Using flat_X as features.')
#             self.update_distances(already_selected, only_new=True, reset_dist=False)
#         new_batch = []
        
#         for idx in range(N):
#             if self.already_selected is None or idx == 0:
#                 ind = np.random.choice(np.arange(self.n_obs))
#             else:
#                 ind = np.argmax(self.min_distances)
#             assert ind not in already_selected
#             self.update_distances([ind], only_new=True, reset_dist=False)

#             new_batch.append(ind)
#         print('Maximum distance from cluster centers is %0.2f'% max(self.min_distances))
#         self.already_selected = already_selected
#         return new_batch



# def coreset(points_:list, image_ids:list, budget:int):
#     """
#     points : image feature, if 100 images, 100x1000x12
#     image_id : image Ids
#     distance_fn: euclidean
#     budget
#     return : list of selected ids. 
#     """

#     points_ = np.stack(points_)
#     points_ = softmax(points_,axis=-1) # converting logits to softmax
#     a,b,c = points_.shape
        
#     already_selected = []
#     points_ = np.reshape(points_,(a,b*c))
#     kc = kCenterGreedy(points_) # returns centers for the clusters
#     select = np.sort(kc.select_batch(already_selected,budget)) # select contains the index of selected features
    
#     selection = []   
#     for s in select:
#         selection.append(image_ids[s]) # image ids for selected features
#     return selection
