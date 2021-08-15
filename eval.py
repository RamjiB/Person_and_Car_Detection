import numpy as np 
import pandas as pd 
import os
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.anchor_utils import AnchorGenerator
from PIL import Image
import matplotlib.patches as patches

from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(64)

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' -q

from pycocotools.coco import COCO

#reading the annoattion file
coco = COCO('data/trainval/annotations/bbox-annotations.json')

#generate target
def generate_target(img_ids):
    """Generate the target dictionary for each image_id.

    Args:
        img_ids: (int) image_id number of the whole dataset
        
    Returns:
        (str) filename : 'image001.png'
        (dict) target :{'boxes':[[xmin,ymin,xmax,ymax],.....],'labels':[0,...],'image_id','1'}
    """
    #get the annotations for each image_ids
    annotation_ids = coco.getAnnIds(img_ids)
    annotations = coco.loadAnns(annotation_ids)
    image_meta = coco.loadImgs(annotations[0]["image_id"])[0]
    boxes=[]
    labels=[]
    for ann in annotations:
        x,y,w,h = ann['bbox']
        if w!=0  and h!=0:
            bbox = [x,y,x+w,y+h]
            boxes.append(bbox)
            labels.append(ann['category_id'])
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    img_id = torch.tensor([img_ids])
    target = {}
    target["boxes"] = boxes    
    target["labels"] = labels
    target["image_id"] = img_id
    return image_meta["file_name"], target

    
# Dataset class for Validation - 10%of data
class Dataset_Val(object):
    def __init__(self, transforms):
        self.transforms = transforms
        # load all image files
        self.img_ids = range(int(0.9*len(os.listdir('data/trainval/images/'))), 
                             len(os.listdir('data/trainval/images/')))
    def __getitem__(self, idx):
        image_name, target = generate_target(idx)
        img_path = os.path.join("data/trainval/images/", image_name)
        img = Image.open(img_path).convert("RGB")
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.img_ids)

#preprocessing transformation      
data_transform = transforms.Compose([
        transforms.ToTensor(), ])
#batch processing
def collate_fn(batch):
    return tuple(zip(*batch))
# Test Dataset loader for pytorch model
test_dataset = Dataset_Val(data_transform)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

def Model():
    # load an model pre-trained pre-trained on COCO
    backbone = resnet_fpn_backbone('resnet50', True, returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256))
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((1.0, 2.0,5.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios)
#     anchor_generator = AnchorGenerator(sizes=((128, 256, 512),),aspect_ratios=((0.5,1.0, 2.0),))
    model = RetinaNet(backbone, num_classes=3,anchor_generator=anchor_generator)
    return model

model = Model()

#devic info (GPU/CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Load the trained model
model.load_state_dict(torch.load('models/retinanet_resnet50_fpn.pth'))

def calc_iou( gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p= pred_bbox
    
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct",x_topleft_p, x_bottomright_p,y_topleft_p,y_bottomright_gt)
        
         
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt< x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        
        return 0.0
    if(y_bottomright_gt< y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        
        return 0.0
    if(x_topleft_gt> x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        
        return 0.0
    if(y_topleft_gt> y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        
        return 0.0
    
    
    GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
    Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)
    
    x_top_left =np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area

def calc_precision_recall(image_results):
"""Calculates precision and recall from the set of images
Args:
    img_results (dict): dictionary formatted like:
        {
            'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
            'img_id2': ...
            ...
        }
Returns:
    tuple: of floats of (precision, recall)
"""
    true_positive=0
    false_positive=0
    false_negative=0
    for img_id, res in image_results.items():
        true_positive +=res['true_positive']
        false_positive += res['false_positive']
        false_negative += res['false_negative']
        try:
            precision = true_positive/(true_positive+ false_positive)
        except ZeroDivisionError:
            precision=0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall=0.0
    return (precision, recall)


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    if len(all_gt_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou= calc_iou(gt_box, pred_box)
            
            if iou >iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp= len(gt_match_idx)
        fp= len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


def get_model_scores(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_score={}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_score.keys():
                model_score[score]=[img_id]
            else:
                model_score[score].append(img_id)
    return model_score



def  get_avg_precision_at_iou(gt_boxes, pred_bb, iou_thr=0.5):
    
    model_scores = get_model_scores(pred_bb)
    sorted_model_scores= sorted(model_scores.keys())
# Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_bb.keys():
        
        arg_sort = np.argsort(pred_bb[img_id]['scores'])
        pred_bb[img_id]['scores'] = np.array(pred_bb[img_id]['scores'])[arg_sort].tolist()
        pred_bb[img_id]['boxes'] = np.array(pred_bb[img_id]['boxes'])[arg_sort].tolist()
    pred_boxes_pruned = deepcopy(pred_bb)
    
    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
            # On first iteration, define img_results for the first time:
  
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores[model_score_thr]
        for img_id in img_ids:
               
            gt_boxes_img = gt_boxes[img_id]
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break 
            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores']= pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes']= pred_boxes_pruned[img_id]['boxes'][start_idx:]
# Recalculate image results for this image
     
            img_results[img_id] = get_single_image_results(gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr=0.5)
# calculate precision and recall
        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args= np.argwhere(recalls>recall_level).flatten()
            prec= max(precisions[args])
            # print(recalls,"Recall")
            # print(      recall_level,"Recall Level")
            # print(       args, "Args")
            # print(       prec, "precision")
        except ValueError:
            prec=0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec) 
    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}

def generate_eval(class_):
    gt_boxes={}
    pr_boxes={}
    for imgs, annotations in test_data_loader:
        imgs = list(img.to(device) for img in imgs)
        model.eval()
        preds = model(imgs)
        
        for i, annotation in enumerate(annotations):
            bbox=[]
            p_bbox = []
            scores=[]
            for box, label in zip(annotation['boxes'],annotation['labels']):
                if label==class_:
                    bbox.append(list(box.numpy()))
            for box, label,score in zip(preds[i]['boxes'],preds[i]['labels'],preds[i]['scores']):
                if label == class_:
                    p_bbox.append(list(box.detach().numpy()))
                    scores.append(float(score.detach().numpy()))
                
            pred_dict=   {'boxes':p_bbox, 'scores':scores}
            gt_boxes[annotation['image_id']] = bbox
            pr_boxes[annotation['image_id']] = pred_dict
    return gt_boxes, pr_boxes


#evaluate person class
gt_boxes_pr,pr_boxes_pr = generate_eval(1)


person_res = get_avg_precision_at_iou(gt_boxes_pr, pr_boxes_pr, iou_thr=0.5)

plt.plot(person_res['recalls'], person_res['precisions'])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('person_class_PR_curve.png')

print('Person class evaluation', person_res)

#evaluate car class
gt_boxes_car,pr_boxes_car = generate_eval(0)


car_res = get_avg_precision_at_iou(gt_boxes_car, pr_boxes_car, iou_thr=0.5)

print('Car class evaluation', car_res)

plt.plot(car_res['recalls'], car_res['precisions'])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('Car_class_PR_curve.png')









