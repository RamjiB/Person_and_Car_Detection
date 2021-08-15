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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    
# Dataset class for Training - 90%of data
class Dataset_Train(object):
    
    def __init__(self, transforms):
        self.transforms = transforms
        # load all image files
        self.img_ids = range(1, int(0.9*(len(os.listdir('data/trainval/images/')))))
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

# Train Dataset loader for pytorch model
train_dataset = Dataset_Train(data_transform)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)

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

#Training code

num_epochs = 100
model.to(device)   

# parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

len_dataloader = len(train_data_loader)

for epoch in range(num_epochs):
    model.train()
    i = 0    
    epoch_loss = 0
    for imgs, annotations in train_data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model([imgs[0]], [annotations[0]])
        losses = sum(loss for loss in loss_dict.values())        

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        epoch_loss += losses
    torch.save(model.state_dict(),'models/retinanet_resnet50_fpn.pth')
    print('Epoch '+str(epoch)+'----> loss: ',epoch_loss)

