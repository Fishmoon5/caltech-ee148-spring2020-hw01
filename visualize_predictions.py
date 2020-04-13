import os
import numpy as np
import json
from PIL import Image


# set the path to the downloaded data:
data_path = './data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = './data/hw01_preds'

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'r') as f:
    data = json.load(f)
    good_imgs = ['018','027','037','114']
    bad_imgs = ['006','012','042','185']
    for img in good_imgs:
        img_name = 'RL-' + img + '.jpg'
        I = Image.open(os.path.join(data_path,img_name))
        I = np.array(I)
        bounding_boxes = data[img_name]
        for b in bounding_boxes:
            [tl_row,tl_col,br_row,br_col] = b
            I[tl_row:br_row, tl_col, :] = [0,255,0]
            I[tl_row:br_row, br_col-1, :] = [0,255,0]
            I[tl_row, tl_col:br_col, :] = [0,255,0]
            I[br_row-1, tl_col:br_col, :] = [0,255,0]
        img = Image.fromarray(I, 'RGB')
        img.save(os.path.join(preds_path,img_name))
    for img in bad_imgs:
        img_name = 'RL-' + img + '.jpg'
        I = Image.open(os.path.join(data_path,img_name))
        I = np.array(I)
        bounding_boxes = data[img_name]
        for b in bounding_boxes:
            [tl_row,tl_col,br_row,br_col] = b
            I[tl_row:br_row, tl_col, :] = [0,255,0]
            I[tl_row:br_row, br_col-1, :] = [0,255,0]
            I[tl_row, tl_col:br_col, :] = [0,255,0]
            I[br_row-1, tl_col:br_col, :] = [0,255,0]
        img = Image.fromarray(I, 'RGB')
        img.save(os.path.join(preds_path,img_name))
