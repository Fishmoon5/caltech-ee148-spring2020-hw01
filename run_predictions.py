import os
import numpy as np
import json
from PIL import Image

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''


    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.

    # Best Algorithm (Algorithm 2)
    k = Image.open('redlight.jpg')
    k = np.asarray(k)
    k = k / np.linalg.norm(k)
    box_height, box_width, _ = k.shape

    height, width, _ = I.shape
    for i in range(height - box_height + 1):
        for j in range(width - box_width + 1):
            tmp = I[i:i+box_height, j:j+box_width, :]
            tmp = tmp / np.linalg.norm(tmp)
            val = np.sum(tmp * k)
            #print(val)
            if val > 0.9:
                tl_row = i
                tl_col = j
                br_row = i+box_height
                br_col = j+box_width
                bounding_boxes.append([tl_row,tl_col,br_row,br_col])

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

    # Algorithm 1
    # k = np.zeros([6,8,3])
    # k[0:2,2:5,:] = [255,0,0]
    # k = k / np.linalg.norm(k)
    # print(k)
    # box_height, box_width, _ = k.shape
    #
    # height, width, _ = I.shape
    # for i in range(height - box_height + 1):
    #     for j in range(width - box_width + 1):
    #         tmp = I[i:i+box_height, j:j+box_width, :]
    #         tmp = tmp / np.linalg.norm(tmp)
    #         val = np.sum(tmp * k)
    #         print(val)
    #         if val > 0.3:
    #             tl_row = i
    #             tl_col = j
    #             br_row = i+box_height
    #             br_col = j+box_width
    #             bounding_boxes.append([tl_row,tl_col,br_row,br_col])

    # Algorithm 3
    # k = Image.open('redlight.jpg')
    # k = np.asarray(k)
    # k = k[:, :, 0]
    # k = k / np.linalg.norm(k)
    # print(k)
    # box_height, box_width = k.shape
    #
    # height, width, _ = I.shape
    # for i in range(height - box_height + 1):
    #     for j in range(width - box_width + 1):
    #         tmp = I[i:i+box_height, j:j+box_width, :]
    #         tmp = tmp[:, :, 0]
    #         tmp = tmp / np.linalg.norm(tmp)
    #         val = np.sum(tmp * k)
    #         print(val)
    #         if val > 0.9:
    #             tl_row = i
    #             tl_col = j
    #             br_row = i+box_height
    #             br_col = j+box_width
    #             bounding_boxes.append([tl_row,tl_col,br_row,br_col])

    # # Algorithm 4
    # k = Image.open('redlight.jpg')
    # k = np.asarray(k)
    # k = k[:, :, 0] * 0.5 + k[:, :, 1] * 0.1 + k[:, :, 2] * 0.4
    # k = k / np.linalg.norm(k)
    # box_height, box_width = k.shape
    #
    # height, width, _ = I.shape
    # I = I[:,:,0] * 0.5 + I[:,:,1]*0.1 + I[:,:,2] * 0.4
    # for i in range(height - box_height + 1):
    #     for j in range(width - box_width + 1):
    #         tmp = I[i:i+box_height, j:j+box_width]
    #         tmp = tmp / np.linalg.norm(tmp)
    #         val = np.sum(tmp * k)
    #         #print(val)
    #         if val > 0.9:
    #             tl_row = i
    #             tl_col = j
    #             br_row = i+box_height
    #             br_col = j+box_width
    #             bounding_boxes.append([tl_row,tl_col,br_row,br_col])


# set the path to the downloaded data:
data_path = './data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = './data/hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for i in range(len(file_names)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
