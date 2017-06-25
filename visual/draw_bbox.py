import cv2
from ..utils.bbox_transform import bbox_transform

def draw_bbox(im, box_list, label_list, color=(0,255,0),cdict=None,form='center'):
    assert form=='center' or form=='diagonal',\
            'bounding box format not accepted: {}.'.format(form)

    for bbox,label in zip(box_list,label_list):
        if form == 'center':
            bbox = bbox_transform(bbox)

        xmin, ymin, xmax, ymax = [int(b) for b in bbox]

        l = label.split(':')[0]
        if cdict and l in cdict:
            c = cdict[l]

        else:
            c = color

        cv2.rectangle(im,(xmin,ymin),(xmax,ymax),c,1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, label,(xmin,ymin),font,0.3,c,1)


