import cv2
import numpy as np
import math
def zoom(feat_map,zoom_factor=1):
    '''
    zoom the feature map
    inputs
    - feat_map: np.array with(c,h,w)
    returns
    - zoom_map: np.array with(c,h,w)*zoom_factor
    '''
    c,h,w = feat_map.shape
    out_w = (w-1)*zoom_factor + 1
    out_h = (h-1)*zoom_factor + 1
    trans_map = feat_map.transpose((1,2,0))
    
    mean = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
    #zoom_map = cv2.resize(trans_map,None,fx=zoom_factor,fy=zoom_factor)
    zoom_map = cv2.resize(trans_map,(out_w,out_h))
    return zoom_map.transpose((2,0,1))
def random_crop(current_image,key_image = None,seg_label,crop_height,crop_witdh,mean,train=True):
        '''
        apply the deeplab crop methods: fisrt padding the seg_image using mean values and padding 
        the seg_label using ignore_label if necerary.Then random crop when trainning and middle crop when val
        inputs
        - seg_image: from cv2, with order(B,G,R)
        - seg_label: with the same shape with seg_image.
        returns:
        seg_image: with shape(crop_size,crop_size,c)
        seg_label: with shape(crop_size,crop_size)
        '''
        assert len(current_image.shape) == 3,'the image should be BGR channels'
        assert len(seg_label.shape) == 2,'the image seg label should be 2 dim'
        data_height,data_width,data_c =  current_image.shape
        label_height,label_width = seg_label.shape
        assert data_height == label_height,'image and label should have the same height'
        assert data_width == label_width,'image and label should have the same width'
        
        pad_height = max(crop_height - data_height, 0)
        pad_width = max(crop_witdh - data_width, 0)

        # when crop is needed
        # pad 
        if pad_height > 0 or pad_width > 0:
            #cv2 copymakevorder need float64 as value parameter
            mean = np.array(mean,dtype=np.float64)
            current_image = cv2.copyMakeBorder(current_image,0,pad_height,0,pad_width,cv2.BORDER_CONSTANT,value=mean)
            if key_image is not None:
                key_image = cv2.copyMakeBorder(key_image,0,pad_height,0,pad_width,cv2.BORDER_CONSTANT,value=mean)
            seg_label = cv2.copyMakeBorder(seg_label,0,pad_height,0,pad_width,cv2.BORDER_CONSTANT,value=self.ignore_label)

            # update height/width
            data_height = current_image.shape[0]
            data_width = current_image.shape[1]
            label_height = seg_label.shape[0]
            label_width = seg_label.shape[1]
        #crop
        if  train:
            # random crop
            h_off = np.random.randint(data_height - crop_height+ 1)
            w_off = np.random.randint(data_width - crop_witdh+ 1)
        else:
            h_off = (data_height - crop_height) / 2
            w_off = (data_width - crop_witdh) / 2

        # roi image
        if key_crop_imgage is not None:
            return current_image[h_off:h_off+crop_height, w_off:w_off+crop_witdh,:],key_image[h_off:h_off+crop_height,
        else:
            return current_image[h_off:h_off+crop_height, w_off:w_off+crop_witdh,:],
                w_off:w_off+crop_witdh,:],seg_label[h_off:h_off+crop_height,w_off:w_off+crop_witdh]




def image_crop(image,label,crop_height,crop_witdh,crop_strategy='resize'):
    '''
    data arguement using crop and implemented strategys for crop are: 1)resize; 2)random crop
    '''
    if crop_strategy == 'resize':
       return cv2.resize(image,(crop_witdh,crop_height)),cv2.resize(label,(crop_witdh,crop_height),interpolation=cv2.INTER_NEAREST) 
    elif crop_strategy == 'random':
        pass
    else:
        raise Exception("not implemented crop strategy",crop_strategy)
def split_forward(process,img,fea_cha,crop_size):
    '''
    apply stride method to split the big image and forward a image grid seperately
    inputs:
    - process: implement forward
    - img: is preprocesed and with shape(c,h,w)
    - fea_cha: number of feature channels
    - crop_size: suitable size for net forward
    '''
    stride_rate = 2/3.
    stride = math.ceil(crop_size * stride_rate)
    
    _,h,w = img.shape
    h_grid = int(math.ceil(float(h-crop_size)/stride) + 1)
    w_grid = int(math.ceil(float(w-crop_size)/stride) + 1)


    data = np.zeros((fea_cha,h,w),dtype=np.float32)
    count = np.zeros((h,w),dtype=np.int32)

    for grid_h in range(h_grid):
        for grid_w in range(w_grid):
            s_h = grid_h * stride
            s_w = grid_w * stride

            e_h = min(s_h + crop_size, h)
            e_w = min(s_w + crop_size, w)
            s_h = e_h - crop_size
            s_w = e_w - crop_size
            img_sub = img[:,s_h:e_h,s_w:e_w]
            count[s_h:e_h,s_w:e_w] += 1
            out = process(img_sub)
            data[:,s_h:e_h,s_w:e_w] += out
    data = data / count
    return data
