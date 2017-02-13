import sys
import cv2
import numpy as np
def decode_label(label_file):
    '''
    Read label image as raw data.
    '''
    with open(label_file,'rb') as infile:
        buf = infile.read()
    raw = np.fromstring(buf,dtype='uint8')
    img = cv2.imdecode(raw,cv2.IMREAD_UNCHANGED)
    return img


class Cityscapes(object):
    '''
    some helper function for Cityscapes dataset.
    '''
    data_dir = '/home/yuzhile/data/cityscapes'
    sys.path.insert(0,'{}/scripts/helpers/'.format(data_dir))
    labels = __import__('labels')
    id2trainId ={label.id: label.trainId for label in labels.labels}
    w = 2048
    h = 1024
    c_image = 4
    c_sequence = 3


    @classmethod
    def assing_trainIds(clas_obj, label):
        '''
        Map the given label IDs to the train IDs appropriate for training.
        '''
        for k,v in clas_obj.id2trainId.iteritems():
            label[label == k] = v
        return label

    @classmethod
    def lmdb_load_image_label(cls_obj,image_txn,sequence_txn,idx,error_list,train_strategy='separate'):
        '''
        Load two frames from video squence: current image and key image, and the label coresponding to key image.
        - cast to float
        '''
        print 'idx ',idx 

        raw_data = np.fromstring(image_txn.get(idx),dtype='uint8')
        ndata = raw_data.reshape(cls_obj.h,cls_obj.w,cls_obj.c_image)
        image = ndata[:,:,:3]
        key_im = np.array(image,dtype=np.float32)
        # load from lmdb file
        label = ndata[:,:,-1]
        label = np.squeeze(label)
        #my_visual.draw_label(label)
        #print cls_obj.id2trainId.values()
        assert len(label.shape) == 2,'the image seg labe should be 2 dim'
 

        city, shot, frame = idx.split('_')
        if train_strategy == 'separate':
            SEQ_LEN = -4
        elif train_strategy == 'fix_n':
            SEQ_LEN = - np.random.randint(1,10)
        elif train_strategy == 'fix_f':
            SEQ_LEN = - np.random.randint(1,10)
        elif train_strategy == 'free':
            SEQ_LEN = np.random.randint(-9,10)
            if SEQ_LEN >=0:
                #cls_obj.strategy_id = cls_obj.strategy2int['fix_f']
                pass

        #key_im = cls_obj.half_crop_image(key_im,position,label=False)
        key_in_ = np.array(key_im,dtype=np.float32)
        #handing the crash image, we find the no-error image with decreasing the idx
        for i in range(10):
            current_name = '{}_{}_{:0>6d}'.format(city,shot,int(frame) + SEQ_LEN-i)
            if current_name in error_list:
                continue
            current_raw_im = np.fromstring(sequence_txn.get(current_name),dtype='uint8')
            current_im = current_raw_im.reshape(cls_obj.h,cls_obj.w,cls_obj.c_sequence)
            #current_im = cls_obj.half_crop_image(current_im,position,label=False)
            current_in_ = np.array(current_im,dtype=np.float32)
            break

        
        return current_in_,key_in_, label

