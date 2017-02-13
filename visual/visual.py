import numpy as np
import copy
from PIL import Image 

class Visual(object):
    '''
    This script is used to show the image and its label
    '''
    def __init__(self):
        pass
    def config(self, palette_data_path):
    #palette_data_path supply the palette image
        palette_im = Image.open(palette_data_path)
        self.palette = palette_im.palette 
        print 'init finished'

    def palette_out(self,label_im):
       # render the labe_im to color image using palette     
        print 'begin palette' 
        if label_im.ndim == 3:
            label_im = label_im[0]
        label = Image.fromarray(label_im,mode='P')
        label.palette = copy.copy(self.palette)
        return label
    def color_image(self,image,num_classes=30):
        print num_classes
        import matplotlib as mpl
        norm = mpl.colors.Normalize(vmin=0.,vmax=num_classes)
        mycm = mpl.cm.get_cmap('Set1')
        return mycm(norm(image))
    def rescore(self,img,label,class_file=None,ignore_label=255):
        '''
        This function is used to show image and its label
        - img
        - label:size(w,h) and uint8
        - class_file: txt file, which define the semantic of label
        - ignore_label: the ignore_label will not shom in labe image
        '''
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print dir_path
        import matplotlib.pyplot as plt
        plt.rcParams['image.interpolation'] = 'nearest'
        if class_file is None:
		 class_file = os.path.join(dir_path,'classes.txt')
        all_labels =  open(class_file).readlines()
        #all_labels = ["0:Background"] + open(class_file).readlines()
        label[label==ignore_label] = 0
        scores = np.unique(label)
        print scores

        labels = [all_labels[s] for s in scores]
        num_scores = len(scores)

        def rescore(c):
            return np.where(scores == c)[0][0]

        rescore = np.vectorize(rescore)
        painted = rescore(label)
        plt.figure(figsize=(10,10))
        plt.imshow(painted,cmap=plt.cm.get_cmap('jet',num_scores))
        
        formatter = plt.FuncFormatter(lambda val, loc: labels[val])
        plt.colorbar(ticks=range(0,num_scores),format=formatter)
        plt.clim(-0.5, num_scores-0.5)

        if img is not None: 
            plt.figure(5)
            plt.imshow(img)
        plt.show()
    def label_to_paint(self,label,label_map,ignore_label=255):
        '''
        convery the segmentation label to paint image
        - label:(w,h), label(i,j) represent the pixel class
        - label_map: list,label_map[i] represent the semantic of class i
        - ignore_label: int, -1 represent no ignore label
        '''

        if ignore_label != -1:
            ig_num = len(label_map)
            label_map.append("{}:Ignore label".format(ig_num))
            label[label == ignore_label] = ig_num
        scores = np.unique(label)
        print scores
        labels = [label_map[s] for s in scores]
        num_scores = len(scores)

        def rescore(c):
            return np.where(scores == c)[0][0]

        rescore = np.vectorize(rescore)
        painted = rescore(label)
        return painted, labels,num_scores
    def show_label(self,gt_label,pd_label,class_file=None,ignore_label=255):
        '''
        This function is used to show image and its label
        - img
        - label:size(w,h) and uint8
        - class_file: txt file, which define the semantic of label
        - ignore_label: the ignore_label will not shom in labe image
        '''
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print dir_path
        import matplotlib.pyplot as plt
        plt.rcParams['image.interpolation'] = 'nearest'
        if class_file is None:
		 class_file = os.path.join(dir_path,'classes.txt')
        all_labels =  open(class_file).readlines()
        print type(all_labels)
        #all_labels = ["0:Background"] + open(class_file).readlines()
        plt.figure(figsize=(10,10))
        #p1 = plt.subplot(211)
        #p2 = plt.subplot(212)
        pd_paint,labels_ ,num_scores_ = self.label_to_paint(pd_label,all_labels,-1)
        print 'pd',num_scores_,len(labels_)
        print 'pd unique',np.unique(pd_paint)

        plt.imshow(pd_label,cmap=plt.cm.get_cmap('jet',num_scores_))
        
        formatter_ = plt.FuncFormatter(lambda val, loc: labels_[val])
        plt.colorbar(ticks=range(0,num_scores_),format=formatter_)
        plt.clim(-0.5, num_scores_ - 0.5)
        print 'gt'        
        gt_paint,labels,num_scores = self.label_to_paint(gt_label,all_labels)
        plt.figure(5)
        plt.imshow(gt_paint,cmap=plt.cm.get_cmap('jet',num_scores))
        
        formatter = plt.FuncFormatter(lambda val, loc: labels[val])
        plt.colorbar(ticks=range(0,num_scores),format=formatter)
        plt.clim(-0.5, num_scores-0.5)

        plt.show()

    def draw_label(self,label,class_file = None, ignore_label = 255):
        '''
        This function is used to show image and its label
        - img
        - label:size(w,h) and uint8
        - class_file: txt file, which define the semantic of label
        - ignore_label: the ignore_label will not shom in labe image
        '''
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print dir_path
        import matplotlib.pyplot as plt
        plt.rcParams['image.interpolation'] = 'nearest'
        if class_file is None:
		 class_file = os.path.join(dir_path,'classes.txt')
        all_labels =  open(class_file).readlines()
        plt.figure(figsize=(10,10))
        # convery the label to paintable label
        paint_label,labels_map ,num_scores = self.label_to_paint(label,all_labels)

        plt.imshow(paint_label,cmap=plt.cm.get_cmap('jet',num_scores))
        
        formatter = plt.FuncFormatter(lambda val, loc: labels_map[val])
        plt.colorbar(ticks=range(0,num_scores),format=formatter)
        plt.clim(-0.5, num_scores - 0.5)

        plt.show() 
