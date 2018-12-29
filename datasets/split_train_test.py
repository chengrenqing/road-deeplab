import os
from PIL import Image
import numpy as np

def split_train_test():
    IMAGE_PATH = './pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages/'
    out_path = './pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    p1 = os.path.join(out_path,'train.txt')
    p2 = os.path.join(out_path,'val.txt')
    p3 = os.path.join(out_path,'trainval.txt')
    if os.path.isfile(p1):
        os.remove(p1)
        os.remove(p2)
        os.remove(p3)
    train = open(p1,'w')
    val = open(p2,'w')
    trainval = open(p3,'w')
    num=0
    for f in os.listdir(IMAGE_PATH):
        num+=1
    a1 = int(num*0.8)
    i=0
    for fi in os.listdir(IMAGE_PATH):
        f = fi[:-4]
        i+=1
        if i<a1:
            train.write(f+'\n')
        elif i == a1:
            train.write(f)
        
        if i == num:
            val.write(f)
        elif i>a1:
            val.write(f+'\n')

        if i == num:
            trainval.write(f)
        else:
            trainval.write(f+'\n')
IMAGE_PATH = './pascal_voc_seg/VOCdevkit/VOC2012-ORI/JPEGImages/'
IMAGE_PATH_OUT = './pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages/'

LABEL_PATH = './pascal_voc_seg/VOCdevkit/VOC2012-ORI/SegmentationClassRaw/'
LABEL_PATH_OUT = './pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassRaw/'

def split_image(PATH,PATH_OUT,TYPE):
    if not os.path.exists(PATH_OUT):
        os.makedirs(PATH_OUT)
    for f in os.listdir(PATH):
        image_file = os.path.join(PATH,f)
        im = Image.open(image_file)
        img = np.asarray(im)
        img_shape = img.shape
        Height = img_shape[0]
        Width = img_shape[1]

        H_cut = 2
        W_cut = 5
        h = Height/H_cut
        w = Width/W_cut
        print im.format,img_shape,h,w
        index=0
        for i in range(H_cut):
            for j in range(W_cut):
                x1 = j*w
                x2 = x1+w
                y1 = i*h
                y2 = y1+h
                region = im.crop((x1,y1,x2,y2))
                if TYPE == 'jpg':
                    region = region.convert('RGB')
                #print np.asarray(region)
                #print region.format,region.mode,region.size
                o_name = f[:-4]+'-'+str(index)+'.'+TYPE
                out_name = os.path.join(PATH_OUT,o_name)
                if TYPE == 'png':
                    region.save(out_name,'PNG')
                else:
                    region.save(out_name)
                index+=1 
        #break
        print PATH,' finish!!!!'
LIST_PATH = './pascal_voc_seg/VOCdevkit/VOC2012-ORI/ImageSets/Segmentation/'
LIST_PATH_OUT = './pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/'

def redefine_trainval_list(name):
    if not os.path.exists(LIST_PATH_OUT):
        os.makedirs(LIST_PATH_OUT)
    f = open(os.path.join(LIST_PATH,name),'r')
    of = open(os.path.join(LIST_PATH_OUT,name),'w')
    for line in f:
        line = line.strip()
        for index in range(10):
            new_line = line+'-'+ str(index)
            of.write(new_line+'\n')
    f.close()
    of.close()

    
if __name__ == '__main__':
    #split_train_test()
    #split_image(IMAGE_PATH,IMAGE_PATH_OUT,'jpg')
    split_image(LABEL_PATH,LABEL_PATH_OUT,'png')
    #ls = ['train.txt','val.txt','trainval.txt']
    #for l in ls:
    #    redefine_trainval_list(l)
