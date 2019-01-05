import os
from PIL import Image
import numpy as np
import shutil
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

def im_shape(im):
    img = np.asarray(im)
    img_shape = img.shape
    print 'img_shape:',img_shape
    return img_shape[0],img_shape[1]
def resize_val_image(PATH,PATH_OUT,TYPE,w1,w1_cut,w2,w2_cut,h1,h_cut):
    if os.path.exists(PATH_OUT):
        shutil.rmtree(PATH_OUT)
        print 'rm -r ',PATH_OUT
    if not os.path.exists(PATH_OUT):
        os.makedirs(PATH_OUT)
        print 'mkdir',PATH_OUT
    for f in os.listdir(PATH):
        image_file = os.path.join(PATH,f)
        im = Image.open(image_file)
        
        Height,Width = im_shape(im)
        
        H_cut = 2
        W_cut = 5
        if Width == 3040:
            W_cut = w1_cut
            Width = w1
        else:
            W_cut = w2_cut
            Width = w2
        H_cut = h_cut
        Height = h1
       # print Height,Width,W_cut,H_cut

        im = im.resize((Width,Height),Image.ANTIALIAS)
        #im_shape(im)
        h = Height/H_cut
        w = Width/W_cut
        print Height,Width,H_cut,W_cut,h,w
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

def redefine_trainval_list(name,nums):
    if not os.path.exists(LIST_PATH_OUT):
        os.makedirs(LIST_PATH_OUT)
    f = open(os.path.join(LIST_PATH,name),'r')
    of = open(os.path.join(LIST_PATH_OUT,name),'w')
    for line in f:
        line = line.strip()
        for index in range(nums):
            new_line = line+'-'+ str(index)
            of.write(new_line+'\n')
    f.close()
    of.close()

def val_resize_data():
    W1 = [2740,2432,2128,1824,1521,1216,912,608]
    C1 = [5,4,4,3,3,2,2,1]
    W2 = [2810,2496,2184,1872,1560,1248,936,624]
    C2 = [5,4,4,3,3,2,2,1]
    H = [1844,1638,1434,1228,1024,820,615,410]
    C3 = [2,2,2,2,1,1,1,1]
    i = 7
    resize_val_image(IMAGE_PATH,IMAGE_PATH_OUT,'jpg',W1[i],C1[i],W2[i],C2[i],H[i],C3[i])
    resize_val_image(LABEL_PATH,LABEL_PATH_OUT,'png',W1[i],C1[i],W2[i],C2[i],H[i],C3[i])
    redefine_trainval_list('val.txt',C1[i]*C3[i])
if __name__ == '__main__':
    val_resize_data()
    #split_train_test()
    #3040 w1,w1 split num,3120 w2,w2 split num,h,h split num
    #resize_val_image(IMAGE_PATH,IMAGE_PATH_OUT,'jpg',2740,5,2810,5,1844,2)
    #split_image(LABEL_PATH,LABEL_PATH_OUT,'png')
    #ls = ['train.txt','val.txt','trainval.txt']
    #for l in ls:
    #    redefine_trainval_list(l)
