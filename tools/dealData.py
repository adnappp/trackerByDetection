import os
import random

'''这个文件的功能主要是在已有数据集选取训练集，验证集，测试集，这几个txt文件里面存放的是xml和image文件的名称，如00001'''
trainval_percent = 0.66
train_percent = 0.5
xmlfilepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations'
#这个地方imagpath也没用到，只需要xml或者image中一个就可以，可以去掉
imagefilepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages'
#txtsavepath 就是之前创建的空文件夹，但是这里没有用到，我后面直接填写了，没有用这个代替
txtsavepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main'
total_xml = os.listdir(xmlfilepath) #获取所有的xml文件

num=len(total_xml) #有多少xml文件
list_xml =range(num) #改list为list_xml(命名规范)，list_xml就是从0-num的一个list
tv=int(num*trainval_percent) #取66%xml来作为训练验证集
tr=int(tv*train_percent)  #再从训练验证集中取50%作为训练集
trainval= random.sample(list_xml,tv) #从总的xml文件中随机选择66%的文件，trainval里面是文件名称
train=random.sample(trainval,tr) #同上

#这里就是和上面的txtsavepath重复的地方，可以用txtsavepath代替前面一样的路径
ftrainval = open('/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt', 'w')
ftest = open('/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt', 'w')
ftrain = open('/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt', 'w')
fval = open('/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt', 'w')

def deal():
    for i  in list_xml:
      #遍历所有xml文件
        name=total_xml[i].split(".")[0]+ '\n'
        if i in trainval: #对于在之前选取的训练验证集中的文件名称
           ftrainval.write(name) #写入trainval.txt
           if i in train: #又在其中选择50%的写入trian.txt
              ftrain.write(name)
           else: #另外训练验证集的50%写入验证集
              fval.write(name)
        else: #剩下的34%写入测试集test.txt
            ftest.write(name)
    #关闭文件
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest .close()

if __name__ == "__main__":
    deal()