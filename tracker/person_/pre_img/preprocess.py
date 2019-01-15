import PIL
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


for i in os.listdir('/home/msadegh/workspace/embedded_project/tracker/person_/precarious_dataset/trainAno/'):
    annotation = open('/home/msadegh/workspace/embedded_project/tracker/person_/precarious_dataset/trainAno/'+i)
    annotation = annotation.readlines()[1:]
    for j in annotation:
        g = 0
        annot = j.split()
        image = Image.open('/home/msadegh/workspace/embedded_project/tracker/person_/precarious_dataset/train/'+ i[0:-4]+ '.jpg')
        new_image = image.crop((int(annot[1]), int(annot[2]), int(annot[1])+ int(annot[3]), int(annot[2])+ int(annot[4])))
        new_image = new_image.resize((120,360))
        new_image.save('/home/msadegh/workspace/embedded_project/tracker/person_/person_dataset/'+
                       i[0:-4] + '_' +str(g) + '_1.jpg')
        g+=1



for i in os.listdir('/home/msadegh/workspace/embedded_project/tracker/person_/precarious_dataset/trainAno/'):
    annotation = open('/home/msadegh/workspace/embedded_project/tracker/person_/precarious_dataset/trainAno/'+i)
    annotation = annotation.readlines()[1:]
    for j in annotation:
        g = 0
        annot = j.split()
        image = Image.open('/home/msadegh/workspace/embedded_project/tracker/person_/precarious_dataset/train/'+ i[0:-4]+ '.jpg')
        new_image = image.crop((int(annot[1])+200, int(annot[2]), int(annot[1])+ int(annot[3])+200, int(annot[2])+ int(annot[4])))
        new_image = new_image.resize((120,360))
        new_image.save('/home/msadegh/workspace/embedded_project/tracker/person_/not_person/'+
                       i[0:-4] + '_' +str(g) + '_0.jpg')
        g+=1
