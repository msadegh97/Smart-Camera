from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

myannotation = open(file='./annotation.csv', mode='w')
for i in os.listdir('/home/msadegh/workspace/embedded_project/tracker/precarious_dataset/trainAno/'):
    annotation = open('/home/msadegh/workspace/embedded_project/tracker/precarious_dataset/trainAno/'+i)
    annotation = annotation.readlines()[1:]
    for j in annotation:
        annot = j.split()
        myannotation.write('/home/msadegh/workspace/embedded_project/tracker/precarious_dataset/train/'+
                           i[0:-4] + '.jpg'+','+
                           annot[1] + ','+
                           annot[2]+ ','+
                           str(int(annot[3]) + int(annot[1])) + ',' +
                           str(int(annot[4]) + int(annot[2])) + ',' +
                           'Person' + '\n')
myannotation.flush()
myannotation.close()


