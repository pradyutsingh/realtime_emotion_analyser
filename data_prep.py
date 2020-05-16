import os
import csv
import argparse
import numpy as np 
import scipy.misc
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help="path of the csv file")
parser.add_argument('-o', '--output', required=True, help="path of the output")

args=parser.parse_args()

#we will reshape to 48 by 48 as the image is given as a string which is a row (1,2304)
w=48
h=48

image = np.zeros((h, w), dtype=np.uint8)
id = 1
emo_list = []

with open(args.file , 'r') as csvfile:
    dataread = csv.reader(csvfile,delimiter = ',')
    headers = next(dataread)
    print(headers)
    
    for row in dataread: 
        emotions = row[0]
        pixel = map(int,row[1].split())
        usage = row[2]
        
        pixel_array = np.asarray(list(pixel))
        image = np.reshape(pixel_array,(w,h))
        
        stacked_image = np.dstack((image,)*3)
        
        emo_list += emotions
        
        imagefolder = os.path.join(args.output,usage)
        if not os.path.exists(imagefolder):
            os.makedirs(imagefolder)
        
        imagefile = os.path.join(imagefolder,str(id)+'.jpg')
        imageio.imwrite(imagefile,stacked_image)
        
        id+=1
        
        #to see the progress of the save
        if id%100 == 0:
            print("processed {} images".format(id))
            
np.savetxt("emotions.csv",emo_list,delimiter=',',fmt="%s")
print("finished processing {} images".format(id))
        