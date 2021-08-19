import cv2
import numpy as np
import json
import glob
import os

def calcuAngle(a):
    if a[0][0]==a[1][0] or a[0][1]== a[1][1] :
        return 0
    else:
        try:
            if (a[1][1]-a[0][1])/(a[0][0]-a[1][0])>0:
                return np.arctan((a[1][1]-a[0][1])/(a[0][0]-a[1][0]))
            else:
                return np.arctan((a[0][0] - a[1][0]) / (a[0][1] - a[1][1]))
        except Exception as e:
            print(str(e))

cropdata = np.array([[0,0],
                         [0,58],
                         [29,29],
                         [58,0],
                         [58,58]]) #頂點裁剪
depthf = glob.glob(os.path.join('C:/Users/Administrator/Desktop/DatabaseTT/one/', '*d.jpg'))
length = len(depthf)
rgbf = [f.replace('d.jpg', '.jpg') for f in depthf]
jsonf = [f.replace('.jpg', '.json') for f in rgbf]

for m in range(len(depthf)):
    print("working on ",rgbf[m][46:-4],":",str(int(m/length*100)),"%")
    rgb_o = cv2.imread(rgbf[m],-1)
    depth_o = cv2.imread(depthf[m],-1)
    Angle_o = np.zeros((658,658,1),dtype=float)
    Grasp_o = np.zeros((658,658,1))
    with open(jsonf[m],'r') as load_f:
         load_dict = json.load(load_f)
         for i in load_dict['shapes']:
             if i['shape_type'] =='circle':
                 r2 = pow((i['points'][1][0] - i['points'][0][0]), 2) + pow((i['points'][1][1] - i['points'][0][1]), 2)
                 for j in load_dict['shapes']:
                     if j['shape_type'] =='line':
                         if i['label'][1:] == j['label'][1:]:
                             Angle = calcuAngle(j['points'])
                 for x in range(0,658):
                     for y in range(0,658):
                         if pow((x-i['points'][0][0]),2)+pow((y-i['points'][0][1]),2) <= r2:
                             Grasp_o[y][x][0]=255
                             Angle_o[y][x][0]=Angle
    for t in range(5): #crop and save
        cv2.imwrite("C:/Users/Administrator/Desktop/rgbdGccData/"+str(t)+"/"+str(rgbf[m][46:-4])+".jpg",rgb_o[cropdata[t][1]:cropdata[t][1]+600,cropdata[t][0]:cropdata[t][0]+600])
        cv2.imwrite("C:/Users/Administrator/Desktop/rgbdGccData/"+str(t)+"/"+str(rgbf[m][46:-4])+"d.jpg",depth_o[cropdata[t][1]:cropdata[t][1]+600,cropdata[t][0]:cropdata[t][0]+600])
        cv2.imwrite("C:/Users/Administrator/Desktop/rgbdGccData/"+str(t)+"/"+str(rgbf[m][46:-4])+"q.jpg",Grasp_o[cropdata[t][1]:cropdata[t][1]+600,cropdata[t][0]:cropdata[t][0]+600])
        np.save("C:/Users/Administrator/Desktop/rgbdGccData/"+str(t)+"/"+str(rgbf[m][46:-4])+".npy",Angle_o[cropdata[t][1]:cropdata[t][1]+600,cropdata[t][0]:cropdata[t][0]+600])