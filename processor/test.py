
from .AngleCalculation import *
import numpy as np
import json
from pprint import pprint
import csv
import cv2
import numpy



def test(v_name,ass_id):
    
    cc=AngleCalculation(host='192.168.2.106', port=3306, user='eguser@192.168.5.19', password='Fred!*!@!&123', db='mydb',table_name='EG_CLEAN_TMP' , video_name=v_name.split('.')[0])
    print('The neck angle is',cc.CalNeckAngle(framenum=0))
    print('The left shoulder angle is', cc.CalLeftShoulderAngle())
    print('The right shoulder angle is', cc.CalRightShoulderAngle())
    print('The left elbow angle is', cc.CalLeftElbow())
    print('The left elbow angle is', cc.CalRightElbow())
    print('The back angle is', cc.CalBackAngle())
    print('The Neck Posture is', cc.calNecktime())
    print('The Right Elbow Posture is', cc.calREtime())
    print('The Left Elbow Posture is',cc.calLEtime())
    print('The Back Posture is',cc.calBtime())
    print('The Right Shoulder Posture is', cc.calRStime())
    print('The Left Shoulder Posture is', cc.calLStime())
    print('Orientation', cc.ShowOrientation())
    

    
    cc.savesqldata(host='192.168.2.106', port=3306, user='eguser@192.168.5.19', password='Fred!*!@!&123', db='ergo_raw', table_name='data_posture', video_name=v_name.split('.')[0],assessment_id=ass_id)
    cc.saveoperator(host='192.168.2.106', port=3306, user='eguser@192.168.5.19', password='Fred!*!@!&123', db='ergo_raw',table_name='data_operator', video_name=v_name.split('.')[0])
    cc.savehandling(host='192.168.2.106', port=3306, user='eguser@192.168.5.19', password='Fred!*!@!&123', db='ergo_raw',table_name='data_handling', video_name=v_name.split('.')[0], video_path=r'/home/fred/dev/project/st-gcn/data/demo_result/{}'.format(v_name))

    cc.VideoCap(videopath=r'/home/fred/dev/project/st-gcn/data/demo_result/{}'.format(v_name),writeimagepath=r'/home/fred/dev/project/st-gcn/processor/Image')

    cc.readimage(datapath=r'/home/fred/dev/project/st-gcn/processor/Image')
    cc.drawskeleton()
    cc.DrawNeckAngle()
    cc.DrawLSAngle()
    cc.DrawRSAngle()
    cc.DrawBackAngle()
    cc.DrawLEAngle()
    cc.DrawREAngle()
    cc.Ninegridbox()
    #cc.writeimage(path=r'/home/fred/dev/project/st-gcn/processor/Test')
    cc.outputimage(dirpath=r'/home/fred/dev/project/dataServer/downloads/{}'.format(v_name.split('.')[0]))

    cc.ImagetoVideo(output=r'/home/fred/dev/project/dataServer/downloads/{}/V_{}'.format(v_name.split('.')[0],v_name))




    
