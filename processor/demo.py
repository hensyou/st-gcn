#!/usr/bin/env python
import os
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils
import cv2
from .test import *

class Demo(IO):
    """
        Demo for Skeleton-based Action Recgnition
    """

    def load_to_db(self,video_path,video):
        print("load_to_db")
        import pymysql as mysql
        import cv2
        db = mysql.connect("127.0.0.1","eguser@192.168.5.19","Fred!*!@!&123","mydb")
        cursor = db.cursor()
        sql = "delete from EG_CLEAN_TMP where VIDEO_NAME = '"+video+"'"
        cursor.execute(sql)
        db.commit()
        cap = cv2.VideoCapture(video_path)
        print(sql)

        if not cap.isOpened():
            print
            "could not open :", video_path
            return

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(3))
        height = int(cap.get(4))
        print('width',width)
        fps = cap.get(5)

        for i_loop in range(0,length,1):
            if i_loop < 10:
                filename = '/home/ergoadm/dev/project/st-gcn/data/openpose_estimation/snippets/'+video+'/'+video+'_00000000000'+str(i_loop)+'_keypoints.json'
            elif i_loop < 100:
                filename = '/home/ergoadm/dev/project/st-gcn/data/openpose_estimation/snippets/'+video+'/'+video+'_0000000000' + str(i_loop) + '_keypoints.json'
            elif i_loop < 1000:
                filename = '/home/ergoadm/dev/project/st-gcn/data/openpose_estimation/snippets/'+video+'/'+video+'_000000000' + str(i_loop) + '_keypoints.json'
            elif i_loop < 10000:
                filename = '/home/ergoadm/dev/project/st-gcn/data/openpose_estimation/snippets/'+video+'/'+video+'_00000000' + str(i_loop) + '_keypoints.json'
            import os.path
            if os.path.isfile(filename) != True :
                sql = "INSERT INTO EG_CLEAN_TMP(VIDEO_NAME,FRAME_NUMBER,PERSON_COUNT,PERSON_NUMBER) VALUES ('"+video+"',"+str(i_loop)+",0,0)"
                try:
                    print(sql)
                    cursor.execute(sql)
                    db.commit()
                except:
                    db.rollback()
                continue;
            data = json.load(open(filename))
            person_number = 0
            #if len(data['people']) == 0:
            #    sql = "INSERT INTO EG_CLEAN_TMP(VIDEO_NAME,FRAME_NUMBER,PERSON_COUNT,PERSON_NUMBER) VALUES ('"+video+"',"+str(i_loop)+",0,0)"
            #    try:
            #        print(sql)
            #        cursor.execute(sql)
            #        db.commit()
            #    except:
            #        db.rollback()
            #else:
            for person in data['people']:
                    score, coordinates = [], []
                    skeleton = {}
                    keypoints = person['pose_keypoints_2d']
                    l_hands = person['hand_left_keypoints_2d']
                    r_hands = person['hand_right_keypoints_2d']
                    sql = "INSERT INTO EG_CLEAN_TMP(VIDEO_NAME,FRAME_NUMBER,PERSON_COUNT,PERSON_NUMBER,\
                     B_0_X,B_0_Y,S_0,B_1_X,B_1_Y,S_1,B_2_X,B_2_Y,S_2,B_3_X,B_3_Y,S_3,B_4_X,B_4_Y,S_4,B_5_X,B_5_Y,S_5,B_6_X,B_6_Y,S_6,\
                     B_7_X,B_7_Y,S_7,B_8_X,B_8_Y,S_8,B_9_X,B_9_Y,S_9,B_10_X,B_10_Y,S_10,B_11_X,B_11_Y,S_11,B_12_X,B_12_Y,S_12,B_13_X,B_13_Y,S_13,\
                     B_14_X,B_14_Y,S_14,B_15_X,B_15_Y,S_15,B_16_X,B_16_Y,S_16,B_17_X,B_17_Y,S_17,\
                     L_0_X,L_0_Y,L_0,L_1_X,L_1_Y,L_1,L_2_X,L_2_Y,L_2,L_3_X,L_3_Y,L_3,L_4_X,L_4_Y,L_4,\
                     L_5_X,L_5_Y,L_5,L_6_X,L_6_Y,L_6,L_7_X,L_7_Y,L_7,L_8_X,L_8_Y,L_8,\
                     L_9_X,L_9_Y,L_9,L_10_X,L_10_Y,L_10,L_11_X,L_11_Y,L_11,L_12_X,L_12_Y,L_12,L_13_X,L_13_Y,L_13,\
                     L_14_X,L_14_Y,L_14,L_15_X,L_15_Y,L_15,L_16_X,L_16_Y,L_16,L_17_X,L_17_Y,L_17,L_18_X,L_18_Y,L_18,\
                     L_19_X,L_19_Y,L_19,L_20_X,L_20_Y,L_20,\
                     R_0_X,R_0_Y,R_0,R_1_X,R_1_Y,R_1,R_2_X,R_2_Y,R_2,R_3_X,R_3_Y,R_3,R_4_X,R_4_Y,R_4,\
                     R_5_X,R_5_Y,R_5,R_6_X,R_6_Y,R_6,R_7_X,R_7_Y,R_7,R_8_X,R_8_Y,R_8,\
                     R_9_X,R_9_Y,R_9,R_10_X,R_10_Y,R_10,R_11_X,R_11_Y,R_11,R_12_X,R_12_Y,R_12,R_13_X,R_13_Y,R_13,\
                     R_14_X,R_14_Y,R_14,R_15_X,R_15_Y,R_15,R_16_X,R_16_Y,R_16,R_17_X,R_17_Y,R_17,R_18_X,R_18_Y,R_18,\
                     R_19_X,R_19_Y,R_19,R_20_X,R_20_Y,R_20\
                     ) VALUES ('"+video+"',"+str(i_loop)+","+str(len(data['people']))+","+str(person_number)
                    for i in range(0, 54, 3):
                        x,y,score = [keypoints[i], keypoints[i + 1],keypoints[i + 2]]
                        sql = sql + ","+str(x) +","+str(y)+","+str(score)
                    for i in range(0, 63, 3):
                        x,y,score = [l_hands[i], l_hands[i + 1],l_hands[i + 2]]
                        sql = sql + ","+str(x) +","+str(y)+","+str(score)
                    for i in range(0, 63, 3):
                        x,y,score = [r_hands[i], r_hands[i + 1],r_hands[i + 2]]
                        sql = sql + ","+str(x) +","+str(y)+","+str(score)
                    sql=sql+")"
                    print (sql)
                    person_number = person_number+1
                    try:
                        #print(sql)
                        cursor.execute(sql)
                        db.commit()
                    except:
                        db.rollback()
        db.close()
        return

    def update_label_to_db(self,video,label_name_sequence):
        import pymysql as mysql
        db = mysql.connect("127.0.0.1","eguser@192.168.5.19","Fred!*!@!&123","mydb")
        cursor = db.cursor()
        print('update_label_to_db')

        for n in range(0,len(label_name_sequence),1):
            sql = 'update EG_CLEAN_TMP set LABEL_NAME = "{}" where VIDEO_NAME = "{}" and PERSON_NUMBER=0 and FRAME_NUMBER = {}'.format(label_name_sequence[n][0],video,str(4*n))
            cursor.execute(sql)
            sql = 'update EG_CLEAN_TMP set LABEL_NAME = "{}" where VIDEO_NAME = "{}" and PERSON_NUMBER=0 and FRAME_NUMBER = {}'.format(label_name_sequence[n][0],video,str(4*n+1))
            cursor.execute(sql)
            sql = 'update EG_CLEAN_TMP set LABEL_NAME = "{}" where VIDEO_NAME = "{}" and PERSON_NUMBER=0 and FRAME_NUMBER = {}'.format(label_name_sequence[n][0],video,str(4*n+2))
            cursor.execute(sql)
            sql = 'update EG_CLEAN_TMP set LABEL_NAME = "{}" where VIDEO_NAME = "{}" and PERSON_NUMBER=0 and FRAME_NUMBER = {}'.format(label_name_sequence[n][0],video,str(4*n+3))
            cursor.execute(sql)
            sql = 'update EG_CLEAN_TMP set LABEL_NAME = "{}" where VIDEO_NAME = "{}" and PERSON_NUMBER=1 and FRAME_NUMBER = {}'.format(label_name_sequence[n][1],video,str(4*n))
            cursor.execute(sql)
            sql = 'update EG_CLEAN_TMP set LABEL_NAME = "{}" where VIDEO_NAME = "{}" and PERSON_NUMBER=1 and FRAME_NUMBER = {}'.format(label_name_sequence[n][1],video,str(4*n+1))
            cursor.execute(sql)
            sql = 'update EG_CLEAN_TMP set LABEL_NAME = "{}" where VIDEO_NAME = "{}" and PERSON_NUMBER=1 and FRAME_NUMBER = {}'.format(label_name_sequence[n][1],video,str(4*n+2))
            cursor.execute(sql)
            sql = 'update EG_CLEAN_TMP set LABEL_NAME = "{}" where VIDEO_NAME = "{}" and PERSON_NUMBER=1 and FRAME_NUMBER = {}'.format(label_name_sequence[n][1],video,str(4*n+3))
            cursor.execute(sql)
            print('sql:'+sql)
        print('committing')
        db.commit()
        print('committed')

    def update_video_state(self,video_name,state):
        import pymysql as mysql
        db = mysql.connect("127.0.0.1","eguser@192.168.5.19","Fred!*!@!&123","ergo_raw")
        cursor = db.cursor()
        sql = 'update data_video set video_status = {},processed_file_name = "V_{}",processed_file_path = "/home/ergoadm/dev/project/dataServer/downloads/{}" where video_name like "{}%"'.format(state,video_name, video_name.split('.')[0], video_name.split('.')[0]) 
        print('update_video_state')
        print(sql)
        try:
        
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()
        db.close()
        return
 
        


    def start(self):
      video_name = self.arg.video.split('/')[-1].split('.')[0]
      try:
        openpose = '{}/examples/openpose/openpose.bin'.format(self.arg.openpose)
        output_snippets_dir = './data/openpose_estimation/snippets/{}'.format(video_name)
        output_sequence_dir = './data/openpose_estimation/data'
        output_sequence_path = '{}/{}.json'.format(output_sequence_dir, video_name)
        output_result_dir = self.arg.output_dir
        output_result_path = '{}/{}.mp4'.format(output_result_dir, video_name)
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]

        # pose estimation
        openpose_args = dict(
            video=self.arg.video,
            write_json=output_snippets_dir,
            display=0,
            model_folder='/home/ergoadm/dev/project/openpose/models',
            render_pose=0,
            model_pose='COCO')
        command_line = openpose + ' --hand '
        #command_line = openpose + ' '
        command_line += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])
        shutil.rmtree(output_snippets_dir, ignore_errors=True)
        os.makedirs(output_snippets_dir)
        print('******'+command_line)
        os.system(command_line)
        # pack openpose ouputs
       
        video = utils.video.get_video_frames(self.arg.video)
        height, width, _ = video[0].shape
        #video_info = utils.egdata.json_pack(
        #    output_snippets_dir, video_name, width,height)
        #print('dir='+output_snippets_dir)
        #print('video='+video_name)
        #if not os.path.exists(output_sequence_dir):
        #    os.makedirs(output_sequence_dir)
        #with open(output_sequence_path, 'w') as outfile:
        #    json.dump(video_info, outfile)
        #if len(video_info['data']) == 0:
        #    print('Can not find pose estimation results.')
        #    return
        #else:
        print('Pose estimation complete.width='+str(width)+'height='+str(height))
        # load to db
        self.load_to_db(self.arg.video,video_name) 
        utils.egdata.adjust(video_name)
        print('start video_info')
        video_info = utils.egdata.json_pack_from_db(output_snippets_dir, video_name, width,height)
        # parse skeleton data
        print('start parsing video')
        pose, _ = utils.video.video_info_parsing(video_info)
        print('parsing result')
        print('pose shape:')
        print(pose.shape)
        data = torch.from_numpy(pose)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()

        # extract feature
        print('\nNetwork forwad...')
        self.model.eval()
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        print('output=')
        print(output)
        print('feature=')
        print(feature)
        intensity = (feature * feature).sum(dim=0) ** 0.5
        intensity = intensity.cpu().detach().numpy()
        label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        print('Prediction result: {}'.format(label_name[label]))
        print('Done.')
        # visualization
        print('\nVisualization...')
        print(output.shape)
        print(output)
        label_sequence = output.sum(dim=2).argmax(dim=0)
        label_name_sequence = [[label_name[p] for p in l] for l in label_sequence]
        print('len='+str(len(label_name_sequence)))
        print('label_name_sequence:')
        print(label_name_sequence)
        self.update_label_to_db(video_name,label_name_sequence)        
        edge = self.model.graph.edge
        _, T, V, M = pose.shape
        print(' _, T,'+str(T)+' V,'+str(V)+' M'+str(M))

        
        images = utils.visualization.stgcn_visualize(
            pose, edge, intensity,  self.arg.video, label_name[label], label_name_sequence, self.arg.height)
        print('Done.')
        # save video
        print('\nSaving...')
        if not os.path.exists(output_result_dir):
            os.makedirs(output_result_dir)
        writer = skvideo.io.FFmpegWriter(output_result_path,inputdict={
                  '-r': '10'}, outputdict={'-b': '300000000'})
        #writer = skvideo.io.FFmpegWriter(output_result_path,
        #          outputdict={'-b': '300000000'})
        for img in images:
            writer.writeFrame(img)
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))
        video_name = self.arg.video.split('/')[-1]
        utils.egdata.save_to_pro_db(video_name)
        test(video_name,utils.egdata.get_assement_id(video_name))
        self.update_video_state(video_name,2)
      except Exception as e:
        print(e)
        self.update_video_state(video_name,-1)
        


    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/skateboarding.mp4',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default='3dparty/openpose/build',
                            help='Path to openpose')
        parser.add_argument('--output_dir',
                            default='./data/demo_result',
                            help='Path to save results')
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='Path to save results')
        parser.set_defaults(config='./config/st_gcn/kinetics-skeleton/demo.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
