from pathlib import Path
import numpy as np
import json
import pandas as pd
import math

def json_pack(snippets_dir, video_name, frame_width, frame_height, label='unknown', label_index=-1):
    sequence_info = []
    p = Path(snippets_dir)
    for path in p.glob(video_name+'*.json'):
        json_path = str(path)
        print(path)
        frame_id = int(path.stem.split('_')[-2])
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            skeleton = {}
            keypoints = person['pose_keypoints_2d']
            for i in range(0, len(keypoints), 3):
                coordinates += [keypoints[i]/frame_width, keypoints[i + 1]/frame_height]
                score += [keypoints[i + 2]]
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index

    return video_info

def adjustFrame(first_frame,second_frame):
    need_adj = 0
    f_fPerson =  first_frame[first_frame.PERSON_NUMBER==0]
    s_fPerson = second_frame[second_frame.PERSON_NUMBER == 0]
    s_sPerson = second_frame[second_frame.PERSON_NUMBER == 1]
    print('f_f id=%d pn=%d x0=%d y0=%d x1=%d y2=%d x3=%d y3=%d' % \
          (f_fPerson.iloc[0]['ID'], f_fPerson.iloc[0]['PERSON_NUMBER'], f_fPerson.iloc[0]['B_0_X'],\
           f_fPerson.iloc[0]['B_0_Y'], f_fPerson.iloc[0]['B_1_X'], f_fPerson.iloc[0]['B_1_Y'],\
           f_fPerson.iloc[0]['B_2_X'], f_fPerson.iloc[0]['B_2_Y']))
    min_v=0
    min_i=0
    adjs=[]
    for index_x in range(19, 39, 1):
        if (f_fPerson.iloc[0][index_x] == 0):
            index_x = index_x + 1
        else:
            break
    for index_p in range(0,len(second_frame),1):
        s_Person = second_frame[second_frame.PERSON_NUMBER == 2]
        print('s_p id=%d pn=%d x0=%d y0=%d x1=%d y2=%d x3=%d y3=%d' % \
              (second_frame.iloc[index_p]['ID'],second_frame.iloc[index_p]['PERSON_NUMBER'],\
               second_frame.iloc[index_p]['B_0_X'],second_frame.iloc[index_p]['B_0_Y'],\
               second_frame.iloc[index_p]['B_1_X'],second_frame.iloc[index_p]['B_1_Y'],\
               second_frame.iloc[index_p]['B_2_X'],second_frame.iloc[index_p]['B_2_Y']))
        index_xx = index_x
        for index_xx in range(index_x, 39, 1):
            if (second_frame.iloc[index_p][index_xx] == 0):
                index_xx = index_xx + 1
            else:
                break
        #if (f_fPerson.iloc[0]['B_0_X'] != 0 and f_fPerson.iloc[0]['B_0_Y'] != 0 and second_frame.iloc[index_p]['B_0_X'] !=0 and second_frame.iloc[index_p]['B_0_Y'] !=0):
        xd = abs(f_fPerson.iloc[0][index_xx]-second_frame.iloc[index_p][index_xx]) ** 2 + abs(f_fPerson.iloc[0][index_xx+1]-second_frame.iloc[index_p][index_xx+1]) ** 2
        #elif (f_fPerson.iloc[0]['B_1_X'] != 0 and f_fPerson.iloc[0]['B_1_Y'] != 0 and second_frame.iloc[index_p]['B_1_X'] != 0 and second_frame.iloc[index_p]['B_1_Y'] != 0):
        #    xd = abs(f_fPerson.iloc[0]['B_1_X'] - second_frame.iloc[index_p]['B_1_X']) ** 2 + abs(f_fPerson.iloc[0]['B_1_Y'] - second_frame.iloc[index_p]['B_1_Y']) ** 2
        #elif (f_fPerson.iloc[0]['B_2_X'] != 0 and f_fPerson.iloc[0]['B_2_Y'] != 0 and second_frame.iloc[index_p]['B_2_X'] != 0 and second_frame.iloc[index_p]['B_2_Y'] != 0):
        #    xd = abs(f_fPerson.iloc[0]['B_2_X'] - second_frame.iloc[index_p]['B_2_X']) ** 2 + abs(f_fPerson.iloc[0]['B_2_Y'] - second_frame.iloc[index_p]['B_2_Y']) ** 2
        #else:
        #    #xd = abs(f_fPerson.iloc[0]['B_5_X'] - second_frame.iloc[index_p]['B_5_X']) ** 2 + abs(f_fPerson.iloc[0]['B_5_Y'] - second_frame.iloc[index_p]['B_5_Y']) ** 2
        #    return (second_frame, 0, adjs)
        print('person_number=%d xd=%f' % (second_frame.iloc[index_p]['PERSON_NUMBER'],xd))
        if index_p == 0:
            min_v = xd
            miv_i = second_frame.iloc[index_p]['ID']
        if xd < min_v:
            min_v = xd
            miv_i = second_frame.iloc[index_p]['ID']


    if second_frame[second_frame.ID == miv_i].iloc[0]['PERSON_NUMBER'] != 0:
        adj = [miv_i,0]
        adjs.append(adj)
        adj = [second_frame[second_frame.PERSON_NUMBER == 0].iloc[0]['ID'],second_frame[second_frame.ID == miv_i].iloc[0]['PERSON_NUMBER'] ]
        adjs.append(adj)
        second_frame.loc[second_frame.PERSON_NUMBER == 0,'PERSON_NUMBER'] = second_frame[second_frame.ID == miv_i].iloc[0]['PERSON_NUMBER']
        second_frame.loc[second_frame.ID == miv_i,'PERSON_NUMBER'] = 0

        print('0=',second_frame[second_frame.ID == miv_i].iloc[0]['PERSON_NUMBER'])
        print('1=',second_frame[second_frame.ID == adj[0]].iloc[0]['PERSON_NUMBER'])
        need_adj = 1

    return (second_frame,need_adj,adjs)



def adjust(video_name): 
    import pymysql as mysql
 
    db = mysql.connect("127.0.0.1","eguser@192.168.5.19","Fred!*!@!&123","mydb")
    cursor = db.cursor()

    sql='SELECT * FROM EG_CLEAN_TMP WHERE VIDEO_NAME="{}" ORDER BY FRAME_NUMBER asc '.format(video_name)
    df = pd.read_sql(sql, con=db)
    
    frames = df.groupby('FRAME_NUMBER')
    i=0
    adj_buffer = []
    for key, value in frames:
        if i  == 0:
            first_frame=value
        else:
            second_frame = value
            (first_frame,adj_flag,adjs) = adjustFrame(first_frame, second_frame)
            if adj_flag:
                adj_buffer.extend(adjs)
        i=i+1
    
    for adj in adj_buffer:
        sql = 'UPDATE EG_CLEAN_TMP SET PERSON_NUMBER='+str(adj[1]) +' WHERE ID = '+str(adj[0])
        try:
            # Execute the SQL command
            cursor.execute(sql)
            # Commit your changes in the database
            db.commit()
        except:
            # Rollback in case there is any error
            db.rollback()
    db.close()

def json_pack_from_db(snippets_dir, video_name, frame_width, frame_height, label='unknown', label_index=-1):

    import pymysql as mysql

    db = mysql.connect("127.0.0.1","eguser@192.168.5.19","Fred!*!@!&123","mydb")
    cursor = db.cursor()

    sql='SELECT * FROM EG_CLEAN_TMP WHERE VIDEO_NAME="{}" ORDER BY FRAME_NUMBER asc '.format(video_name)
    df = pd.read_sql(sql, con=db)
    frames = df.groupby('FRAME_NUMBER')
    sequence_info = []
    print(frames)
    for key, value in frames:
        frame_data = {'frame_index': key}
        skeletons = []
        for i in range(0,value.shape[0],1):
            aFrame  = value.iloc[i][1]

            score, coordinates = [], []
            skeleton = {}
            for j in range(5, 59, 3):
                coordinates += [value.iloc[i][j]/frame_width, value.iloc[i][j + 1]/frame_height]
                score += [value.iloc[i][j + 2]]
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index
    print('end of load data from db')
    return video_info

def frames_to_time(frames):
    hours=minutes=seconds=0
    if frames >= 36000:
        hours = int(frames/36000)
        frames = int(frames % 36000)
    if frames >= 600:
        minutes = int(frames/600)
        frames = int(frames % 600)
    if frames >= 10:
        seconds = int(frames/10)
        if (frames % 10) > 4:
            seconds += 1
    ret = '{:0>2}:{:0>2}:{:0>2}'.format(hours,minutes,seconds)

    return ret
    
def action_id (ac_t,action_name):
    try:
        actionId = ac_t.loc[ac_t['action_name'] == action_name]
        actionId = actionId.iloc[0][0]
    except IndexError as ie:
        print(action_name+' is missing in  ref_action_type ')
        print(str(ie))
        raise 
    return actionId

def get_assement_id (video_name):
    import pymysql as mysql
    db = mysql.connect("127.0.0.1", "eguser@192.168.5.19", "Fred!*!@!&123", "ergo_raw")
    sql = 'select assessment_id,video_name from data_video where video_name like "%{}%"'.format(video_name.split('.')[0])
    assid_col = pd.read_sql(sql, con=db)
    assid = assid_col.iloc[0][0]
    return assid

    

def save_to_pro_db(video_name='10048_1'):
    import pymysql as mysql
    import uuid
    assessment_id = get_assement_id(video_name)
    
    print('egdata.save_to_pro_db')
    

    video_name = video_name.split('.')[0]
    save_db = mysql.connect("127.0.0.1", "eguser@192.168.5.19", "Fred!*!@!&123", "ergo_raw")
    cursor = save_db.cursor()

    db = mysql.connect("127.0.0.1", "eguser@192.168.5.19", "Fred!*!@!&123", "mydb")

    sql = 'select LABEL_NAME,FRAME_NUMBER from EG_CLEAN_TMP where VIDEO_NAME ="{}" and PERSON_NUMBER=0 order by FRAME_NUMBER'.format(video_name)
    ac_col = pd.read_sql(sql, con=db)
    print(sql)
    print(ac_col.head(20))


    sql = 'select * from (select LABEL_NAME,count(*) as number from EG_CLEAN_TMP where VIDEO_NAME ="{}" and PERSON_NUMBER=0 and LABEL_NAME is not null group by LABEL_NAME order by count(*) desc ) t where number >50 '.format(video_name)
    ac_g = pd.read_sql(sql, con=db)
    print(sql)
    print(ac_g.head(20))
    

    sql = 'select id,action_name from ref_action_type'
    ac_t = pd.read_sql(sql,save_db)


    save_buffer = []
    sql = ' delete from data_action where assessment_id = "{}" '.format(assessment_id)
    save_buffer.append(sql)

    for action_num in range(0,len(ac_g['number']),1):
        print(ac_g.iloc[action_num][0],ac_g.iloc[action_num][1])
        ac_a = ac_col.loc[ac_col['LABEL_NAME'] == ac_g.iloc[action_num][0]]
        frame_begin = ac_a.iloc[0][1]
        frame_before = frame_begin
        for frame_num in range(1,len(ac_a),1):
            frame_cur=ac_a.iloc[frame_num][1]
            if (frame_cur - frame_before) < 20:
                frame_before = frame_cur
            else:
                print('begin={} end={} count={}'.format(frame_begin, frame_before,frame_before-frame_begin+1))
                #write to db
                if  (frame_before-frame_begin+1) >= 20:
                    sql = 'insert into data_action (action_id,start_time,end_time,total_time,assessment_id,action_key) ' \
                          'values({},"{}","{}","{}",{},"{}")'.format(action_id(ac_t,ac_g.iloc[action_num][0]),frames_to_time(frame_begin),frames_to_time(frame_before),frames_to_time(frame_before-frame_begin+1),assessment_id,uuid.uuid4())
                    save_buffer.append(sql)
                frame_begin = frame_before = frame_cur

        print('begin={} end={} count={}'.format(frame_begin, frame_before, frame_before - frame_begin + 1))
        if (frame_before - frame_begin + 1) >= 20:
            sql = 'insert into data_action (action_id,start_time,end_time,total_time,assessment_id,action_key) ' \
                  'values({},"{}","{}","{}",{},"{}")'.format(action_id(ac_t,ac_g.iloc[action_num][0]), frames_to_time(frame_begin), frames_to_time(frame_before),
                                                frames_to_time(frame_before - frame_begin + 1), assessment_id,uuid.uuid4())
            save_buffer.append(sql)

    for sql in save_buffer:
        try:
            # Execute the SQL command
            cursor.execute(sql)
            # Commit your changes in the database
            save_db.commit()
        except:
            # Rollback in case there is any error
            save_db.rollback()
    save_db.close()
    db.close()
