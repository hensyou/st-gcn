import cv2
import os
import numpy as np
import pymysql
import math
import skvideo.io
import skvideo.datasets
import skvideo.utils
import pandas as pd
import datetime


class AngleCalculation:

    def __init__(self, host, port, user, password, db, table_name, video_name, person_number=0):
        conn = pymysql.connect(host=host, port=port, user=user, passwd=password,
                               db=db)
        cur = conn.cursor()

        cur.execute("SELECT * FROM %s.%s where VIDEO_NAME='%s';" % (db, table_name, video_name))
        num_fields = len(cur.description)
        field_names = [i[0] for i in cur.description]
        self._column = {}
        self._orientation = []
        for h in field_names:
            self._column[h] = []

        for row in cur.fetchall():
            if row[4] == person_number:
                for h, v in zip(field_names, row):
                    if h in ['VIDEO_NAME', 'LABEL_NAME']:
                        v = str(v)
                    else:
                        if v is None:
                            v = 0
                        v = float(v)
                    self._column[h].append(v)
        self._framenumber = len(self._column[field_names[0]])
        self._column['CTR_X'] = []
        self._column['CTR_Y'] = []

        for i in range(self._framenumber):
            if self._column['B_8_X'][i] !=0 and self._column['B_8_Y'][i] !=0:
                self._column['CTR_X'].append((self._column['B_8_X'][i] + self._column['B_11_X'][i]) / 2)
                self._column['CTR_Y'].append((self._column['B_8_Y'][i] + self._column['B_11_Y'][i]) / 2)
            else:
                self._column['CTR_X'].append((self._column['B_2_X'][i] + self._column['B_5_X'][i]) / 2)
                self._column['CTR_Y'].append((self._column['B_2_Y'][i] + self._column['B_5_Y'][i]) / 2)
            ##########################To Manipulate Null Value Point###########################################
            if self._column['B_6_X'][i] == 0:
                self._column['B_6_X'][i] = self._column['B_3_X'][i]
                self._column['B_6_Y'][i] = self._column['B_3_Y'][i]
            if self._column['B_7_X'][i] == 0:
                self._column['B_7_X'][i] = self._column['B_4_X'][i]
                self._column['B_7_Y'][i] = self._column['B_4_Y'][i]
            if self._column['B_3_X'][i] == 0:
                self._column['B_3_X'][i] = self._column['B_6_X'][i]
                self._column['B_3_Y'][i] = self._column['B_6_Y'][i]
            if self._column['B_4_X'][i] == 0:
                self._column['B_4_X'][i] = self._column['B_7_X'][i]
                self._column['B_4_Y'][i] = self._column['B_7_Y'][i]
            if self._column['B_5_X'][i] == 0:
                self._column['B_5_X'][i] = self._column['B_2_X'][i]
                self._column['B_5_Y'][i] = self._column['B_2_Y'][i]

            if self._column['B_2_X'][i] == 0:
                self._column['B_2_X'][i] = self._column['B_5_X'][i]
                self._column['B_2_Y'][i] = self._column['B_5_Y'][i]
            ###################################################################################################
            if self._column['B_0_X'][i] >= self._column['B_2_X'][i] and self._column['B_0_X'][i] < \
                    self._column['B_5_X'][i]:
                self._orientation.append('front')
            elif self._column['B_0_X'][i] <= self._column['B_2_X'][i] and self._column['B_0_X'][i] > \
                    self._column['B_5_X'][i]:
                self._orientation.append('back')
            elif self._column['B_0_X'][i] >= self._column['B_2_X'][i] and self._column['B_0_X'][i] >= \
                    self._column['B_5_X'][i]:
                self._orientation.append('right')
            else:
                self._orientation.append('left')


        print(self._column['PERSON_NUMBER'])
        print('Read Successfully!')
        print(self._framenumber)
        print(self._column['LABEL_NAME'])

    def ShowOrientation(self):
        orientation = self._orientation
        return orientation
###############################################Video Capture###########################################################
    def VideoCap(self, videopath, writeimagepath):

        vid = skvideo.io.vread(fname=videopath)
        for i in range(len(vid)):
            skvideo.io.vwrite("%s/frame%05d.jpg" % (writeimagepath, i), vid[i])
            print("read successfully!")

    def Getfps(self,videopath):
        vidcap = cv2.VideoCapture(videopath)
        framerate=round(vidcap.get(cv2.CAP_PROP_FPS),0)

        return framerate

    def VideoCapbycv(self, videopath, writeimagepath):

        vidcap = cv2.VideoCapture(videopath)

        framediff = int(round(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / self._framenumber, 0))

        success, image = vidcap.read()
        count = 0
        count2 = 0
        if framediff == 0:
            while (success):
                cv2.imwrite("%s/frame%05d.jpg" % (writeimagepath, count), image)
                success, image = vidcap.read()

                print('Read a new frame: ', success)
                count += 1
        else:
            while (success):
                if count % framediff != 0:
                    vidcap.grab()
                    count += 1
                else:
                    cv2.imwrite("%s/frame%05d.jpg" % (writeimagepath, count2), image)  # save frame as JPEG file

                    success, image = vidcap.read()

                    print('Read a new frame: ', success)
                    count += 1
                    count2 += 1

################################################Angle Calculation######################################################
    def CalNeckAngle(self, framenum=-1):

        NeckAngle = []
        if framenum == -1:
            i = 0
            while i < self._framenumber:

                a = np.array((self._column['CTR_X'][i], 0))
                b = np.array((self._column['B_1_X'][i], self._column['B_1_Y'][i]))
                if self._orientation[i] == 'left':
                    c = np.array((self._column['B_17_X'][i], self._column['B_17_Y'][i]))  # use left ear
                elif self._orientation[i] == 'right':
                    c = np.array((self._column['B_16_X'][i], self._column['B_16_Y'][i]))  # use right ear
                else:
                    c = np.array((self._column['B_0_X'][i], self._column['B_0_Y'][i]))  # use nose

                ba = a - b
                bc = c - b

                if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    neckangle = np.degrees(np.arccos(cosine_angle))
                    if self._orientation[i] == 'front' or self._orientation[i] == 'back':
                        if neckangle >= 20:
                            neckangle = neckangle/180 + 19
                    NeckAngle.append(neckangle)
                else:
                    NeckAngle.append(0)

                i = i + 1
        else:
            a = np.array((self._column['CTR_X'][framenum], 0))
            b = np.array((self._column['B_1_X'][framenum], self._column['B_1_Y'][framenum]))
            if self._orientation[framenum] == 'left':
                c = np.array((self._column['B_17_X'][framenum], self._column['B_17_Y'][framenum]))
            elif self._orientation[framenum] == 'right':
                c = np.array((self._column['B_16_X'][framenum], self._column['B_16_Y'][framenum]))
            else:
                c = np.array((self._column['B_0_X'][framenum], self._column['B_0_Y'][framenum]))

            ba = a - b
            bc = c - b

            if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                neckangle = np.degrees(np.arccos(cosine_angle))
                if self._orientation[framenum]=='front' or self._orientation[framenum]=='back':
                    if neckangle >= 20:
                            neckangle = neckangle/180 + 19
                NeckAngle.append(neckangle)
            else:
                NeckAngle.append(0)

            NeckAngle = NeckAngle[0]

        return NeckAngle

    def CalLeftShoulderAngle(self, framenum=-1):
        LShoulderAngle = []
        if framenum == -1:
            i = 0
            while i < self._framenumber:
                a = np.array((self._column['B_6_X'][i], self._column['B_6_Y'][i]))
                b = np.array((self._column['B_5_X'][i], self._column['B_5_Y'][i]))
                c = np.array((self._column['B_5_X'][i], self._column['B_5_Y'][i]+100))

                ba = a - b
                bc = c - b

                if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    leftshoulderangle = np.degrees(np.arccos(cosine_angle))
                    LShoulderAngle.append(leftshoulderangle)
                else:
                    LShoulderAngle.append(0)

                i = i + 1
        else:
            a = np.array((self._column['B_6_X'][framenum], self._column['B_6_Y'][framenum]))
            b = np.array((self._column['B_5_X'][framenum], self._column['B_5_Y'][framenum]))
            c = np.array((self._column['B_5_X'][framenum], self._column['B_5_Y'][framenum]+100))

            ba = a - b
            bc = c - b

            if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                leftshoulderangle = np.degrees(np.arccos(cosine_angle))
                LShoulderAngle.append(leftshoulderangle)
            else:
                LShoulderAngle.append(0)
            LShoulderAngle = LShoulderAngle[0]

        return LShoulderAngle

    def CalRightShoulderAngle(self, framenum=-1):
        RShoulderAngle = []
        if framenum == -1:
            i = 0
            while i < self._framenumber:
                a = np.array((self._column['B_3_X'][i], self._column['B_3_Y'][i]))
                b = np.array((self._column['B_2_X'][i], self._column['B_2_Y'][i]))
                c = np.array((self._column['B_3_X'][i], self._column['B_3_Y'][i]+100))

                ba = a - b
                bc = c - b

                if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    rightshoulderangle = np.degrees(np.arccos(cosine_angle))
                    RShoulderAngle.append(rightshoulderangle)
                else:
                    RShoulderAngle.append(0)

                i = i + 1
        else:
            a = np.array((self._column['B_3_X'][framenum], self._column['B_3_Y'][framenum]))
            b = np.array((self._column['B_2_X'][framenum], self._column['B_2_Y'][framenum]))
            c = np.array((self._column['B_3_X'][framenum], self._column['B_3_Y'][framenum]+100))

            ba = a - b
            bc = c - b

            if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                rightshoulderangle = np.degrees(np.arccos(cosine_angle))
                RShoulderAngle.append(rightshoulderangle)
            else:
                RShoulderAngle.append(0)
            RShoulderAngle = RShoulderAngle[0]

        return RShoulderAngle

    def CalLeftElbow(self, framenum=-1):
        LElbowAngle = []
        if framenum == -1:
            i = 0
            while i < self._framenumber:
                a = np.array((self._column['B_7_X'][i], self._column['B_7_Y'][i]))
                b = np.array((self._column['B_6_X'][i], self._column['B_6_Y'][i]))
                c = np.array((self._column['B_5_X'][i], self._column['B_5_Y'][i]))

                ba = a - b
                bc = c - b

                if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    leftelbowangle = 180 - np.degrees(np.arccos(cosine_angle))
                    LElbowAngle.append(leftelbowangle)
                else:
                    LElbowAngle.append(0)

                i = i + 1
        else:
            a = np.array((self._column['B_7_X'][framenum], self._column['B_7_Y'][framenum]))
            b = np.array((self._column['B_6_X'][framenum], self._column['B_6_Y'][framenum]))
            c = np.array((self._column['B_5_X'][framenum], self._column['B_5_Y'][framenum]))

            ba = a - b
            bc = c - b

            if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                leftelbowangle = 180 - np.degrees(np.arccos(cosine_angle))
                LElbowAngle.append(leftelbowangle)
            else:
                LElbowAngle.append(0)

            LElbowAngle = LElbowAngle[0]

        return LElbowAngle

    def CalRightElbow(self, framenum=-1):
        RElbowAngle = []
        if framenum == -1:
            i = 0
            while i < self._framenumber:
                a = np.array((self._column['B_4_X'][i], self._column['B_4_Y'][i]))
                b = np.array((self._column['B_3_X'][i], self._column['B_3_Y'][i]))
                c = np.array((self._column['B_2_X'][i], self._column['B_2_Y'][i]))

                ba = a - b
                bc = c - b

                if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    rightelbowangle = 180 - np.degrees(np.arccos(cosine_angle))
                    RElbowAngle.append(rightelbowangle)
                else:
                    RElbowAngle.append(0)

                i = i + 1
        else:
            a = np.array((self._column['B_4_X'][framenum], self._column['B_4_Y'][framenum]))
            b = np.array((self._column['B_3_X'][framenum], self._column['B_3_Y'][framenum]))
            c = np.array((self._column['B_2_X'][framenum], self._column['B_2_Y'][framenum]))

            ba = a - b
            bc = c - b

            if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                rightelbowangle = 180 - np.degrees(np.arccos(cosine_angle))
                RElbowAngle.append(rightelbowangle)
            else:
                RElbowAngle.append(0)

            RElbowAngle = RElbowAngle[0]

        return RElbowAngle

    def CalBackAngle(self, framenum=-1):
        BackAngle = []
        if framenum == -1:
            i = 0
            while i < self._framenumber:

                a = np.array((self._column['CTR_X'][i], self._column['CTR_Y'][i] - 100))
                b = np.array((self._column['CTR_X'][i], self._column['CTR_Y'][i]))
                c = np.array((self._column['B_1_X'][i], self._column['B_1_Y'][i]))

                ba = a - b
                bc = c - b

                if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    backangle = np.degrees(np.arccos(cosine_angle))
                    if backangle >= 90:
                        backangle = 45 + backangle/10
                    BackAngle.append(backangle)
                else:
                    BackAngle.append(0)

                i = i + 1
        else:
            hipcentre = ((self._column['B_8_X'][framenum] + self._column['B_11_X'][framenum]) / 2,
                         (self._column['B_8_Y'][framenum] + self._column['B_11_Y'][framenum]) / 2)
            a = np.array((hipcentre[0], hipcentre[1] - 100))
            b = np.array((hipcentre[0], hipcentre[1]))
            c = np.array((self._column['B_1_X'][framenum], self._column['B_1_Y'][framenum]))

            ba = a - b
            bc = c - b

            if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                backangle = np.degrees(np.arccos(cosine_angle))
                if backangle >= 90:
                    backangle = 45 + backangle/10
                BackAngle.append(backangle)
            else:
                BackAngle.append(0)

            BackAngle = BackAngle[0]

        return BackAngle

    def CalRightLeg(self,framenum=-1):
        Rightleg= []
        if framenum == -1:
            i = 0
            while i < self._framenumber:

                a = np.array((self._column['B_8_X'][i], self._column['B_8_Y'][i]))
                b = np.array((self._column['B_9_X'][i], self._column['B_9_Y'][i]))
                c = np.array((self._column['B_10_X'][i], self._column['B_10_Y'][i]))

                ba = a - b
                bc = c - b

                if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    rightlegangle = np.degrees(np.arccos(cosine_angle))
                    Rightleg.append(rightlegangle)
                else:
                    Rightleg.append(0)

                i = i + 1
        else:
            a = np.array((self._column['B_8_X'][framenum], self._column['B_8_Y'][framenum]))
            b = np.array((self._column['B_9_X'][framenum], self._column['B_9_Y'][framenum]))
            c = np.array((self._column['B_10_X'][framenum], self._column['B_10_Y'][framenum]))

            ba = a - b
            bc = c - b

            if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                rightlegangle = np.degrees(np.arccos(cosine_angle))
                Rightleg.append(rightlegangle)
            else:
                Rightleg.append(0)

            Rightleg = Rightleg[0]

        return Rightleg

    def CalLeftLeg(self,framenum=-1):
        Leftleg= []
        if framenum == -1:
            i = 0
            while i < self._framenumber:

                a = np.array((self._column['B_11_X'][i], self._column['B_11_Y'][i]))
                b = np.array((self._column['B_12_X'][i], self._column['B_12_Y'][i]))
                c = np.array((self._column['B_13_X'][i], self._column['B_13_Y'][i]))

                ba = a - b
                bc = c - b

                if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    leftlegangle = np.degrees(np.arccos(cosine_angle))
                    Leftleg.append(leftlegangle)
                else:
                    Leftleg.append(0)

                i = i + 1
        else:
            a = np.array((self._column['B_11_X'][framenum], self._column['B_11_Y'][framenum]))
            b = np.array((self._column['B_12_X'][framenum], self._column['B_12_Y'][framenum]))
            c = np.array((self._column['B_13_X'][framenum], self._column['B_13_Y'][framenum]))

            ba = a - b
            bc = c - b

            if int((np.linalg.norm(ba) * np.linalg.norm(bc))) != 0:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                leftlegangle = np.degrees(np.arccos(cosine_angle))
                Leftleg.append(leftlegangle)
            else:
                Leftleg.append(0)

            Leftleg = Leftleg[0]

        return Leftleg

    ################################calculate Noish parameters########################################################
    def Calratio(self, framenum=-1, height=180):
        operator = {}
        if framenum == -1:

            realratio=[]

            i = 0
            while i < self._framenumber:

                pose0 = (int(self._column['B_0_X'][i]), int(self._column['B_0_Y'][i]))
                pose1 = (int(self._column['B_1_X'][i]), int(self._column['B_1_Y'][i]))
                pose2 = (int(self._column['B_2_X'][i]), int(self._column['B_2_Y'][i]))
                pose3 = (int(self._column['B_3_X'][i]), int(self._column['B_3_Y'][i]))
                pose4 = (int(self._column['B_4_X'][i]), int(self._column['B_4_Y'][i]))
                pose5 = (int(self._column['B_5_X'][i]), int(self._column['B_5_Y'][i]))
                pose6 = (int(self._column['B_6_X'][i]), int(self._column['B_6_Y'][i]))
                pose7 = (int(self._column['B_7_X'][i]), int(self._column['B_7_Y'][i]))
                pose8 = (int(self._column['B_8_X'][i]), int(self._column['B_8_Y'][i]))
                pose9 = (int(self._column['B_9_X'][i]), int(self._column['B_9_Y'][i]))
                pose10 = (int(self._column['B_10_X'][i]), int(self._column['B_10_Y'][i]))
                pose11 = (int(self._column['B_11_X'][i]), int(self._column['B_11_Y'][i]))
                pose12 = (int(self._column['B_12_X'][i]), int(self._column['B_12_Y'][i]))
                pose13 = (int(self._column['B_13_X'][i]), int(self._column['B_13_Y'][i]))
                pose14 = (int(self._column['B_14_X'][i]), int(self._column['B_14_Y'][i]))
                pose15 = (int(self._column['B_15_X'][i]), int(self._column['B_15_Y'][i]))
                pose16 = (int(self._column['B_16_X'][i]), int(self._column['B_16_Y'][i]))
                pose17 = (int(self._column['B_17_X'][i]), int(self._column['B_17_Y'][i]))

                if pose10[0] * pose9[0] + pose10[1] * pose9[1] == 0:
                    foot = np.linalg.norm(np.array(pose13) - np.array(pose12))
                elif pose13[0] * pose12[0] + pose13[1] * pose12[1] == 0:
                    foot = np.linalg.norm(np.array(pose10) - np.array(pose9))
                else:
                    foot = max(np.linalg.norm(np.array(pose10) - np.array(pose9)),
                               np.linalg.norm(np.array(pose13) - np.array(pose12)))

                if pose8[0] * pose9[0] + pose8[1] * pose9[1] == 0:
                    leg = np.linalg.norm(np.array(pose11) - np.array(pose12))
                elif pose11[0] * pose12[0] + pose11[1] * pose12[1] == 0:
                    leg = np.linalg.norm(np.array(pose8) - np.array(pose9))
                else:
                    leg = max(np.linalg.norm(np.array(pose8) - np.array(pose9)),
                              np.linalg.norm(np.array(pose11) - np.array(pose12)))

                if pose4[0] * pose3[0] + pose4[1] * pose3[1] == 0:
                    hand = np.linalg.norm(np.array(pose7) - np.array(pose6))
                elif pose7[0] * pose6[0] + pose7[1] * pose6[1] == 0:
                    hand = np.linalg.norm(np.array(pose4) - np.array(pose3))
                else:
                    hand = max(np.linalg.norm(np.array(pose4) - np.array(pose3)),
                               np.linalg.norm(np.array(pose7) - np.array(pose6)))

                if pose2[0] * pose3[0] + pose2[1] * pose3[1] == 0:
                    arm = np.linalg.norm(np.array(pose5) - np.array(pose6))
                elif pose5[0] * pose6[0] + pose5[1] * pose6[1] == 0:
                    arm = np.linalg.norm(np.array(pose2) - np.array(pose3))
                else:
                    arm = max(np.linalg.norm(np.array(pose2) - np.array(pose3)),
                              np.linalg.norm(np.array(pose5) - np.array(pose6)))

                bodylength = np.linalg.norm(
                    np.array(pose1) - np.array((self._column['CTR_X'][i], self._column['CTR_Y'][i])))

                pixelheight = leg + foot + bodylength + arm
                if pixelheight != 0:
                    ratio = height / pixelheight

                else:
                    ratio=0

                realratio.append(ratio)

                i = i + 1
        else:
            pose0 = (int(self._column['B_0_X'][framenum]), int(self._column['B_0_Y'][framenum]))
            pose1 = (int(self._column['B_1_X'][framenum]), int(self._column['B_1_Y'][framenum]))
            pose2 = (int(self._column['B_2_X'][framenum]), int(self._column['B_2_Y'][framenum]))
            pose3 = (int(self._column['B_3_X'][framenum]), int(self._column['B_3_Y'][framenum]))
            pose4 = (int(self._column['B_4_X'][framenum]), int(self._column['B_4_Y'][framenum]))
            pose5 = (int(self._column['B_5_X'][framenum]), int(self._column['B_5_Y'][framenum]))
            pose6 = (int(self._column['B_6_X'][framenum]), int(self._column['B_6_Y'][framenum]))
            pose7 = (int(self._column['B_7_X'][framenum]), int(self._column['B_7_Y'][framenum]))
            pose8 = (int(self._column['B_8_X'][framenum]), int(self._column['B_8_Y'][framenum]))
            pose9 = (int(self._column['B_9_X'][framenum]), int(self._column['B_9_Y'][framenum]))
            pose10 = (int(self._column['B_10_X'][framenum]), int(self._column['B_10_Y'][framenum]))
            pose11 = (int(self._column['B_11_X'][framenum]), int(self._column['B_11_Y'][framenum]))
            pose12 = (int(self._column['B_12_X'][framenum]), int(self._column['B_12_Y'][framenum]))
            pose13 = (int(self._column['B_13_X'][framenum]), int(self._column['B_13_Y'][framenum]))
            pose14 = (int(self._column['B_14_X'][framenum]), int(self._column['B_14_Y'][framenum]))
            pose15 = (int(self._column['B_15_X'][framenum]), int(self._column['B_15_Y'][framenum]))
            pose16 = (int(self._column['B_16_X'][framenum]), int(self._column['B_16_Y'][framenum]))
            pose17 = (int(self._column['B_17_X'][framenum]), int(self._column['B_17_Y'][framenum]))

            if pose10[0] * pose9[0] + pose10[1] * pose9[1] == 0:
                foot = np.linalg.norm(np.array(pose13) - np.array(pose12))
            elif pose13[0] * pose12[0] + pose13[1] * pose12[1] == 0:
                foot = np.linalg.norm(np.array(pose10) - np.array(pose9))
            else:
                foot = max(np.linalg.norm(np.array(pose10) - np.array(pose9)),
                           np.linalg.norm(np.array(pose13) - np.array(pose12)))

            if pose8[0] * pose9[0] + pose8[1] * pose9[1] == 0:
                leg = np.linalg.norm(np.array(pose11) - np.array(pose12))
            elif pose11[0] * pose12[0] + pose11[1] * pose12[1] == 0:
                leg = np.linalg.norm(np.array(pose8) - np.array(pose9))
            else:
                leg = max(np.linalg.norm(np.array(pose8) - np.array(pose9)),
                          np.linalg.norm(np.array(pose11) - np.array(pose12)))

            if pose4[0] * pose3[0] + pose4[1] * pose3[1] == 0:
                hand = np.linalg.norm(np.array(pose7) - np.array(pose6))
            elif pose7[0] * pose6[0] + pose7[1] * pose6[1] == 0:
                hand = np.linalg.norm(np.array(pose4) - np.array(pose3))
            else:
                hand = max(np.linalg.norm(np.array(pose4) - np.array(pose3)),
                           np.linalg.norm(np.array(pose7) - np.array(pose6)))

            if pose2[0] * pose3[0] + pose2[1] * pose3[1] == 0:
                arm = np.linalg.norm(np.array(pose5) - np.array(pose6))
            elif pose5[0] * pose6[0] + pose5[1] * pose6[1] == 0:
                arm = np.linalg.norm(np.array(pose2) - np.array(pose3))
            else:
                arm = max(np.linalg.norm(np.array(pose2) - np.array(pose3)),
                          np.linalg.norm(np.array(pose5) - np.array(pose6)))

            bodylength = np.linalg.norm(
                np.array(pose1) - np.array((self._column['CTR_X'][framenum], self._column['CTR_Y'][framenum])))

            pixelheight = leg + foot + bodylength + arm
            if pixelheight != 0:
                ratio = height / pixelheight

            else:
                ratio = 0

            realratio=ratio

        return realratio

    def Calhorizontaldistance(self,framenum=-1):
        horizontaldis = []
        if framenum == -1:
            i = 0
            while i < self._framenumber:
                if self._column['B_10_X'][i]==0 or self._column['B_13_X'][i]==0:
                    a=self._column['CTR_X'][i]
                else:
                    a=(self._column['B_10_X'][i] + self._column['B_13_X'][i]) / 2

                if self._column['B_4_X'][i]==0 and self._column['B_7_X'][i]==0:
                    horizontaldis.append(0)
                else:
                    b1=self._column['B_4_X'][i]
                    b2=self._column['B_7_X'][i]

                    imagehorizontaldis=max(abs(a-b1),abs(a-b2))
                    ratio=self.Calratio(framenum=i)
                    realhorizontaldis=imagehorizontaldis*ratio/2.54 #inch
                    horizontaldis.append(realhorizontaldis)

                i=i+1
        else:
            if self._column['B_10_X'][framenum] == 0 or self._column['B_13_X'][framenum] == 0:
                a = self._column['CTR_X'][framenum]
            else:
                a = (self._column['B_10_X'][framenum] + self._column['B_13_X'][framenum]) / 2

            if self._column['B_4_X'][framenum] == 0 and self._column['B_7_X'][framenum] == 0:
                horizontaldis=0
            else:
                b1 = self._column['B_4_X'][framenum]
                b2 = self._column['B_7_X'][framenum]

                imagehorizontaldis = max(abs(a - b1), abs(a - b2))
                ratio = self.Calratio(framenum=framenum)
                realhorizontaldis = imagehorizontaldis * ratio/2.54 #inch
                horizontaldis=realhorizontaldis

        return horizontaldis


    def Calverticaldistance(self,framenum=-1):
        verticaldis=[]
        if framenum == -1:
            i = 0
            while i < self._framenumber:
                if self._column['B_10_Y'][i] == 0 and self._column['B_13_Y'][i] == 0:
                    a = 0
                elif self._column['B_10_Y'][i] >self._column['B_13_Y'][i]:
                    a=  self._column['B_10_Y'][i]
                else:
                    a = self._column['B_13_Y'][i]

                if self._column['B_4_Y'][i] == 0 and self._column['B_7_Y'][i] == 0:
                    verticaldis.append(0)
                elif a==0:
                    verticaldis.append(0)
                else:
                    b1 = self._column['B_4_Y'][i]
                    b2 = self._column['B_7_Y'][i]
                    if b1==0:
                        imageverticaldis=abs(a-b2)
                    elif b2==0:
                        imageverticaldis=abs(a-b1)
                    else:
                        imageverticaldis = max(abs(a - b1), abs(a - b2))
                    ratio = self.Calratio(framenum=i)
                    realverticaldis = imageverticaldis * ratio/2.54 #inch
                    verticaldis.append(realverticaldis)

                i = i + 1
        else:
            if self._column['B_10_Y'][framenum] == 0 and self._column['B_13_Y'][framenum] == 0:
                a = 0
            elif self._column['B_10_Y'][framenum] > self._column['B_13_Y'][framenum]:
                a = self._column['B_10_Y'][framenum]
            else:
                a = self._column['B_13_Y'][framenum]

            if self._column['B_4_Y'][framenum] == 0 and self._column['B_7_Y'][framenum] == 0:
                verticaldis=0
            elif a == 0:
                verticaldis=0
            else:
                b1 = self._column['B_4_Y'][framenum]
                b2 = self._column['B_7_Y'][framenum]
                if b1 == 0:
                    imageverticaldis = abs(a - b2)
                elif b2 == 0:
                    imageverticaldis = abs(a - b1)
                else:
                    imageverticaldis = max(abs(a - b1), abs(a - b2))
                ratio = self.Calratio(framenum=framenum)
                realverticaldis = imageverticaldis * ratio/2.54 #inch
                verticaldis=realverticaldis


        return verticaldis

    def drawhorizontaldist(self,framenum=-1, position=(20,290),color=(255, 0, 0), thickness=2):
        if framenum==-1 and self._column['image']!=[]:
            i=0
            while i<self._framenumber:
                img=self._column['image'][i]
                horizontaldis=self.Calhorizontaldistance(framenum=i)

                self._column['image'][i] = cv2.putText(img, 'Horizontal Dist:' + str(round(horizontaldis, 2))+'inches', position,
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness)

                i=i+1

        return self._column['image']


    def drawverticaldist(self, framenum=-1, position=(20, 320),color = (255, 0, 0), thickness = 2):
        if framenum == -1 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                verticaldis = self.Calverticaldistance(framenum=i)
                self._column['image'][i] = cv2.putText(img, 'Vertical Dist:' + str(round(verticaldis, 2)) + 'inches',
                                                       position,
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness)

                i = i + 1

        return self._column['image']





    ################################calculate posture time###########################################################
    def calNecktime(self):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        Neck = {}

        for i in range(self._framenumber):
            angle = self.CalNeckAngle(framenum=i)
            if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                if angle <= 20:
                    count1 = count1 + 1
                elif angle <= 29:
                    count2 = count2 + 1
                elif angle <= 44:
                    count3 = count3 + 1
                else:
                    count4 = count4 + 1
            else:
                if angle <= 10:
                    count5 = count5 + 1
                elif angle <= 19:
                    count6 = count6 + 1
                else:
                    count7 = count7 + 1

        Neck['75'] = [datetime.timedelta(seconds=round((count1 / self._framenumber) * (self._framenumber / 10)))]
        Neck['76'] = [datetime.timedelta(seconds=round((count2 / self._framenumber) * (self._framenumber / 10)))]
        Neck['77'] = [datetime.timedelta(seconds=round((count3 / self._framenumber) * (self._framenumber / 10)))]
        Neck['78'] = [datetime.timedelta(seconds=round((count4 / self._framenumber) * (self._framenumber / 10)))]
        Neck['85'] = [datetime.timedelta(seconds=round((count5 / self._framenumber) * (self._framenumber / 10)))]
        Neck['86'] = [datetime.timedelta(seconds=round((count6 / self._framenumber) * (self._framenumber / 10)))]
        Neck['87'] = [datetime.timedelta(seconds=round((count7 / self._framenumber) * (self._framenumber / 10)))]

        x = []
        x = self.CalNeckAngle()
        orientation = self._orientation
        y = {}
        count = 0
        a = []
        b = []
        c = []
        d = []
        e = []
        f = []
        g = []
        yy={}



        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            nestedori = orientation[i:i + 10]
            front = 0
            side = 0
            j = 0
            total = 0
            #print(z)
            while j < len(z):
                #print(j)
                #print(z[j])

                if z[j] == 0:
                    del z[j]
                    del nestedori[j]
                else:
                    total = total + z[j]
                    if nestedori[j] == 'left' or nestedori[j] == 'right':
                        side = side + 1
                    else:
                        front = front + 1
                    j=j+1
            if j != 0:
                y[count] = [total / j]
                if side >= front:
                    y[count].append(0)
                    # 0 stands for flexion
                else:
                    y[count].append(1)
                    # 1 stands for abduction or side bend
                count = count + 1

        for key, value in y.items():
            if value[1] == 0:
                if value[0] <= 20:
                    a.append(key)
                elif value[0] <= 29:
                    b.append(key)
                elif value[0] <= 44:
                    c.append(key)
                else:
                    d.append(key)
            else:
                if value[0] <= 10:
                    e.append(key)
                elif value[0] < 20:
                    f.append(key)
                else:
                    g.append(key)
        print('y:', y)
        print(a)
        print(b)
        print(c)
        print(d)
        print(e)
        print(f)
        print(g)


        repetition1 = 0
        repetition2 = 0
        repetition3 = 0
        repetition4 = 0
        repetition5 = 0
        repetition6 = 0
        repetition7 = 0
        static1 = 0
        static2 = 0
        static3 = 0
        static4 = 0
        static5 = 0
        static6 = 0
        static7 = 0



        if len(a) > 1:
            for i in range(0, len(a)-1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1


        if len(b) > 1:
            for i in range(0, len(b)-1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c)-1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        if len(d) > 1:
            for i in range(0, len(d)-1):
                if abs(d[i] - d[i + 1]) > 1:
                    repetition4 = repetition4 + 1

###########################################Cal Static################################################

        if len(a) > 3:
            i = 3
            flag = 0
            while i < len(a):
                sumdiff = 0
                sumdiff = abs(a[i] - a[i - 1]) + abs(a[i - 1] - a[i - 2]) + abs(a[i - 2] - a[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(a[i] - a[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static1 = static1 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2=0 # flag2=1 at the end if the difference of all 4 elements are less than 2 degree to add 4s on static
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static1 = static1 + 4
                        i = i + 1
                        flag = 1

        if len(b) > 3:
            i = 3
            flag = 0
            while i < len(b):
                sumdiff = 0
                sumdiff = abs(b[i] - b[i - 1]) + abs(b[i - 1] - b[i - 2]) + abs(b[i - 2] - b[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(b[i] - b[i-1]) == 1:
                        if abs(y[i][0]-y[i-1][0])<=2:
                            static2=static2+1
                            i=i + 1
                        else:
                            flag=0
                            i=i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2=0
                    for g in range(i-2,i+1):
                        sumdiff2=abs(y[g][0]-y[g-1][0])
                        if sumdiff2 > 2:
                            flag2=0
                            break
                        else:
                            flag2=1
                    if flag2==0:
                        i=i+1
                    else:
                        static2=static2+4
                        i=i+1
                        flag=1

        if len(c) > 3:
            i = 3
            flag = 0
            while i < len(c):
                sumdiff = 0
                sumdiff = abs(c[i] - c[i - 1]) + abs(c[i - 1] - c[i - 2]) + abs(c[i - 2] - c[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(c[i] - c[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static3 = static3 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2=0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static3 = static3 + 4
                        i = i + 1
                        flag = 1



        if len(d) > 3:
            i = 3
            flag = 0
            while i < len(d):
                sumdiff = 0
                sumdiff = abs(d[i] - d[i - 1]) + abs(d[i - 1] - d[i - 2]) + abs(d[i - 2] - d[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(d[i] - d[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static4 = static4 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2=0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static4 = static4 + 4
                        i = i + 1
                        flag = 1

        Neck['75'].append(repetition1)
        Neck['76'].append(repetition2)
        Neck['77'].append(repetition3)
        Neck['78'].append(repetition4)
        Neck['85'].append(repetition5)
        Neck['86'].append(repetition6)
        Neck['87'].append(repetition7)

        Neck['75'].append(static1)
        Neck['76'].append(static2)
        Neck['77'].append(static3)
        Neck['78'].append(static4)
        Neck['85'].append(static5)
        Neck['86'].append(static6)
        Neck['87'].append(static7)

        return Neck

    def calREtime(self):
        count1 = 0
        count2 = 0
        count3 = 0

        ElbowRight = {}

        for i in range(self._framenumber):
            angle = self.CalRightElbow(framenum=i)
            if angle < 60:
                count1 = count1 + 1
            elif angle <= 100:
                count2 = count2 + 1
            else:
                count3 = count3 + 1

        ElbowRight['36'] = [datetime.timedelta(seconds=round((count1 / self._framenumber) * (self._framenumber / 10)))]
        ElbowRight['37'] = [datetime.timedelta(seconds=round((count2 / self._framenumber) * (self._framenumber / 10)))]
        ElbowRight['38'] = [datetime.timedelta(seconds=round((count3 / self._framenumber) * (self._framenumber / 10)))]

        x = []
        x = self.CalRightElbow()
        y = {}
        count = 0
        a = []
        b = []
        c = []

        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            j = 0
            total = 0

            while j < len(z):
                if z[j] == 0:
                    del z[j]
                else:
                    total = total + z[j]
                    j = j + 1
            if j != 0:
                y[count] = total / j
                count = count + 1
        for key, value in y.items():
            if value < 60:
                a.append(key)
            elif value <= 100:
                b.append(key)
            else:
                c.append(key)
        print(y)
        repetition1 = 0
        repetition2 = 0
        repetition3 = 0
        static1=0
        static2=0
        static3=0

        if len(a) > 1:
            for i in range(0, len(a) - 1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1

        if len(b) > 1:
            for i in range(0, len(b) - 1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c) - 1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        #############################RIGHT ELBOW STATIC##############################################

        if len(a) > 3:
            i = 3
            flag = 0
            while i < len(a):
                sumdiff = 0
                sumdiff = abs(a[i] - a[i - 1]) + abs(a[i - 1] - a[i - 2]) + abs(a[i - 2] - a[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(a[i] - a[i - 1]) == 1:
                        if abs(y[i] - y[i - 1]) <= 2: #note that y is only 1 dimensional dict for elbows, may be change later
                            static1 = static1 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2=0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g] - y[g - 1])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static1 = static1 + 4
                        i = i + 1
                        flag = 1

        if len(b) > 3:
            i = 3
            flag = 0
            while i < len(b):
                sumdiff = 0
                sumdiff = abs(b[i] - b[i - 1]) + abs(b[i - 1] - b[i - 2]) + abs(b[i - 2] - b[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(b[i] - b[i - 1]) == 1:
                        if abs(y[i] - y[i - 1]) <= 2:
                            static2 = static2 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g] - y[g - 1])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static2 = static2 + 4
                        i = i + 1
                        flag = 1


        if len(c) > 3:
            i = 3
            flag = 0
            while i < len(c):
                sumdiff = 0
                sumdiff = abs(c[i] - c[i - 1]) + abs(c[i - 1] - c[i - 2]) + abs(c[i - 2] - c[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(c[i] - c[i - 1]) == 1:
                        if abs(y[i] - y[i - 1]) <= 2:
                            static3 = static3 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2=0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g] - y[g - 1])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static3 = static3 + 4
                        i = i + 1
                        flag = 1

        ElbowRight['36'].append(repetition1)
        ElbowRight['37'].append(repetition2)
        ElbowRight['38'].append(repetition3)

        ElbowRight['36'].append(static1)
        ElbowRight['37'].append(static2)
        ElbowRight['38'].append(static3)

        return ElbowRight

    def calLEtime(self):
        count1 = 0
        count2 = 0
        count3 = 0

        ElbowLeft = {}

        for i in range(self._framenumber):
            angle = self.CalLeftElbow(framenum=i)
            if angle < 60:
                count1 = count1 + 1
            elif angle <= 100:
                count2 = count2 + 1
            else:
                count3 = count3 + 1

        ElbowLeft['33'] = [datetime.timedelta(seconds=round((count1 / self._framenumber) * (self._framenumber / 10)))]
        ElbowLeft['34'] = [datetime.timedelta(seconds=round((count2 / self._framenumber) * (self._framenumber / 10)))]
        ElbowLeft['35'] = [datetime.timedelta(seconds=round((count3 / self._framenumber) * (self._framenumber / 10)))]

        x = []
        x = self.CalLeftElbow()
        y = {}
        count = 0
        a = []
        b = []
        c = []

        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            j = 0
            total = 0

            while j < len(z):
                if z[j] == 0:
                    del z[j]
                else:
                    total = total + z[j]
                    j = j + 1
            if j != 0:
                y[count] = total / j
                count = count + 1
        for key, value in y.items():
            if value < 60:
                a.append(key)
            elif value <= 100:
                b.append(key)
            else:
                c.append(key)

        repetition1 = 0
        repetition2 = 0
        repetition3 = 0

        static1 = 0
        static2 = 0
        static3 = 0



        if len(a) > 1:
            for i in range(0, len(a) - 1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1

        if len(b) > 1:
            for i in range(0, len(b) - 1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c) - 1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        #########################Left ELBOW STATIC#######################################

        if len(a) > 3:
            i = 3
            flag = 0
            while i < len(a):
                sumdiff = 0
                sumdiff = abs(a[i] - a[i - 1]) + abs(a[i - 1] - a[i - 2]) + abs(a[i - 2] - a[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(a[i] - a[i - 1]) == 1:
                        if abs(y[i] - y[i - 1]) <= 2:
                            static1 = static1 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2=0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g] - y[g - 1])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static1 = static1 + 4
                        i = i + 1
                        flag = 1

        if len(b) > 3:
            i = 3
            flag = 0
            while i < len(b):
                sumdiff = 0
                sumdiff = abs(b[i] - b[i - 1]) + abs(b[i - 1] - b[i - 2]) + abs(b[i - 2] - b[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(b[i] - b[i - 1]) == 1:
                        if abs(y[i] - y[i - 1]) <= 2:
                            static2 = static2 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g] - y[g - 1])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static2 = static2 + 4
                        i = i + 1
                        flag = 1


        if len(c) > 3:
            i = 3
            flag = 0
            while i < len(c):
                sumdiff = 0
                sumdiff = abs(c[i] - c[i - 1]) + abs(c[i - 1] - c[i - 2]) + abs(c[i - 2] - c[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(c[i] - c[i - 1]) == 1:
                        if abs(y[i] - y[i - 1]) <= 2:
                            static3 = static3 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2=0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g] - y[g - 1])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static3 = static3 + 4
                        i = i + 1
                        flag = 1

        ElbowLeft['33'].append(repetition1)
        ElbowLeft['34'].append(repetition2)
        ElbowLeft['35'].append(repetition3)

        ElbowLeft['33'].append(static1)
        ElbowLeft['34'].append(static2)
        ElbowLeft['35'].append(static3)

        return ElbowLeft

    def calBtime(self):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        Back = {}

        for i in range(self._framenumber):
            angle = self.CalBackAngle(framenum=i)
            if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                if angle <= 30:
                    count1 = count1 + 1
                elif angle < 45:
                    count2 = count2 + 1
                elif angle <= 60:
                    count3 = count3 + 1
                elif angle < 90:
                    count4 = count4 + 1
                else:
                    count5 = count5 + 1
            else:
                if angle <= 10:
                    count6 = count6 + 1
                elif angle < 30:
                    count7 = count7 + 1
                else:
                    count8 = count8 + 1

        Back['89'] = [datetime.timedelta(seconds=round((count1 / self._framenumber) * (self._framenumber / 10)))]
        Back['90'] = [datetime.timedelta(seconds=round((count2 / self._framenumber) * (self._framenumber / 10)))]
        Back['91'] = [datetime.timedelta(seconds=round((count3 / self._framenumber) * (self._framenumber / 10)))]
        Back['92'] = [datetime.timedelta(seconds=round((count4 / self._framenumber) * (self._framenumber / 10)))]
        Back['93'] = [datetime.timedelta(seconds=round((count5 / self._framenumber) * (self._framenumber / 10)))]
        Back['100'] = [datetime.timedelta(seconds=round((count6 / self._framenumber) * (self._framenumber / 10)))]
        Back['101'] = [datetime.timedelta(seconds=round((count7 / self._framenumber) * (self._framenumber / 10)))]
        Back['102'] = [datetime.timedelta(seconds=round((count8 / self._framenumber) * (self._framenumber / 10)))]

        x = []
        x = self.CalBackAngle()
        orientation = self._orientation
        y = {}
        count = 0
        a = []
        b = []
        c = []
        d = []
        e = []
        f = []
        g = []
        h = []
        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            nestedori = orientation[i:i + 10]
            front = 0
            side = 0
            j = 0
            total = 0

            while j < len(z):
                if z[j] == 0:
                    del z[j]
                    del nestedori[j]
                else:
                    total = total + z[j]

                    if nestedori[j] == 'left' or nestedori[j] == 'right':
                        side = side + 1
                    else:
                        front = front + 1
                    j = j + 1
            if j != 0:
                y[count] = [total / j]
                if side >= front:
                    y[count].append(0)
                    # 0 stands for flexion
                else:
                    y[count].append(1)
                    # 1 stands for abduction or side bend
                count = count + 1

        for key, value in y.items():
            if value[1] == 0:
                if value[0] <= 30:
                    a.append(key)
                elif value[0] < 45:
                    b.append(key)
                elif value[0] <= 60:
                    c.append(key)
                elif value[0] < 90:
                    d.append(key)
                else:
                    e.append(key)
            else:
                if value[0] <= 10:
                    f.append(key)
                elif value[0] < 30:
                    g.append(key)
                else:
                    h.append(key)

        repetition1 = 0
        repetition2 = 0
        repetition3 = 0
        repetition4 = 0
        repetition5 = 0
        repetition6 = 0
        repetition7 = 0
        repetition8 = 0

        static1 = 0
        static2 = 0
        static3 = 0
        static4 = 0
        static5 = 0
        static6 = 0
        static7 = 0
        static8 = 0



        if len(a) > 1:
            for i in range(0, len(a) - 1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1

        if len(b) > 1:
            for i in range(0, len(b) - 1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c) - 1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        if len(d) > 1:
            for i in range(0, len(d) - 1):
                if abs(d[i] - d[i + 1]) > 1:
                    repetition4 = repetition4 + 1

        if len(e) > 1:
            for i in range(0, len(e) - 1):
                if abs(e[i] - e[i + 1]) > 1:
                    repetition5 = repetition5 + 1

        ########################BACK STATIC###########################################################

        if len(a) > 3:
            i = 3
            flag = 0
            while i < len(a):
                sumdiff = 0
                sumdiff = abs(a[i] - a[i - 1]) + abs(a[i - 1] - a[i - 2]) + abs(a[i - 2] - a[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(a[i] - a[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static1 = static1 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0  # flag2=1 at the end if the difference of all 4 elements are less than 2 degree to add 4s on static
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static1 = static1 + 4
                        i = i + 1
                        flag = 1

        if len(b) > 3:
            i = 3
            flag = 0
            while i < len(b):
                sumdiff = 0
                sumdiff = abs(b[i] - b[i - 1]) + abs(b[i - 1] - b[i - 2]) + abs(b[i - 2] - b[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(b[i] - b[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static2 = static2 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static2 = static2 + 4
                        i = i + 1
                        flag = 1

        if len(c) > 3:
            i = 3
            flag = 0
            while i < len(c):
                sumdiff = 0
                sumdiff = abs(c[i] - c[i - 1]) + abs(c[i - 1] - c[i - 2]) + abs(c[i - 2] - c[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(c[i] - c[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static3 = static3 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static3 = static3 + 4
                        i = i + 1
                        flag = 1

        if len(d) > 3:
            i = 3
            flag = 0
            while i < len(d):
                sumdiff = 0
                sumdiff = abs(d[i] - d[i - 1]) + abs(d[i - 1] - d[i - 2]) + abs(d[i - 2] - d[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(d[i] - d[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static4 = static4 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static4 = static4 + 4
                        i = i + 1
                        flag = 1

        if len(e) > 3:
            i = 3
            flag = 0
            while i < len(e):
                sumdiff = 0
                sumdiff = abs(e[i] - e[i - 1]) + abs(e[i - 1] - e[i - 2]) + abs(e[i - 2] - e[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(e[i] - e[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static5 = static5 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static5 = static5 + 4
                        i = i + 1
                        flag = 1



        Back['89'].append(repetition1)
        Back['90'].append(repetition2)
        Back['91'].append(repetition3)
        Back['92'].append(repetition4)
        Back['93'].append(repetition5)
        Back['100'].append(repetition6)
        Back['101'].append(repetition7)
        Back['102'].append(repetition8)

        Back['89'].append(static1)
        Back['90'].append(static2)
        Back['91'].append(static3)
        Back['92'].append(static4)
        Back['93'].append(static5)
        Back['100'].append(static6)
        Back['101'].append(static7)
        Back['102'].append(static8)

        return Back

    def calRStime(self):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        count9 = 0
        count10 = 0
        ShoulderRight = {}

        for i in range(self._framenumber):
            angle = self.CalRightShoulderAngle(framenum=i)
            if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                if angle <= 30:
                    count1 = count1 + 1
                elif angle <= 60:
                    count2 = count2 + 1
                elif angle <= 90:
                    count3 = count3 + 1
                elif angle < 120:
                    count4 = count4 + 1
                else:
                    count5 = count5 + 1
            else:
                if angle <= 30:
                    count6 = count6 + 1
                elif angle <= 60:
                    count7 = count7 + 1
                elif angle <= 90:
                    count8 = count8 + 1
                elif angle < 120:
                    count9 = count9 + 1
                else:
                    count10 = count10 + 1

        ShoulderRight['6'] = [datetime.timedelta(seconds=round((count1 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderRight['7'] = [datetime.timedelta(seconds=round((count2 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderRight['8'] = [datetime.timedelta(seconds=round((count3 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderRight['9'] = [datetime.timedelta(seconds=round((count4 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderRight['10'] = [datetime.timedelta(seconds=round((count5 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderRight['16'] = [datetime.timedelta(seconds=round((count6 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderRight['17'] = [datetime.timedelta(seconds=round((count7 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderRight['18'] = [datetime.timedelta(seconds=round((count8 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderRight['19'] = [datetime.timedelta(seconds=round((count9 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderRight['20'] = [datetime.timedelta(seconds=round((count10 / self._framenumber) * (self._framenumber / 10)))]

        x = []
        x = self.CalRightShoulderAngle()
        orientation = self._orientation
        y = {}
        count = 0
        a, b, c, d, e, f, g, h, k, l = [], [], [], [], [], [], [], [], [], []
        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            nestedori = orientation[i:i + 10]
            front = 0
            side = 0
            j = 0
            total = 0

            while j < len(z):
                if z[j] == 0:
                    del z[j]
                    del nestedori[j]
                else:
                    total = total + z[j]

                    if nestedori[j] == 'left' or nestedori[j] == 'right':
                        side = side + 1
                    else:
                        front = front + 1
                    j = j + 1
            if j != 0:
                y[count] = [total / j]
                if side >= front:
                    y[count].append(0)
                    # 0 stands for flexion
                else:
                    y[count].append(1)
                    # 1 stands for abduction or side bend
                count = count + 1

        for key, value in y.items():
            if value[1] == 0:
                if value[0] <= 30:
                    a.append(key)
                elif value[0] <= 60:
                    b.append(key)
                elif value[0] <= 90:
                    c.append(key)
                elif value[0] < 120:
                    d.append(key)
                else:
                    e.append(key)
            else:
                if value[0] <= 30:
                    f.append(key)
                elif value[0] <= 60:
                    g.append(key)
                elif value[0] <= 90:
                    h.append(key)
                elif value[0] < 120:
                    k.append(key)
                else:
                    l.append(key)

        repetition1 = 0
        repetition2 = 0
        repetition3 = 0
        repetition4 = 0
        repetition5 = 0
        repetition6 = 0
        repetition7 = 0
        repetition8 = 0
        repetition9 = 0
        repetition10 = 0

        static1 = 0
        static2 = 0
        static3 = 0
        static4 = 0
        static5 = 0
        static6 = 0
        static7 = 0
        static8 = 0
        static9 = 0
        static10 = 0

        if len(a) > 1:
            for i in range(0, len(a) - 1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1

        if len(b) > 1:
            for i in range(0, len(b) - 1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c) - 1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        if len(d) > 1:
            for i in range(0, len(d) - 1):
                if abs(d[i] - d[i + 1]) > 1:
                    repetition4 = repetition4 + 1

        if len(e) > 1:
            for i in range(0, len(e) - 1):
                if abs(e[i] - e[i + 1]) > 1:
                    repetition5 = repetition5 + 1

        #####################################Right Shoulder Static######################################

        if len(a) > 3:
            i = 3
            flag = 0
            while i < len(a):
                sumdiff = 0
                sumdiff = abs(a[i] - a[i - 1]) + abs(a[i - 1] - a[i - 2]) + abs(a[i - 2] - a[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(a[i] - a[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static1 = static1 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0  # flag2=1 at the end if the difference of all 4 elements are less than 2 degree to add 4s on static
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static1 = static1 + 4
                        i = i + 1
                        flag = 1

        if len(b) > 3:
            i = 3
            flag = 0
            while i < len(b):
                sumdiff = 0
                sumdiff = abs(b[i] - b[i - 1]) + abs(b[i - 1] - b[i - 2]) + abs(b[i - 2] - b[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(b[i] - b[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static2 = static2 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static2 = static2 + 4
                        i = i + 1
                        flag = 1

        if len(c) > 3:
            i = 3
            flag = 0
            while i < len(c):
                sumdiff = 0
                sumdiff = abs(c[i] - c[i - 1]) + abs(c[i - 1] - c[i - 2]) + abs(c[i - 2] - c[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(c[i] - c[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static3 = static3 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static3 = static3 + 4
                        i = i + 1
                        flag = 1

        if len(d) > 3:
            i = 3
            flag = 0
            while i < len(d):
                sumdiff = 0
                sumdiff = abs(d[i] - d[i - 1]) + abs(d[i - 1] - d[i - 2]) + abs(d[i - 2] - d[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(d[i] - d[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static4 = static4 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static4 = static4 + 4
                        i = i + 1
                        flag = 1

        if len(e) > 3:
            i = 3
            flag = 0
            while i < len(e):
                sumdiff = 0
                sumdiff = abs(e[i] - e[i - 1]) + abs(e[i - 1] - e[i - 2]) + abs(e[i - 2] - e[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(e[i] - e[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static5 = static5 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static5 = static5 + 4
                        i = i + 1
                        flag = 1


        ShoulderRight['6'].append(repetition1)
        ShoulderRight['7'].append(repetition2)
        ShoulderRight['8'].append(repetition3)
        ShoulderRight['9'].append(repetition4)
        ShoulderRight['10'].append(repetition5)
        ShoulderRight['16'].append(repetition6)
        ShoulderRight['17'].append(repetition7)
        ShoulderRight['18'].append(repetition8)
        ShoulderRight['19'].append(repetition9)
        ShoulderRight['20'].append(repetition10)

        ShoulderRight['6'].append(static1)
        ShoulderRight['7'].append(static2)
        ShoulderRight['8'].append(static3)
        ShoulderRight['9'].append(static4)
        ShoulderRight['10'].append(static5)
        ShoulderRight['16'].append(static6)
        ShoulderRight['17'].append(static7)
        ShoulderRight['18'].append(static8)
        ShoulderRight['19'].append(static9)
        ShoulderRight['20'].append(static10)

        return ShoulderRight

    def calLStime(self):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        count9 = 0
        count10 = 0
        ShoulderLeft = {}

        for i in range(self._framenumber):
            angle = self.CalLeftShoulderAngle(framenum=i)
            if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                if angle <= 30:
                    count1 = count1 + 1
                elif angle <= 60:
                    count2 = count2 + 1
                elif angle <= 90:
                    count3 = count3 + 1
                elif angle < 120:
                    count4 = count4 + 1
                else:
                    count5 = count5 + 1
            else:
                if angle <= 30:
                    count6 = count6 + 1
                elif angle <= 60:
                    count7 = count7 + 1
                elif angle <= 90:
                    count8 = count8 + 1
                elif angle < 120:
                    count9 = count9 + 1
                else:
                    count10 = count10 + 1

        ShoulderLeft['1'] = [datetime.timedelta(seconds=round((count1 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderLeft['2'] = [datetime.timedelta(seconds=round((count2 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderLeft['3'] = [datetime.timedelta(seconds=round((count3 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderLeft['4'] = [datetime.timedelta(seconds=round((count4 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderLeft['5'] = [datetime.timedelta(seconds=round((count5 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderLeft['11'] = [datetime.timedelta(seconds=round((count6 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderLeft['12'] = [datetime.timedelta(seconds=round((count7 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderLeft['13'] = [datetime.timedelta(seconds=round((count8 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderLeft['14'] = [datetime.timedelta(seconds=round((count9 / self._framenumber) * (self._framenumber / 10)))]
        ShoulderLeft['15'] = [datetime.timedelta(seconds=round((count10 / self._framenumber) * (self._framenumber / 10)))]

        x = []
        x = self.CalLeftShoulderAngle()
        orientation = self._orientation
        y = {}
        count = 0
        a, b, c, d, e, f, g, h, k, l = [], [], [], [], [], [], [], [], [], []
        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            nestedori = orientation[i:i + 10]
            front = 0
            side = 0
            j = 0
            total = 0

            while j < len(z):
                if z[j] == 0:
                    del z[j]
                    del nestedori[j]
                else:
                    total = total + z[j]

                    if nestedori[j] == 'left' or nestedori[j] == 'right':
                        side = side + 1
                    else:
                        front = front + 1
                    j = j + 1
            if j != 0:
                y[count] = [total / j]
                if side >= front:
                    y[count].append(0)
                    # 0 stands for flexion
                else:
                    y[count].append(1)
                    # 1 stands for abduction or side bend
                count = count + 1

        for key, value in y.items():
            if value[1] == 0:
                if value[0] <= 30:
                    a.append(key)
                elif value[0] <= 60:
                    b.append(key)
                elif value[0] <= 90:
                    c.append(key)
                elif value[0] < 120:
                    d.append(key)
                else:
                    e.append(key)
            else:
                if value[0] <= 30:
                    f.append(key)
                elif value[0] <= 60:
                    g.append(key)
                elif value[0] <= 90:
                    h.append(key)
                elif value[0] < 120:
                    k.append(key)
                else:
                    l.append(key)

        repetition1 = 0
        repetition2 = 0
        repetition3 = 0
        repetition4 = 0
        repetition5 = 0
        repetition6 = 0
        repetition7 = 0
        repetition8 = 0
        repetition9 = 0
        repetition10 = 0

        static1 = 0
        static2 = 0
        static3 = 0
        static4 = 0
        static5 = 0
        static5 = 0
        static6 = 0
        static7 = 0
        static8 = 0
        static9 = 0
        static10 = 0

        if len(a) > 1:
            for i in range(0, len(a) - 1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1

        if len(b) > 1:
            for i in range(0, len(b) - 1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c) - 1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        if len(d) > 1:
            for i in range(0, len(d) - 1):
                if abs(d[i] - d[i + 1]) > 1:
                    repetition4 = repetition4 + 1

        if len(e) > 1:
            for i in range(0, len(e) - 1):
                if abs(e[i] - e[i + 1]) > 1:
                    repetition5 = repetition5 + 1


        #########################################LEFT SHOULDER STATIC#########################################

        if len(a) > 3:
            i = 3
            flag = 0
            while i < len(a):
                sumdiff = 0
                sumdiff = abs(a[i] - a[i - 1]) + abs(a[i - 1] - a[i - 2]) + abs(a[i - 2] - a[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(a[i] - a[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static1 = static1 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0  # flag2=1 at the end if the difference of all 4 elements are less than 2 degree to add 4s on static
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static1 = static1 + 4
                        i = i + 1
                        flag = 1

        if len(b) > 3:
            i = 3
            flag = 0
            while i < len(b):
                sumdiff = 0
                sumdiff = abs(b[i] - b[i - 1]) + abs(b[i - 1] - b[i - 2]) + abs(b[i - 2] - b[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(b[i] - b[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static2 = static2 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static2 = static2 + 4
                        i = i + 1
                        flag = 1

        if len(c) > 3:
            i = 3
            flag = 0
            while i < len(c):
                sumdiff = 0
                sumdiff = abs(c[i] - c[i - 1]) + abs(c[i - 1] - c[i - 2]) + abs(c[i - 2] - c[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(c[i] - c[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static3 = static3 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static3 = static3 + 4
                        i = i + 1
                        flag = 1

        if len(d) > 3:
            i = 3
            flag = 0
            while i < len(d):
                sumdiff = 0
                sumdiff = abs(d[i] - d[i - 1]) + abs(d[i - 1] - d[i - 2]) + abs(d[i - 2] - d[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(d[i] - d[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static4 = static4 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static4 = static4 + 4
                        i = i + 1
                        flag = 1

        if len(e) > 3:
            i = 3
            flag = 0
            while i < len(e):
                sumdiff = 0
                sumdiff = abs(e[i] - e[i - 1]) + abs(e[i - 1] - e[i - 2]) + abs(e[i - 2] - e[i - 3])
                if sumdiff != 3 and flag == 0:
                    i = i + 1
                elif flag == 1:
                    if abs(e[i] - e[i - 1]) == 1:
                        if abs(y[i][0] - y[i - 1][0]) <= 2:
                            static5 = static5 + 1
                            i = i + 1
                        else:
                            flag = 0
                            i = i + 1
                    else:
                        flag = 0
                        i = i + 1
                else:
                    flag2 = 0
                    for g in range(i - 2, i + 1):
                        sumdiff2 = abs(y[g][0] - y[g - 1][0])
                        if sumdiff2 > 2:
                            flag2 = 0
                            break
                        else:
                            flag2 = 1
                    if flag2 == 0:
                        i = i + 1
                    else:
                        static5 = static5 + 4
                        i = i + 1
                        flag = 1


        ShoulderLeft['1'].append(repetition1)
        ShoulderLeft['2'].append(repetition2)
        ShoulderLeft['3'].append(repetition3)
        ShoulderLeft['4'].append(repetition4)
        ShoulderLeft['5'].append(repetition5)
        ShoulderLeft['11'].append(repetition6)
        ShoulderLeft['12'].append(repetition7)
        ShoulderLeft['13'].append(repetition8)
        ShoulderLeft['14'].append(repetition9)
        ShoulderLeft['15'].append(repetition10)

        ShoulderLeft['1'].append(static1)
        ShoulderLeft['2'].append(static2)
        ShoulderLeft['3'].append(static3)
        ShoulderLeft['4'].append(static4)
        ShoulderLeft['5'].append(static5)
        ShoulderLeft['11'].append(static6)
        ShoulderLeft['12'].append(static7)
        ShoulderLeft['13'].append(static8)
        ShoulderLeft['14'].append(static9)
        ShoulderLeft['15'].append(static10)

        return ShoulderLeft
    

    def legaction(self):        
        count1=0
        count2=0
        count3=0
        count4=0
        count5=0
        Legaction={}
        for i in range(self._framenumber):
            backangle = self.CalBackAngle(framenum=i)
            leftlegangle = self.CalLeftLeg(framenum=i)
            rightlegangle = self.CalRightLeg(framenum=i)
            if backangle>20:
                if leftlegangle > 150 or rightlegangle > 150:
                    count1=count1+1  #stooping
                else:
                    count2=count2+1 #bending
            else:
                if leftlegangle>130 or rightlegangle>130:
                    count3=count3+1 #standing
                else:
                    leftdiff1 = abs(self._column['B_8_Y'][i] - self._column['B_9_Y'][i])
                    leftdiff2 = abs(self._column['B_10_Y'][i] - self._column['B_9_Y'][i])
                    rightdiff1 = abs(self._column['B_11_Y'][i] - self._column['B_12_Y'][i])
                    rightdiff2 = abs(self._column['B_13_Y'][i] - self._column['B_12_Y'][i])

                    if leftdiff1 < leftdiff2 or rightdiff1 < rightdiff2:
                        count4=count4+1  #crouching
                    else:
                        count5=count5+1  #kneeling

        Legaction['104']=[datetime.timedelta(seconds=round((count3 / self._framenumber) * (self._framenumber / 10)))]
        Legaction['105'] = [datetime.timedelta(seconds=round((count5 / self._framenumber) * (self._framenumber / 10)))]
        Legaction['108'] = [datetime.timedelta(seconds=round((count4 / self._framenumber) * (self._framenumber / 10)))]

        Legaction['104'].append(0)
        Legaction['105'].append(0)
        Legaction['108'].append(0)

        Legaction['104'].append(0)
        Legaction['105'].append(0)
        Legaction['108'].append(0)

        return Legaction
    '''
    def calNecktime(self):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        Neck = {}

        for i in range(self._framenumber):
            if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                if self.CalNeckAngle(framenum=i) <= 20:
                    count1 = count1 + 1
                elif self.CalNeckAngle(framenum=i) <= 30:
                    count2 = count2 + 1
                elif self.CalNeckAngle(framenum=i) <= 44:
                    count3 = count3 + 1
                else:
                    count4 = count4 + 1
            else:
                if self.CalNeckAngle(framenum=i) <= 10:
                    count5 = count5 + 1
                elif self.CalNeckAngle(framenum=i) <= 20:
                    count6 = count6 + 1
                else:
                    count7 = count7 + 1

        Neck['75'] = (count1 / self._framenumber) * (self._framenumber / 10)
        Neck['76'] = (count2 / self._framenumber) * (self._framenumber / 10)
        Neck['77'] = (count3 / self._framenumber) * (self._framenumber / 10)
        Neck['78'] = (count4 / self._framenumber) * (self._framenumber / 10)
        Neck['85'] = (count5 / self._framenumber) * (self._framenumber / 10)
        Neck['86'] = (count6 / self._framenumber) * (self._framenumber / 10)
        Neck['87'] = (count7 / self._framenumber) * (self._framenumber / 10)

        return Neck
      

    def calREtime(self):
        count1 = 0
        count2 = 0
        count3 = 0

        ElbowRight = {}

        for i in range(self._framenumber):
            if self.CalRightElbow(framenum=i) < 60:
                count1 = count1 + 1
            elif self.CalRightElbow(framenum=i) <= 100:
                count2 = count2 + 1
            else:
                count3 = count3 + 1

        ElbowRight['36'] = (count1 / self._framenumber) * (self._framenumber / 10)
        ElbowRight['37'] = (count2 / self._framenumber) * (self._framenumber / 10)
        ElbowRight['38'] = (count3 / self._framenumber) * (self._framenumber / 10)

        return ElbowRight

    def calLEtime(self):
        count1 = 0
        count2 = 0
        count3 = 0

        ElbowLeft = {}

        for i in range(self._framenumber):
            if self.CalLeftElbow(framenum=i) < 60:
                count1 = count1 + 1
            elif self.CalLeftElbow(framenum=i) <= 100:
                count2 = count2 + 1
            else:
                count3 = count3 + 1

        ElbowLeft['33'] = (count1 / self._framenumber) * (self._framenumber / 10)
        ElbowLeft['34'] = (count2 / self._framenumber) * (self._framenumber / 10)
        ElbowLeft['35'] = (count3 / self._framenumber) * (self._framenumber / 10)

        return ElbowLeft

    def calBtime(self):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        Back = {}

        for i in range(self._framenumber):
            if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                if self.CalBackAngle(framenum=i) < 30:
                    count1 = count1 + 1
                elif self.CalBackAngle(framenum=i) < 45:
                    count2 = count2 + 1
                elif self.CalBackAngle(framenum=i) < 60:
                    count3 = count3 + 1
                elif self.CalBackAngle(framenum=i) < 90:
                    count4 = count4 + 1
                else:
                    count5 = count5 + 1
            else:
                if self.CalBackAngle(framenum=i) <= 10:
                    count6 = count6 + 1
                elif self.CalBackAngle(framenum=i) < 30:
                    count7 = count7 + 1
                else:
                    count8 = count8 + 1

        Back['89'] = (count1 / self._framenumber) * (self._framenumber / 10)
        Back['90'] = (count2 / self._framenumber) * (self._framenumber / 10)
        Back['91'] = (count3 / self._framenumber) * (self._framenumber / 10)
        Back['92'] = (count4 / self._framenumber) * (self._framenumber / 10)
        Back['93'] = (count5 / self._framenumber) * (self._framenumber / 10)
        Back['100'] = (count6 / self._framenumber) * (self._framenumber / 10)
        Back['101'] = (count7 / self._framenumber) * (self._framenumber / 10)
        Back['102'] = (count8 / self._framenumber) * (self._framenumber / 10)

        return Back

    def calRStime(self):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        count9 = 0
        count10 = 0
        ShoulderRight = {}

        for i in range(self._framenumber):
            if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                if self.CalRightShoulderAngle(framenum=i) <= 30:
                    count1 = count1 + 1
                elif self.CalRightShoulderAngle(framenum=i) <= 60:
                    count2 = count2 + 1
                elif self.CalRightShoulderAngle(framenum=i) <= 90:
                    count3 = count3 + 1
                elif self.CalRightShoulderAngle(framenum=i) < 120:
                    count4 = count4 + 1
                else:
                    count5 = count5 + 1
            else:
                if self.CalRightShoulderAngle(framenum=i) <= 30:
                    count6 = count6 + 1
                elif self.CalRightShoulderAngle(framenum=i) <= 60:
                    count7 = count7 + 1
                elif self.CalRightShoulderAngle(framenum=i) <= 90:
                    count8 = count8 + 1
                elif self.CalRightShoulderAngle(framenum=i) < 120:
                    count9 = count9 + 1
                else:
                    count10 = count10 + 1

        ShoulderRight['6'] = (count1 / self._framenumber) * (self._framenumber / 10)
        ShoulderRight['7'] = (count2 / self._framenumber) * (self._framenumber / 10)
        ShoulderRight['8'] = (count3 / self._framenumber) * (self._framenumber / 10)
        ShoulderRight['9'] = (count4 / self._framenumber) * (self._framenumber / 10)
        ShoulderRight['10'] = (count5 / self._framenumber) * (self._framenumber / 10)
        ShoulderRight['16'] = (count6 / self._framenumber) * (self._framenumber / 10)
        ShoulderRight['17'] = (count7 / self._framenumber) * (self._framenumber / 10)
        ShoulderRight['18'] = (count8 / self._framenumber) * (self._framenumber / 10)
        ShoulderRight['19'] = (count9 / self._framenumber) * (self._framenumber / 10)
        ShoulderRight['20'] = (count10 / self._framenumber) * (self._framenumber / 10)

        return ShoulderRight

    def calLStime(self):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        count9 = 0
        count10 = 0
        ShoulderLeft = {}

        for i in range(self._framenumber):
            if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                if self.CalLeftShoulderAngle(framenum=i) <= 30:
                    count1 = count1 + 1
                elif self.CalLeftShoulderAngle(framenum=i) <= 60:
                    count2 = count2 + 1
                elif self.CalLeftShoulderAngle(framenum=i) <= 90:
                    count3 = count3 + 1
                elif self.CalLeftShoulderAngle(framenum=i) < 120:
                    count4 = count4 + 1
                else:
                    count5 = count5 + 1
            else:
                if self.CalLeftShoulderAngle(framenum=i) <= 30:
                    count6 = count6 + 1
                elif self.CalLeftShoulderAngle(framenum=i) <= 60:
                    count7 = count7 + 1
                elif self.CalLeftShoulderAngle(framenum=i) <= 90:
                    count8 = count8 + 1
                elif self.CalLeftShoulderAngle(framenum=i) < 120:
                    count9 = count9 + 1
                else:
                    count10 = count10 + 1

        ShoulderLeft['1'] = (count1 / self._framenumber) * (self._framenumber / 10)
        ShoulderLeft['2'] = (count2 / self._framenumber) * (self._framenumber / 10)
        ShoulderLeft['3'] = (count3 / self._framenumber) * (self._framenumber / 10)
        ShoulderLeft['4'] = (count4 / self._framenumber) * (self._framenumber / 10)
        ShoulderLeft['5'] = (count5 / self._framenumber) * (self._framenumber / 10)
        ShoulderLeft['11'] = (count6 / self._framenumber) * (self._framenumber / 10)
        ShoulderLeft['12'] = (count7 / self._framenumber) * (self._framenumber / 10)
        ShoulderLeft['13'] = (count8 / self._framenumber) * (self._framenumber / 10)
        ShoulderLeft['14'] = (count9 / self._framenumber) * (self._framenumber / 10)
        ShoulderLeft['15'] = (count10 / self._framenumber) * (self._framenumber / 10)

        return ShoulderLeft
        '''
    ################################################################################################################

    def readimage(self, datapath, framenum=0):
        self._column['image'] = []
        if framenum == 0:
            i = 0
            while i < self._framenumber:
                img = cv2.imread('%s/frame%05d.jpg' % (datapath, i))
                self._column['image'].append(img)
                i = i + 1
        else:
            img = cv2.imread(r'%s/frame%05d.jpg' % (datapath, framenum))
            self._column['image'].append(img)

        return self._column['image']

    def drawskeleton(self, framenum=0, color=(255, 0, 0), thickness=3):
        if framenum == 0 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]

                pose0 = (int(self._column['B_0_X'][i]), int(self._column['B_0_Y'][i]))
                pose1 = (int(self._column['B_1_X'][i]), int(self._column['B_1_Y'][i]))
                pose2 = (int(self._column['B_2_X'][i]), int(self._column['B_2_Y'][i]))
                pose3 = (int(self._column['B_3_X'][i]), int(self._column['B_3_Y'][i]))
                pose4 = (int(self._column['B_4_X'][i]), int(self._column['B_4_Y'][i]))
                pose5 = (int(self._column['B_5_X'][i]), int(self._column['B_5_Y'][i]))
                pose6 = (int(self._column['B_6_X'][i]), int(self._column['B_6_Y'][i]))
                pose7 = (int(self._column['B_7_X'][i]), int(self._column['B_7_Y'][i]))
                pose8 = (int(self._column['B_8_X'][i]), int(self._column['B_8_Y'][i]))
                pose9 = (int(self._column['B_9_X'][i]), int(self._column['B_9_Y'][i]))
                pose10 = (int(self._column['B_10_X'][i]), int(self._column['B_10_Y'][i]))
                pose11 = (int(self._column['B_11_X'][i]), int(self._column['B_11_Y'][i]))
                pose12 = (int(self._column['B_12_X'][i]), int(self._column['B_12_Y'][i]))
                pose13 = (int(self._column['B_13_X'][i]), int(self._column['B_13_Y'][i]))
                pose14 = (int(self._column['B_14_X'][i]), int(self._column['B_14_Y'][i]))
                pose15 = (int(self._column['B_15_X'][i]), int(self._column['B_15_Y'][i]))
                pose16 = (int(self._column['B_16_X'][i]), int(self._column['B_16_Y'][i]))
                pose17 = (int(self._column['B_17_X'][i]), int(self._column['B_17_Y'][i]))

                neckangle = self.CalNeckAngle(framenum=i)
                if self._orientation[i]=="left" or self._orientation[i]=="right":
                    if neckangle <= 20:
                        neckcolor=(0,150,0)
                    elif neckangle <= 30:
                        neckcolor=(0,255,255)
                    elif neckangle <= 44:
                        neckcolor=(0,150,255)
                    else:
                        neckcolor=(0,0,255)
                else:
                    if neckangle <= 10:
                        neckcolor=(0,150,0)
                    elif neckangle <= 20:
                        neckcolor=(0,255,255)
                    else:
                        neckcolor=(0,0,255)

                leftshoulderangle=self.CalLeftShoulderAngle(framenum=i)
                if leftshoulderangle <= 30:
                    leftshouldercolor = (0, 150, 0)
                elif leftshoulderangle <= 60:
                    leftshouldercolor = (0, 255, 255)
                elif leftshoulderangle <= 90:
                    leftshouldercolor = (0, 150, 255)
                else:
                    leftshouldercolor = (0, 0, 255)

                rightshoulderangle = self.CalRightShoulderAngle(framenum=i)
                if rightshoulderangle <= 30:
                    rightshouldercolor = (0, 150, 0)
                elif rightshoulderangle <= 60:
                    rightshouldercolor = (0, 255, 255)
                elif leftshoulderangle <= 90:
                    rightshouldercolor = (0, 150, 255)
                else:
                    rightshouldercolor = (0, 0, 255)

                backangle = self.CalBackAngle(framenum=i)
                if self._orientation[i] == "left" or self._orientation[i] == "right":
                    if backangle <= 30:
                        backcolor = (0, 150, 0)
                    elif backangle <= 45:
                        backcolor = (0, 255, 255)
                    elif backangle <= 60:
                        backcolor = (0, 150, 255)
                    elif backangle <= 90:
                        backcolor = (0, 0, 255)
                    else:
                        backcolor = (255, 0, 0)
                else:
                    if backangle <= 10:
                        backcolor = (0, 150, 0)
                    elif backangle <= 30:
                        backcolor = (0, 255, 255)
                    else:
                        backcolor = (0, 0, 255)

                leftelbowangle=self.CalLeftElbow(framenum=i)
                if leftelbowangle <= 60:
                    leftelbowcolor = (0, 255, 255)
                elif leftelbowangle <= 100:
                    leftelbowcolor = (0, 150, 0)
                else:
                    leftelbowcolor = (0, 0, 255)

                rightelbowangle = self.CalRightElbow(framenum=i)
                if rightelbowangle <= 60:
                    rightelbowcolor = (0, 255, 255)
                elif rightelbowangle <= 100:
                    rightelbowcolor = (0, 150, 0)
                else:
                    rightelbowcolor = (0, 0, 255)


                if 0 not in pose0 and 0 not in pose1: self._column['image'][i] = cv2.line(img, pose0, pose1, neckcolor,
                                                                                          thickness)
                if 0 not in pose1 and 0 not in pose2: self._column['image'][i] = cv2.line(img, pose1, pose2, rightshouldercolor,
                                                                                          thickness)
                if 0 not in pose2 and 0 not in pose3: self._column['image'][i] = cv2.line(img, pose2, pose3, rightelbowcolor,
                                                                                          thickness)
                if 0 not in pose3 and 0 not in pose4: self._column['image'][i] = cv2.line(img, pose3, pose4, rightelbowcolor,
                                                                                          thickness)
                if 0 not in pose1 and 0 not in pose5: self._column['image'][i] = cv2.line(img, pose1, pose5, leftshouldercolor,
                                                                                          thickness)
                if 0 not in pose5 and 0 not in pose6: self._column['image'][i] = cv2.line(img, pose5, pose6, leftelbowcolor,
                                                                                          thickness)
                if 0 not in pose6 and 0 not in pose7: self._column['image'][i] = cv2.line(img, pose6, pose7, leftelbowcolor,
                                                                                          thickness)
                if 0 not in pose0 and 0 not in pose14: self._column['image'][i] = cv2.line(img, pose0, pose14, color,
                                                                                           thickness)
                if 0 not in pose0 and 0 not in pose15: self._column['image'][i] = cv2.line(img, pose0, pose15, color,
                                                                                           thickness)
                if 0 not in pose14 and 0 not in pose16: self._column['image'][i] = cv2.line(img, pose14, pose16, color,
                                                                                            thickness)
                if 0 not in pose15 and 0 not in pose17: self._column['image'][i] = cv2.line(img, pose15, pose17, color,
                                                                                            thickness)
                #if 0 not in pose1 and 0 not in pose8: self._column['image'][i] = cv2.line(img, pose1, pose8, backcolor,
                                                                                          #thickness)
                if 0 not in pose2 and 0 not in pose8: self._column['image'][i] = cv2.line(img, pose2, pose8, backcolor,
                                                                                          thickness)
                #if 0 not in pose1 and 0 not in pose11: self._column['image'][i] = cv2.line(img, pose1, pose11, backcolor,
                                                                                           #thickness)
                if 0 not in pose5 and 0 not in pose11: self._column['image'][i] = cv2.line(img, pose5, pose11, backcolor,
                                                                                           thickness)
                if 0 not in pose8 and 0 not in pose9: self._column['image'][i] = cv2.line(img, pose8, pose9, color,
                                                                                          thickness)
                if 0 not in pose9 and 0 not in pose10: self._column['image'][i] = cv2.line(img, pose9, pose10, color,
                                                                                           thickness)
                if 0 not in pose11 and 0 not in pose12: self._column['image'][i] = cv2.line(img, pose11, pose12, color,
                                                                                            thickness)
                if 0 not in pose12 and 0 not in pose13: self._column['image'][i] = cv2.line(img, pose12, pose13, color,
                                                                                            thickness)

                i = i + 1

        return self._column['image']

    #######################################################################################################################
    def DrawNeckAngle(self, framenum=0, position=(20, 50), color=(255, 100, 0), thickness=2):
        if framenum == 0 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                angle = self.CalNeckAngle(framenum=i)  # try self.CalNeckAnge(framenum=i)next time
                if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                    if angle <= 20:
                        g = 'Flexion1'
                        self._column['image'][i] = cv2.putText(img, 'Neck ' + g + ':' + str(round(angle, 2)), position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                    elif angle <= 30:
                        g = 'Flexion2'
                        self._column['image'][i] = cv2.putText(img, 'Neck ' + g + ':' + str(round(angle, 2)), position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                    elif angle <= 44:
                        g = 'Flexion3'
                        self._column['image'][i] = cv2.putText(img, 'Neck ' + g + ':' + str(round(angle, 2)), position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 255), thickness)
                    else:
                        g = 'Flexion4'
                        self._column['image'][i] = cv2.putText(img, 'Neck ' + g + ':' + str(round(angle, 2)), position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                else:
                    if angle <= 10:
                        g = 'Side Bend1'
                        self._column['image'][i] = cv2.putText(img, 'Neck ' + g + ':' + str(round(angle, 2)), position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                    elif angle <= 20:
                        g = 'Side Bend2'
                        self._column['image'][i] = cv2.putText(img, 'Neck ' + g + ':' + str(round(angle, 2)), position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                    else:
                        g = 'Side Bend3'
                        self._column['image'][i] = cv2.putText(img, 'Neck ' + g + ':' + str(round(angle, 2)), position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)

                i = i + 1

        return self._column['image']

    def DrawLSAngle(self, framenum=0, position=(20, 80), color=(255, 100, 0), thickness=2):
        if framenum == 0 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                angle = self.CalLeftShoulderAngle(framenum=i)
                if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                    if angle <= 30:
                        g = 'Flexion1'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                    elif angle <= 60:
                        g = 'Flexion2'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                    elif angle <= 90:
                        g = 'Flexion3'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 255), thickness)
                    elif angle <= 120:
                        g = 'Flexion4'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                    else:
                        g = 'Flexion5'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                else:
                    if angle <= 30:
                        g = 'Abduction1'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                    elif angle <= 60:
                        g = 'Abduction2'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                    elif angle <= 90:
                        g = 'Abuction3'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 255), thickness)
                    elif angle <= 120:
                        g = 'Abduction4'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                    else:
                        g = 'Abduction5'
                        self._column['image'][i] = cv2.putText(img, 'Left Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)

                i = i + 1

        return self._column['image']

    def DrawRSAngle(self, framenum=0, position=(20, 110), color=(255, 100, 0), thickness=2):
        if framenum == 0 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                angle = self.CalRightShoulderAngle(framenum=i)
                if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                    if angle <= 30:
                        g = 'Flexion1'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                    elif angle <= 60:
                        g = 'Flexion2'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                    elif angle <= 90:
                        g = 'Flexion3'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 255), thickness)
                    elif angle <= 120:
                        g = 'Flexion4'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                    else:
                        g = 'Flexion5'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                else:
                    if angle <= 30:
                        g = 'Abduction1'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                    elif angle <= 60:
                        g = 'Abduction2'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                    elif angle <= 90:
                        g = 'Abduction3'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 255), thickness)
                    elif angle <= 120:
                        g = 'Abduction4'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                    else:
                        g = 'Abduction5'
                        self._column['image'][i] = cv2.putText(img, 'Right Shoulder ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)

                i = i + 1

        return self._column['image']

    def DrawBackAngle(self, framenum=0, position=(20, 140), color=(255, 100, 0), thickness=2):
        if framenum == 0 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                angle = self.CalBackAngle(framenum=i)
                if self._orientation[i] == 'left' or self._orientation[i] == 'right':
                    if angle <= 30:
                        g = 'Flexion1'
                        self._column['image'][i] = cv2.putText(img, 'Back Angle ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                    elif angle <= 45:
                        g = 'Flexion2'
                        self._column['image'][i] = cv2.putText(img, 'Back Angle ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                    elif angle <= 60:
                        g = 'Flexion3'
                        self._column['image'][i] = cv2.putText(img, 'Back Angle ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 255), thickness)
                    elif angle < 90:
                        g = 'Flexion4'
                        self._column['image'][i] = cv2.putText(img, 'Back Angle ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                    else:
                        g = 'Flexion5'
                        self._column['image'][i] = cv2.putText(img, 'Back Angle ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 70, 150),
                                                               thickness)
                else:
                    if angle <= 10:
                        g = 'Bend1'
                        self._column['image'][i] = cv2.putText(img, 'Back Angle ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                    elif angle < 30:
                        g = 'Bend2'
                        self._column['image'][i] = cv2.putText(img, 'Back Angle ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                    else:
                        g = 'Bend3'
                        self._column['image'][i] = cv2.putText(img, 'Back Angle ' + g + ':' + str(round(angle, 2)),
                                                               position,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)

                i = i + 1

        return self._column['image']

    def DrawLEAngle(self, framenum=0, position=(20, 170), color=(255, 100, 0), thickness=2):
        if framenum == 0 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                angle = self.CalLeftElbow(framenum=i)
                if angle < 60:
                    g = 'Flexion1'
                    self._column['image'][i] = cv2.putText(img, 'Left Elbow ' + g + ':' + str(round(angle, 2)),
                                                           position,
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                elif angle <= 100:
                    g = 'Flexion2'
                    self._column['image'][i] = cv2.putText(img, 'Left Elbow ' + g + ':' + str(round(angle, 2)),
                                                           position,
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                else:
                    g = 'Flexion3'
                    self._column['image'][i] = cv2.putText(img, 'Left Elbow ' + g + ':' + str(round(angle, 2)),
                                                           position,
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                i = i + 1

        return self._column['image']

    def DrawREAngle(self, framenum=0, position=(20, 200), color=(255, 100, 0), thickness=2):
        if framenum == 0 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                angle = self.CalRightElbow(framenum=i)
                if angle < 60:
                    g = 'Flexion1'
                    self._column['image'][i] = cv2.putText(img, 'Right Elbow ' + g + ':' + str(round(angle, 2)),
                                                           position,
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), thickness)
                elif angle <= 100:
                    g = 'Flexion2'
                    self._column['image'][i] = cv2.putText(img, 'Right Elbow ' + g + ':' + str(round(angle, 2)),
                                                           position,
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), thickness)
                else:
                    g = 'Flexion3'
                    self._column['image'][i] = cv2.putText(img, 'Right Elbow ' + g + ':' + str(round(angle, 2)),
                                                           position,
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), thickness)
                i = i + 1

        return self._column['image']

    def DrawLLAngle(self, framenum=0, position=(20, 350), color=(255, 0, 0), thickness=2):
        if framenum == 0 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                angle = self.CalLeftLeg(framenum=i)
                self._column['image'][i] = cv2.putText(img, 'Left leg:' + str(round(angle, 2)),
                                                       position,
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness)

                i=i+1

    def DrawRLAngle(self, framenum=0, position=(20, 380), color=(255, 0, 0), thickness=2):
        if framenum == 0 and self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                angle = self.CalRightLeg(framenum=i)
                self._column['image'][i] = cv2.putText(img, 'Right leg:' + str(round(angle, 2)),
                                                       position,
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness)

                i=i+1


    def action(self,framenum=-1, position=(20,410),color=(255, 0, 0), thickness=2):
        self._column['action']=[]
        if framenum==-1 and self._column['image'] != []:
            i=0
            while i < self._framenumber:
                img = self._column['image'][i]
                backangle=self.CalBackAngle(framenum=i)
                leftlegangle = self.CalLeftLeg(framenum=i)
                rightlegangle = self.CalRightLeg(framenum=i)
                #Lshoulderangle = self.CalLeftShoulderAngle(framenum=i)
                #Rshoulderangle = self.CalRightShoulderAngle(framenum=i)
                if backangle>20:
                    if leftlegangle>150 or rightlegangle>150:
                        self._column['action'].append('stooping')
                    else:
                        self._column['action'].append('bending')
                else:
                    if self._column['B_4_Y'][i]<self._column['B_2_Y'][i] or self._column['B_7_Y'][i]<self._column['B_5_Y'][i]:
                        self._column['action'].append('hand above shoulder')
                    elif leftlegangle>130 or rightlegangle>130:
                        self._column['action'].append('standing')
                    else:
                        leftdiff1=abs(self._column['B_8_Y'][i]-self._column['B_9_Y'][i])
                        leftdiff2=abs(self._column['B_10_Y'][i]-self._column['B_9_Y'][i])
                        rightdiff1=abs(self._column['B_11_Y'][i]-self._column['B_12_Y'][i])
                        rightdiff2=abs(self._column['B_13_Y'][i]-self._column['B_12_Y'][i])

                        if leftdiff1<leftdiff2 or rightdiff1<rightdiff2:
                            self._column['action'].append('crouching')
                        else:
                            self._column['action'].append('kneeling')
                self._column['image'][i] = cv2.putText(img, 'Action:' + str(self._column['action'][i]),
                                                                   position,
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,
                                                                   thickness)
                i=i+1

        return  self._column['image']



    #######################################################################################################################




    def writeimage(self, path):
        if self._column['image'] != []:
            i = 0
            while i < self._framenumber:
                img = self._column['image'][i]
                cv2.imwrite(
                    '%s/frame%05d.jpg' % (path, i),
                    img)
                i = i + 1

    def Ninegridbox(self, framenum=-1, color=(127, 127, 127)):
        if framenum == -1 and self._column['image'] != []:
            i = 0
            self._column['lefthand'] = []
            self._column['righthand'] = []
            while i < self._framenumber:
                img = self._column['image'][i]

                pose0 = (int(self._column['B_0_X'][i]), int(self._column['B_0_Y'][i]))
                pose1 = (int(self._column['B_1_X'][i]), int(self._column['B_1_Y'][i]))
                pose2 = (int(self._column['B_2_X'][i]), int(self._column['B_2_Y'][i]))
                pose3 = (int(self._column['B_3_X'][i]), int(self._column['B_3_Y'][i]))
                pose4 = (int(self._column['B_4_X'][i]), int(self._column['B_4_Y'][i]))
                pose5 = (int(self._column['B_5_X'][i]), int(self._column['B_5_Y'][i]))
                pose6 = (int(self._column['B_6_X'][i]), int(self._column['B_6_Y'][i]))
                pose7 = (int(self._column['B_7_X'][i]), int(self._column['B_7_Y'][i]))
                pose8 = (int(self._column['B_8_X'][i]), int(self._column['B_8_Y'][i]))
                pose9 = (int(self._column['B_9_X'][i]), int(self._column['B_9_Y'][i]))
                pose10 = (int(self._column['B_10_X'][i]), int(self._column['B_10_Y'][i]))
                pose11 = (int(self._column['B_11_X'][i]), int(self._column['B_11_Y'][i]))
                pose12 = (int(self._column['B_12_X'][i]), int(self._column['B_12_Y'][i]))
                pose13 = (int(self._column['B_13_X'][i]), int(self._column['B_13_Y'][i]))
                pose14 = (int(self._column['B_14_X'][i]), int(self._column['B_14_Y'][i]))
                pose15 = (int(self._column['B_15_X'][i]), int(self._column['B_15_Y'][i]))
                pose16 = (int(self._column['B_16_X'][i]), int(self._column['B_16_Y'][i]))
                pose17 = (int(self._column['B_17_X'][i]), int(self._column['B_17_Y'][i]))

                if pose10[0] * pose9[0] + pose10[1] * pose9[1] == 0:
                    foot = np.linalg.norm(np.array(pose13) - np.array(pose12))
                elif pose13[0] * pose12[0] + pose13[1] * pose12[1] == 0:
                    foot = np.linalg.norm(np.array(pose10) - np.array(pose9))
                else:
                    foot = max(np.linalg.norm(np.array(pose10) - np.array(pose9)),
                               np.linalg.norm(np.array(pose13) - np.array(pose12)))

                if pose8[0] * pose9[0] + pose8[1] * pose9[1] == 0:
                    leg = np.linalg.norm(np.array(pose11) - np.array(pose12))
                elif pose11[0] * pose12[0] + pose11[1] * pose12[1] == 0:
                    leg = np.linalg.norm(np.array(pose8) - np.array(pose9))
                else:
                    leg = max(np.linalg.norm(np.array(pose8) - np.array(pose9)),
                              np.linalg.norm(np.array(pose11) - np.array(pose12)))

                if pose4[0] * pose3[0] + pose4[1] * pose3[1] == 0:
                    hand = np.linalg.norm(np.array(pose7) - np.array(pose6))
                elif pose7[0] * pose6[0] + pose7[1] * pose6[1] == 0:
                    hand = np.linalg.norm(np.array(pose4) - np.array(pose3))
                else:
                    hand = max(np.linalg.norm(np.array(pose4) - np.array(pose3)),
                               np.linalg.norm(np.array(pose7) - np.array(pose6)))

                if pose2[0] * pose3[0] + pose2[1] * pose3[1] == 0:
                    arm = np.linalg.norm(np.array(pose5) - np.array(pose6))
                elif pose5[0] * pose6[0] + pose5[1] * pose6[1] == 0:
                    arm = np.linalg.norm(np.array(pose2) - np.array(pose3))
                else:
                    arm = max(np.linalg.norm(np.array(pose2) - np.array(pose3)),
                              np.linalg.norm(np.array(pose5) - np.array(pose6)))

                bodylength = np.linalg.norm(
                    np.array(pose1) - np.array((self._column['CTR_X'][i], self._column['CTR_Y'][i])))
                '''
                foot = max(numpy.linalg.norm(np.array(pose10) - np.array(pose9)),
                           numpy.linalg.norm(np.array(pose13) - np.array(pose12)))
                leg = max(numpy.linalg.norm(np.array(pose8) - np.array(pose9)),
                          numpy.linalg.norm(np.array(pose11) - np.array(pose12)))

                hand = max(numpy.linalg.norm(np.array(pose4) - np.array(pose3)),
                           numpy.linalg.norm(np.array(pose7) - np.array(pose6)))
                arm = max(numpy.linalg.norm(np.array(pose2) - np.array(pose3)),
                          numpy.linalg.norm(np.array(pose5) - np.array(pose6)))
                '''
                ref1 = bodylength / 1.4142

                # print('foot', foot)
                # print('leg', leg)
                # print('bodylength', bodylength)
                # print('hand', hand)
                # print('arm', arm)

                bottompoint = max(pose10[1], pose13[1])

                if self._orientation[i] == 'left':
                    GridPoint4 = (int(self._column['CTR_X'][i]), int(bottompoint))
                    GridPoint3 = (int(self._column['CTR_X'][i]), int(bottompoint - foot))
                    GridPoint2 = (int(self._column['CTR_X'][i]), int(bottompoint - foot - leg - bodylength))
                    GridPoint1 = (
                    int(self._column['CTR_X'][i]), int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint8 = (int(self._column['CTR_X'][i] - hand), int(bottompoint))
                    GridPoint7 = (int(self._column['CTR_X'][i] - hand), int(bottompoint - foot))
                    GridPoint6 = (int(self._column['CTR_X'][i] - hand), int(bottompoint - foot - leg - bodylength))
                    GridPoint5 = (
                    int(self._column['CTR_X'][i] - hand), int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint12 = (int(self._column['CTR_X'][i] - hand - arm), int(bottompoint))
                    GridPoint11 = (int(self._column['CTR_X'][i] - hand - arm), int(bottompoint - foot))
                    GridPoint10 = (
                    int(self._column['CTR_X'][i] - hand - arm), int(bottompoint - foot - leg - bodylength))
                    GridPoint9 = (
                    int(self._column['CTR_X'][i] - hand - arm), int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint16 = (int(self._column['CTR_X'][i] - hand - arm - ref1), int(bottompoint))
                    GridPoint15 = (int(self._column['CTR_X'][i] - hand - arm - ref1), int(bottompoint - foot))
                    GridPoint14 = (
                    int(self._column['CTR_X'][i] - hand - arm - ref1), int(bottompoint - foot - leg - bodylength))
                    GridPoint13 = (int(self._column['CTR_X'][i] - hand - arm - ref1),
                                   int(bottompoint - foot - leg - bodylength - arm - hand))

                    if pose4[0] > GridPoint5[0] and pose4[0] < GridPoint1[0]:
                        rhx = 0
                    elif pose4[0] > GridPoint9[0] and pose4[0] < GridPoint5[0]:
                        rhx = 3
                    elif pose4[0] > GridPoint13[0] and pose4[0] < GridPoint9[0]:
                        rhx = 6
                    else:
                        rhx = 0

                    if pose4[1] > GridPoint1[1] and pose4[1] < GridPoint2[1]:
                        rhy = 1
                    elif pose4[1] > GridPoint2[1] and pose4[1] < GridPoint3[1]:
                        rhy = 2
                    elif pose4[1] > GridPoint3[1] and pose4[1] < GridPoint4[1]:
                        rhy = 3
                    else:
                        rhy = 0

                    if pose7[0] > GridPoint5[0] and pose7[0] < GridPoint1[0]:
                        lhx = 0
                    elif pose7[0] > GridPoint9[0] and pose7[0] < GridPoint5[0]:
                        lhx = 3
                    elif pose7[0] > GridPoint13[0] and pose7[0] < GridPoint9[0]:
                        lhx = 6
                    else:
                        lhx = 0

                    if pose7[1] > GridPoint1[1] and pose7[1] < GridPoint2[1]:
                        lhy = 1
                    elif pose7[1] > GridPoint2[1] and pose7[1] < GridPoint3[1]:
                        lhy = 2
                    elif pose7[1] > GridPoint3[1] and pose7[1] < GridPoint4[1]:
                        lhy = 3
                    else:
                        lhy = 0

                    lh = lhx + lhy
                    rh = rhx + rhy
                    self._column['lefthand'].append(lh)
                    self._column['righthand'].append(rh)

                elif self._orientation[i] == 'right':
                    GridPoint4 = (int(self._column['CTR_X'][i]), int(bottompoint))
                    GridPoint3 = (int(self._column['CTR_X'][i]), int(bottompoint - foot))
                    GridPoint2 = (int(self._column['CTR_X'][i]), int(bottompoint - foot - leg - bodylength))
                    GridPoint1 = (
                        int(self._column['CTR_X'][i]), int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint8 = (int(self._column['CTR_X'][i] + hand), int(bottompoint))
                    GridPoint7 = (int(self._column['CTR_X'][i] + hand), int(bottompoint - foot))
                    GridPoint6 = (int(self._column['CTR_X'][i] + hand), int(bottompoint - foot - leg - bodylength))
                    GridPoint5 = (
                        int(self._column['CTR_X'][i] + hand), int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint12 = (int(self._column['CTR_X'][i] + hand + arm), int(bottompoint))
                    GridPoint11 = (int(self._column['CTR_X'][i] + hand + arm), int(bottompoint - foot))
                    GridPoint10 = (
                        int(self._column['CTR_X'][i] + hand + arm), int(bottompoint - foot - leg - bodylength))
                    GridPoint9 = (
                        int(self._column['CTR_X'][i] + hand + arm),
                        int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint16 = (int(self._column['CTR_X'][i] + hand + arm + ref1), int(bottompoint))
                    GridPoint15 = (int(self._column['CTR_X'][i] + hand + arm + ref1), int(bottompoint - foot))
                    GridPoint14 = (
                        int(self._column['CTR_X'][i] + hand + arm + ref1), int(bottompoint - foot - leg - bodylength))
                    GridPoint13 = (int(self._column['CTR_X'][i] + hand + arm + ref1),
                                   int(bottompoint - foot - leg - bodylength - arm - hand))

                    if pose4[0] < GridPoint5[0] and pose4[0] > GridPoint1[0]:
                        rhx = 0
                    elif pose4[0] < GridPoint9[0] and pose4[0] > GridPoint5[0]:
                        rhx = 3
                    elif pose4[0] < GridPoint13[0] and pose4[0] > GridPoint9[0]:
                        rhx = 6
                    else:
                        rhx = 0

                    if pose4[1] > GridPoint1[1] and pose4[1] < GridPoint2[1]:
                        rhy = 1
                    elif pose4[1] > GridPoint2[1] and pose4[1] < GridPoint3[1]:
                        rhy = 2
                    elif pose4[1] > GridPoint3[1] and pose4[1] < GridPoint4[1]:
                        rhy = 3
                    else:
                        rhy = 0

                    if pose7[0] < GridPoint5[0] and pose7[0] > GridPoint1[0]:
                        lhx = 0
                    elif pose7[0] < GridPoint9[0] and pose7[0] > GridPoint5[0]:
                        lhx = 3
                    elif pose7[0] < GridPoint13[0] and pose7[0] > GridPoint9[0]:
                        lhx = 6
                    else:
                        lhx = 0

                    if pose7[1] > GridPoint1[1] and pose7[1] < GridPoint2[1]:
                        lhy = 1
                    elif pose7[1] > GridPoint2[1] and pose7[1] < GridPoint3[1]:
                        lhy = 2
                    elif pose7[1] > GridPoint3[1] and pose7[1] < GridPoint4[1]:
                        lhy = 3
                    else:
                        lhy = 0

                    lh = lhx + lhy
                    rh = rhx + rhy
                    self._column['lefthand'].append(lh)
                    self._column['righthand'].append(rh)
                else:
                    GridPoint1 = (0, 0)
                    GridPoint2 = (0, 0)
                    GridPoint3 = (0, 0)
                    GridPoint4 = (0, 0)
                    GridPoint5 = (0, 0)
                    GridPoint6 = (0, 0)
                    GridPoint7 = (0, 0)
                    GridPoint8 = (0, 0)
                    GridPoint9 = (0, 0)
                    GridPoint10 = (0, 0)
                    GridPoint11 = (0, 0)
                    GridPoint12 = (0, 0)
                    GridPoint13 = (0, 0)
                    GridPoint14 = (0, 0)
                    GridPoint15 = (0, 0)
                    GridPoint16 = (0, 0)

                    lh = 0
                    rh = 0
                    self._column['lefthand'].append(lh)
                    self._column['righthand'].append(rh)

                img = cv2.rectangle(img, (GridPoint1), (GridPoint6), color, 1)
                img = cv2.rectangle(img, (GridPoint2), (GridPoint7), color, 1)
                img = cv2.rectangle(img, (GridPoint3), (GridPoint8), color, 1)
                img = cv2.rectangle(img, (GridPoint5), (GridPoint10), color, 1)
                img = cv2.rectangle(img, (GridPoint6), (GridPoint11), color, 1)
                img = cv2.rectangle(img, (GridPoint7), (GridPoint12), color, 1)
                img = cv2.rectangle(img, (GridPoint9), (GridPoint14), color, 1)
                img = cv2.rectangle(img, (GridPoint10), (GridPoint15), color, 1)
                img = cv2.rectangle(img, (GridPoint11), (GridPoint16), color, 1)

                if lh != 0:
                    cv2.putText(img, 'Left Hand Grid:' +str(lh), (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), 2)
                else:
                    cv2.putText(img, 'Left Hand Grid: N/A', (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), 2)
                if rh != 0:
                    cv2.putText(img, 'Right Hand Grid:' +str(rh), (20, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), 2)
                else:
                    cv2.putText(img, 'Right Hand Grid: N/A', (20, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), 2)

                self._column['image'][i] = img
                i = i + 1

        return self._column['image']

##########################################################need combined with nine grid box###############################
    def Ninegridboxhandposition(self, framenum=-1):
        if framenum == -1:
            i = 0
            self._column['lefthand'] = []
            self._column['righthand'] = []
            while i < self._framenumber:

                pose0 = (int(self._column['B_0_X'][i]), int(self._column['B_0_Y'][i]))
                pose1 = (int(self._column['B_1_X'][i]), int(self._column['B_1_Y'][i]))
                pose2 = (int(self._column['B_2_X'][i]), int(self._column['B_2_Y'][i]))
                pose3 = (int(self._column['B_3_X'][i]), int(self._column['B_3_Y'][i]))
                pose4 = (int(self._column['B_4_X'][i]), int(self._column['B_4_Y'][i]))
                pose5 = (int(self._column['B_5_X'][i]), int(self._column['B_5_Y'][i]))
                pose6 = (int(self._column['B_6_X'][i]), int(self._column['B_6_Y'][i]))
                pose7 = (int(self._column['B_7_X'][i]), int(self._column['B_7_Y'][i]))
                pose8 = (int(self._column['B_8_X'][i]), int(self._column['B_8_Y'][i]))
                pose9 = (int(self._column['B_9_X'][i]), int(self._column['B_9_Y'][i]))
                pose10 = (int(self._column['B_10_X'][i]), int(self._column['B_10_Y'][i]))
                pose11 = (int(self._column['B_11_X'][i]), int(self._column['B_11_Y'][i]))
                pose12 = (int(self._column['B_12_X'][i]), int(self._column['B_12_Y'][i]))
                pose13 = (int(self._column['B_13_X'][i]), int(self._column['B_13_Y'][i]))
                pose14 = (int(self._column['B_14_X'][i]), int(self._column['B_14_Y'][i]))
                pose15 = (int(self._column['B_15_X'][i]), int(self._column['B_15_Y'][i]))
                pose16 = (int(self._column['B_16_X'][i]), int(self._column['B_16_Y'][i]))
                pose17 = (int(self._column['B_17_X'][i]), int(self._column['B_17_Y'][i]))

                if pose10[0] * pose9[0] + pose10[1] * pose9[1] == 0:
                    foot = np.linalg.norm(np.array(pose13) - np.array(pose12))
                elif pose13[0] * pose12[0] + pose13[1] * pose12[1] == 0:
                    foot = np.linalg.norm(np.array(pose10) - np.array(pose9))
                else:
                    foot = max(np.linalg.norm(np.array(pose10) - np.array(pose9)),
                               np.linalg.norm(np.array(pose13) - np.array(pose12)))

                if pose8[0] * pose9[0] + pose8[1] * pose9[1] == 0:
                    leg = np.linalg.norm(np.array(pose11) - np.array(pose12))
                elif pose11[0] * pose12[0] + pose11[1] * pose12[1] == 0:
                    leg = np.linalg.norm(np.array(pose8) - np.array(pose9))
                else:
                    leg = max(np.linalg.norm(np.array(pose8) - np.array(pose9)),
                              np.linalg.norm(np.array(pose11) - np.array(pose12)))

                if pose4[0] * pose3[0] + pose4[1] * pose3[1] == 0:
                    hand = np.linalg.norm(np.array(pose7) - np.array(pose6))
                elif pose7[0] * pose6[0] + pose7[1] * pose6[1] == 0:
                    hand = np.linalg.norm(np.array(pose4) - np.array(pose3))
                else:
                    hand = max(np.linalg.norm(np.array(pose4) - np.array(pose3)),
                               np.linalg.norm(np.array(pose7) - np.array(pose6)))

                if pose2[0] * pose3[0] + pose2[1] * pose3[1] == 0:
                    arm = np.linalg.norm(np.array(pose5) - np.array(pose6))
                elif pose5[0] * pose6[0] + pose5[1] * pose6[1] == 0:
                    arm = np.linalg.norm(np.array(pose2) - np.array(pose3))
                else:
                    arm = max(np.linalg.norm(np.array(pose2) - np.array(pose3)),
                              np.linalg.norm(np.array(pose5) - np.array(pose6)))

                bodylength = np.linalg.norm(
                    np.array(pose1) - np.array((self._column['CTR_X'][i], self._column['CTR_Y'][i])))

                ref1 = bodylength / 1.4142


                bottompoint = max(pose10[1], pose13[1])

                if self._orientation[i] == 'left':
                    GridPoint4 = (int(self._column['CTR_X'][i]), int(bottompoint))
                    GridPoint3 = (int(self._column['CTR_X'][i]), int(bottompoint - foot))
                    GridPoint2 = (int(self._column['CTR_X'][i]), int(bottompoint - foot - leg - bodylength))
                    GridPoint1 = (
                        int(self._column['CTR_X'][i]), int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint8 = (int(self._column['CTR_X'][i] - hand), int(bottompoint))
                    GridPoint7 = (int(self._column['CTR_X'][i] - hand), int(bottompoint - foot))
                    GridPoint6 = (int(self._column['CTR_X'][i] - hand), int(bottompoint - foot - leg - bodylength))
                    GridPoint5 = (
                        int(self._column['CTR_X'][i] - hand), int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint12 = (int(self._column['CTR_X'][i] - hand - arm), int(bottompoint))
                    GridPoint11 = (int(self._column['CTR_X'][i] - hand - arm), int(bottompoint - foot))
                    GridPoint10 = (
                        int(self._column['CTR_X'][i] - hand - arm), int(bottompoint - foot - leg - bodylength))
                    GridPoint9 = (
                        int(self._column['CTR_X'][i] - hand - arm),
                        int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint16 = (int(self._column['CTR_X'][i] - hand - arm - ref1), int(bottompoint))
                    GridPoint15 = (int(self._column['CTR_X'][i] - hand - arm - ref1), int(bottompoint - foot))
                    GridPoint14 = (
                        int(self._column['CTR_X'][i] - hand - arm - ref1), int(bottompoint - foot - leg - bodylength))
                    GridPoint13 = (int(self._column['CTR_X'][i] - hand - arm - ref1),
                                   int(bottompoint - foot - leg - bodylength - arm - hand))

                    if pose4[0] > GridPoint5[0] and pose4[0] < GridPoint1[0]:
                        rhx = 0
                    elif pose4[0] > GridPoint9[0] and pose4[0] < GridPoint5[0]:
                        rhx = 3
                    elif pose4[0] > GridPoint13[0] and pose4[0] < GridPoint9[0]:
                        rhx = 6
                    else:
                        rhx = 0

                    if pose4[1] > GridPoint1[1] and pose4[1] < GridPoint2[1]:
                        rhy = 1
                    elif pose4[1] > GridPoint2[1] and pose4[1] < GridPoint3[1]:
                        rhy = 2
                    elif pose4[1] > GridPoint3[1] and pose4[1] < GridPoint4[1]:
                        rhy = 3
                    else:
                        rhy = 0

                    if pose7[0] > GridPoint5[0] and pose7[0] < GridPoint1[0]:
                        lhx = 0
                    elif pose7[0] > GridPoint9[0] and pose7[0] < GridPoint5[0]:
                        lhx = 3
                    elif pose7[0] > GridPoint13[0] and pose7[0] < GridPoint9[0]:
                        lhx = 6
                    else:
                        lhx = 0

                    if pose7[1] > GridPoint1[1] and pose7[1] < GridPoint2[1]:
                        lhy = 1
                    elif pose7[1] > GridPoint2[1] and pose7[1] < GridPoint3[1]:
                        lhy = 2
                    elif pose7[1] > GridPoint3[1] and pose7[1] < GridPoint4[1]:
                        lhy = 3
                    else:
                        lhy = 0

                    lh = lhx + lhy
                    rh = rhx + rhy
                    self._column['lefthand'].append(lh)
                    self._column['righthand'].append(rh)

                elif self._orientation[i] == 'right':
                    GridPoint4 = (int(self._column['CTR_X'][i]), int(bottompoint))
                    GridPoint3 = (int(self._column['CTR_X'][i]), int(bottompoint - foot))
                    GridPoint2 = (int(self._column['CTR_X'][i]), int(bottompoint - foot - leg - bodylength))
                    GridPoint1 = (
                        int(self._column['CTR_X'][i]), int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint8 = (int(self._column['CTR_X'][i] + hand), int(bottompoint))
                    GridPoint7 = (int(self._column['CTR_X'][i] + hand), int(bottompoint - foot))
                    GridPoint6 = (int(self._column['CTR_X'][i] + hand), int(bottompoint - foot - leg - bodylength))
                    GridPoint5 = (
                        int(self._column['CTR_X'][i] + hand), int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint12 = (int(self._column['CTR_X'][i] + hand + arm), int(bottompoint))
                    GridPoint11 = (int(self._column['CTR_X'][i] + hand + arm), int(bottompoint - foot))
                    GridPoint10 = (
                        int(self._column['CTR_X'][i] + hand + arm), int(bottompoint - foot - leg - bodylength))
                    GridPoint9 = (
                        int(self._column['CTR_X'][i] + hand + arm),
                        int(bottompoint - foot - leg - bodylength - arm - hand))
                    GridPoint16 = (int(self._column['CTR_X'][i] + hand + arm + ref1), int(bottompoint))
                    GridPoint15 = (int(self._column['CTR_X'][i] + hand + arm + ref1), int(bottompoint - foot))
                    GridPoint14 = (
                        int(self._column['CTR_X'][i] + hand + arm + ref1), int(bottompoint - foot - leg - bodylength))
                    GridPoint13 = (int(self._column['CTR_X'][i] + hand + arm + ref1),
                                   int(bottompoint - foot - leg - bodylength - arm - hand))

                    if pose4[0] < GridPoint5[0] and pose4[0] > GridPoint1[0]:
                        rhx = 0
                    elif pose4[0] < GridPoint9[0] and pose4[0] > GridPoint5[0]:
                        rhx = 3
                    elif pose4[0] < GridPoint13[0] and pose4[0] > GridPoint9[0]:
                        rhx = 6
                    else:
                        rhx = 0

                    if pose4[1] > GridPoint1[1] and pose4[1] < GridPoint2[1]:
                        rhy = 1
                    elif pose4[1] > GridPoint2[1] and pose4[1] < GridPoint3[1]:
                        rhy = 2
                    elif pose4[1] > GridPoint3[1] and pose4[1] < GridPoint4[1]:
                        rhy = 3
                    else:
                        rhy = 0

                    if pose7[0] < GridPoint5[0] and pose7[0] > GridPoint1[0]:
                        lhx = 0
                    elif pose7[0] < GridPoint9[0] and pose7[0] > GridPoint5[0]:
                        lhx = 3
                    elif pose7[0] < GridPoint13[0] and pose7[0] > GridPoint9[0]:
                        lhx = 6
                    else:
                        lhx = 0

                    if pose7[1] > GridPoint1[1] and pose7[1] < GridPoint2[1]:
                        lhy = 1
                    elif pose7[1] > GridPoint2[1] and pose7[1] < GridPoint3[1]:
                        lhy = 2
                    elif pose7[1] > GridPoint3[1] and pose7[1] < GridPoint4[1]:
                        lhy = 3
                    else:
                        lhy = 0

                    lh = lhx + lhy
                    rh = rhx + rhy
                    self._column['lefthand'].append(lh)
                    self._column['righthand'].append(rh)
                else:
                    GridPoint1 = (0, 0)
                    GridPoint2 = (0, 0)
                    GridPoint3 = (0, 0)
                    GridPoint4 = (0, 0)
                    GridPoint5 = (0, 0)
                    GridPoint6 = (0, 0)
                    GridPoint7 = (0, 0)
                    GridPoint8 = (0, 0)
                    GridPoint9 = (0, 0)
                    GridPoint10 = (0, 0)
                    GridPoint11 = (0, 0)
                    GridPoint12 = (0, 0)
                    GridPoint13 = (0, 0)
                    GridPoint14 = (0, 0)
                    GridPoint15 = (0, 0)
                    GridPoint16 = (0, 0)

                    lh = 0
                    rh = 0
                    self._column['lefthand'].append(lh)
                    self._column['righthand'].append(rh)

                i = i + 1

########################################################################################################################

    def CalOperator(self, framenum=-1, height=5.8):
        operator = {}
        if framenum == -1:
            '''
            self._operator['leg']=[]
            self._operator['knee']=[]
            self._operator['arm']=[]
            self._operator['elbow']=[]
            self._operator['torso']=[]
            self._operator['shoulderheight']=[]
            '''
            leglen = []
            kneelen = []
            armlen = []
            elbowhei = []
            torsolen = []
            shoulder = []
            realheight = []
            forearm = []

            i = 0
            while i < self._framenumber:

                pose0 = (int(self._column['B_0_X'][i]), int(self._column['B_0_Y'][i]))
                pose1 = (int(self._column['B_1_X'][i]), int(self._column['B_1_Y'][i]))
                pose2 = (int(self._column['B_2_X'][i]), int(self._column['B_2_Y'][i]))
                pose3 = (int(self._column['B_3_X'][i]), int(self._column['B_3_Y'][i]))
                pose4 = (int(self._column['B_4_X'][i]), int(self._column['B_4_Y'][i]))
                pose5 = (int(self._column['B_5_X'][i]), int(self._column['B_5_Y'][i]))
                pose6 = (int(self._column['B_6_X'][i]), int(self._column['B_6_Y'][i]))
                pose7 = (int(self._column['B_7_X'][i]), int(self._column['B_7_Y'][i]))
                pose8 = (int(self._column['B_8_X'][i]), int(self._column['B_8_Y'][i]))
                pose9 = (int(self._column['B_9_X'][i]), int(self._column['B_9_Y'][i]))
                pose10 = (int(self._column['B_10_X'][i]), int(self._column['B_10_Y'][i]))
                pose11 = (int(self._column['B_11_X'][i]), int(self._column['B_11_Y'][i]))
                pose12 = (int(self._column['B_12_X'][i]), int(self._column['B_12_Y'][i]))
                pose13 = (int(self._column['B_13_X'][i]), int(self._column['B_13_Y'][i]))
                pose14 = (int(self._column['B_14_X'][i]), int(self._column['B_14_Y'][i]))
                pose15 = (int(self._column['B_15_X'][i]), int(self._column['B_15_Y'][i]))
                pose16 = (int(self._column['B_16_X'][i]), int(self._column['B_16_Y'][i]))
                pose17 = (int(self._column['B_17_X'][i]), int(self._column['B_17_Y'][i]))

                if pose10[0] * pose9[0] + pose10[1] * pose9[1] == 0:
                    foot = np.linalg.norm(np.array(pose13) - np.array(pose12))
                elif pose13[0] * pose12[0] + pose13[1] * pose12[1] == 0:
                    foot = np.linalg.norm(np.array(pose10) - np.array(pose9))
                else:
                    foot = max(np.linalg.norm(np.array(pose10) - np.array(pose9)),
                               np.linalg.norm(np.array(pose13) - np.array(pose12)))

                if pose8[0] * pose9[0] + pose8[1] * pose9[1] == 0:
                    leg = np.linalg.norm(np.array(pose11) - np.array(pose12))
                elif pose11[0] * pose12[0] + pose11[1] * pose12[1] == 0:
                    leg = np.linalg.norm(np.array(pose8) - np.array(pose9))
                else:
                    leg = max(np.linalg.norm(np.array(pose8) - np.array(pose9)),
                              np.linalg.norm(np.array(pose11) - np.array(pose12)))

                if pose4[0] * pose3[0] + pose4[1] * pose3[1] == 0:
                    hand = np.linalg.norm(np.array(pose7) - np.array(pose6))
                elif pose7[0] * pose6[0] + pose7[1] * pose6[1] == 0:
                    hand = np.linalg.norm(np.array(pose4) - np.array(pose3))
                else:
                    hand = max(np.linalg.norm(np.array(pose4) - np.array(pose3)),
                               np.linalg.norm(np.array(pose7) - np.array(pose6)))

                if pose2[0] * pose3[0] + pose2[1] * pose3[1] == 0:
                    arm = np.linalg.norm(np.array(pose5) - np.array(pose6)) # arm is back arm hand is forearm
                elif pose5[0] * pose6[0] + pose5[1] * pose6[1] == 0:
                    arm = np.linalg.norm(np.array(pose2) - np.array(pose3))
                else:
                    arm = max(np.linalg.norm(np.array(pose2) - np.array(pose3)),
                              np.linalg.norm(np.array(pose5) - np.array(pose6)))

                bodylength = np.linalg.norm(
                    np.array(pose1) - np.array((self._column['CTR_X'][i], self._column['CTR_Y'][i])))

                pixelheight = leg + foot + bodylength + arm
                if pixelheight != 0:
                    ratio = height / pixelheight
                    realkneelength = foot * ratio
                    realleglength = leg * ratio
                    realtorsolength = bodylength * ratio
                    realarmlength = arm * ratio
                    realelbowheight = (bodylength+leg+foot-arm) * ratio
                    realshoulderheight = (bodylength + leg + foot) * ratio
                    realforearmlength = hand * ratio
                else:
                    realkneelength = 0
                    realleglength = 0
                    realtorsolength = 0
                    realarmlength = 0
                    realelbowheight = 0
                    realshoulderheight = 0
                    realforearmlength = 0

                if realleglength != 0: leglen.append(realleglength)
                if realkneelength != 0: kneelen.append(realkneelength)
                if realtorsolength != 0: torsolen.append(realtorsolength)
                if realarmlength != 0: armlen.append(realarmlength)
                if realelbowheight != 0: elbowhei.append(realelbowheight)
                if realshoulderheight != 0: shoulder.append(realshoulderheight)
                if realforearmlength !=0: forearm.append(realforearmlength)
                realheight.append(realshoulderheight + realarmlength)

                i = i + 1

            operator['shoulder_height'] = sum(shoulder) / len(shoulder)
            operator['knee_height'] = sum(kneelen) / len(kneelen)
            operator['leg_length'] = sum(leglen) / len(leglen)
            operator['arm_length'] = sum(armlen) / len(armlen)
            operator['elbow_length'] = sum(elbowhei) / len(elbowhei)
            operator['torso_height'] = sum(torsolen) / len(torsolen)
            operator['forearm_height']= sum(forearm) / len(forearm)

        return operator

    def savesqldata(self, host, port, user, password, db, table_name, video_name, assessment_id ,person_number=0):
        import uuid
        import datetime

        print('***************In  savesqldata')
        conn = pymysql.connect(host=host, port=port, user=user, passwd=password,
                               db=db)
        #assessment_id, task_id = video_name.split('_')
        cur = conn.cursor()
        sql = "delete from %s where assessment_id = %s" % (table_name, assessment_id)
        try:
            print('***************{}************\n'.format(sql))
            cur.execute(sql)
            conn.commit()
        except:
            conn.rollback()
        save_buffer = []
        x = self.calNecktime()
        x.update(self.calLEtime())
        x.update(self.calREtime())
        x.update(self.calLStime())
        x.update(self.calRStime())
        x.update(self.calBtime())
        x.update(self.legaction())
        g = 1
        print(x)

        for i in x:
            sql = "insert into %s.%s (assessment_id,posture_id, time, count, static_length, posture_key) values (%s,%s,'%s',%s, %s, '%s');" % (
                db, table_name,
                assessment_id, i, x[i][0],x[i][1], x[i][2], uuid.uuid4())
            save_buffer.append(sql)
            g = g + 1

        for r in save_buffer:
            try:
                print('***************{}************\n'.format(r))
                cur.execute(r)
                conn.commit()
            except:
                conn.rollback()

        videolength = datetime.timedelta(seconds=(self._framenumber / 10)-1)

        cur.execute(
            "update  %s.data_video set video_length = '%s' where video_name like '%%%s%%'" % (db, videolength, video_name))
        conn.commit()

        print('***************out  savesqldata')
        conn.close()

    def savehandling(self, host, port, user, password, db, table_name, video_name, video_path):
        import uuid
        self.Ninegridboxhandposition()
        conn = pymysql.connect(host=host, port=port, user=user, password=password,
                               db=db)

        cur = conn.cursor()

        cur.execute("select assessment_id, action_id, start_time, end_time, total_time from ergo_raw.data_action where action_id in (select id from ergo_raw.ref_action_type where action_name in \
                    ('Lifting', 'Pushing', 'Pulling', 'Carrying', 'Reaching', 'Handling') and assessment_id=(select assessment_id from ergo_raw.data_video \
                    where video_name like '%%%s%%'));" % (video_name))
        field_names = [i[0] for i in cur.description]
        field_names.extend(['horizontal_orig','horizontal_dest','vertical_orig','vertical_dest','vertical_height','back_angle','zone_left_orig','zone_right_orig','zone_left_dest','zone_right_dest'])
        writedic = {}
        for h in field_names:
            writedic[h] = []

        for row in cur.fetchall():
            print(row)
            for h, v in zip(field_names, row):
                writedic[h].append(v)

        writedic['handling_id'] = writedic.pop('action_id')
        print(writedic)

        print(len(writedic['assessment_id']))
        save_buffer = []


        for t in range(len(writedic['assessment_id'])):
            fps=self.Getfps(video_path)
            startframe=int(writedic['start_time'][t].total_seconds()*fps)
            endframe=int(writedic['end_time'][t].total_seconds()*fps)-1
            if endframe >= self._framenumber:
                endframe=self._framenumber-1
            writedic['horizontal_orig'].append(self.Calhorizontaldistance(framenum=startframe))
            verticalorig=float(self.Calverticaldistance(framenum=startframe))
            verticaldest=float(self.Calverticaldistance(framenum=endframe))
            writedic['vertical_orig'].append(verticalorig)
            writedic['horizontal_dest'].append(self.Calverticaldistance(framenum=endframe))
            writedic['vertical_dest'].append(verticaldest)
            writedic['vertical_height'].append(abs(verticaldest-verticalorig))
            totalbackangle=self.CalBackAngle()
            sum=0
            for x in range(startframe,endframe):
                sum=sum+totalbackangle[x]
            averbackangle=sum/(endframe-startframe)
            writedic['back_angle'].append(averbackangle)
            writedic['zone_left_orig'].append(self._column['lefthand'][startframe])
            writedic['zone_left_dest'].append(self._column['lefthand'][endframe])
            writedic['zone_right_orig'].append(self._column['righthand'][startframe])
            writedic['zone_right_dest'].append(self._column['righthand'][endframe])

        print(writedic)

        for r in range(len(writedic['assessment_id'])):
            print(writedic['start_time'][r].total_seconds())
            sql = "insert into %s.%s (assessment_id, handling_id, start_time, end_time, total_time, horizontal_orig, \
            horizontal_dest,vertical_orig, vertical_dest, vertical_height, back_angle, distance, zone_left_orig, zone_right_orig,\
            zone_left_dest,zone_right_dest, handling_key) values (%s,%s,'%s','%s','%s',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'%s');" % (
                db, table_name, writedic['assessment_id'][r], writedic['handling_id'][r], writedic['start_time'][r],
                writedic['end_time'][r], writedic['total_time'][r], writedic['horizontal_orig'][r],
                writedic['horizontal_dest'][r],
                writedic['vertical_orig'][r], writedic['vertical_dest'][r], writedic['vertical_height'][r],
                writedic['back_angle'][r],
                abs(writedic['vertical_orig'][r] - writedic['vertical_dest'][r]),
                writedic['zone_left_orig'][r], writedic['zone_right_orig'][r], writedic['zone_left_dest'][r],
                writedic['zone_right_dest'][r],
                uuid.uuid4())
            save_buffer.append(sql)

        for g in save_buffer:
            try:
                print('***************{}************\n'.format(g))
                cur.execute(g)
                conn.commit()
            except:
                conn.rollback()

        print('***************out  savesqldata')

        conn.close()

    def saveoperator(self, host, port, user, password, db, table_name, video_name):

        conn = pymysql.connect(host=host, port=port, user=user, passwd=password,
                               db=db)

        cur = conn.cursor()
        # sql_str="select height from ergo_raw.data_operator where id in (select operator_id FROM ergo_raw.data_assessment where id=(select assessment_id from ergo_raw.data_video where video_name='20181022222552_A00747178673')"

        cur.execute(
            "select height, id from ergo_raw.data_operator where id in (select operator_id FROM ergo_raw.data_assessment where id=(select assessment_id from ergo_raw.data_video where video_name like '%%%s%%'))" % (
                video_name))
        t = []
        for e in cur.fetchone():
            t.append(e)

        realheight = float(t[0])

        # print(float(cur.fetchone()[1])
        # realheight= [i[1] for i in cur.description]
        # print(realheight)

        save_buffer = []

        x = self.CalOperator(height=realheight)

        for r in x:
            sql = "update  %s.%s set %s = '%s' where id=%d" % (
                db, table_name, r, x[r], t[1])
            save_buffer.append(sql)

        print(save_buffer)

        for g in save_buffer:
            try:
                cur.execute(g)
                conn.commit()
            except:
                conn.rollback()

        conn.close()

    def Writecsv(self, path, videoname):
        x = {}
        i = 0
        videoname = videoname
        x['video_name'] = []
        x['frame_number'] = []
        x['neck_angle'] = []
        x['left_shoulder_angle'] = []
        x['right_shoulder_angle'] = []
        x['back_angle'] = []
        x['left_elbow_angle'] = []
        x['right_elbow_angle'] = []
        x['horizontal_distance']=[]
        x['vertical_distance']=[]
        while i < self._framenumber:
            x['video_name'].append(videoname)
            x['frame_number'].append(i)
            x['neck_angle'].append(self.CalNeckAngle(framenum=i))
            x['left_shoulder_angle'].append(self.CalLeftShoulderAngle(framenum=i))
            x['right_shoulder_angle'].append(self.CalRightShoulderAngle(framenum=i))
            x['back_angle'].append(self.CalBackAngle(framenum=i))
            x['left_elbow_angle'].append(self.CalLeftElbow(framenum=i))
            x['right_elbow_angle'].append(self.CalRightElbow(framenum=i))
            x['horizontal_distance'].append(self.Calhorizontaldistance(framenum=i))
            x['vertical_distance'].append(self.Calverticaldistance(framenum=i))
            i = i + 1
        dataframe = pd.DataFrame(x)
        dataframe.to_csv('%s/%s.csv' % (path, videoname), index=False, sep=",")

    def ImagetoVideo(self, output, imgext='jpg', fps=10.0):
        import skvideo.io
        writer = skvideo.io.FFmpegWriter(output, inputdict={
            '-r': '10'}, outputdict={'-b': '300000000'})
        for img in self._column['image']:
            img = img[:, :, ::-1]
            # data = np.asarray(img)
            # img = Image.fromarray(np.roll(data, 1, axis=-1))
            writer.writeFrame(img)
        writer.close()

        # if self._column['image'] !=[]:
        #    i=0
        #    frame = self._column['image'][i]
        #    print(len(self._column['image']))
        #    height, width, channels = frame.shape
        #    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
        #    while i < len(self._column['image'])-1:
        #        i=i+1
        #        frame = self._column['image'][i]
        #        out.write(frame)

        # out.release()

        print("The output video is {}".format(output))

    def ImagetoVideobycv(self, output, imgext='jpg', fps=10.0):
        if self._column['image'] != []:
            i = 0
            frame = self._column['image'][i]
            print(len(self._column['image']))
            height, width, channels = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output, fourcc, fps, (width, height))
            while i < len(self._column['image']) - 1:
                i = i + 1
                frame = self._column['image'][i]
                out.write(frame)

        out.release()

        print("The output video is {}".format(output))

    def outputimage(self,dirpath):
        flag=0
        x=[]
        counter1=0
        counter2=0
        selectedact=['carrying','reaching','handling','pushing','lifting','pulling']
        randomnum=np.random.randint(0,self._framenumber-1)
        while flag <3:
            if self._column['LABEL_NAME'][randomnum].lower() in selectedact:
                x.append(self._column['image'][randomnum])
                #x.append(self._column['LABEL_NAME'][randomnum])
                print(randomnum)
                flag=flag+1
            else:
                counter1=counter1+1

            if counter1>10000:
                break


            newrandomnum = np.random.randint(0, self._framenumber-1)
            while newrandomnum == randomnum:
                newrandomnum = np.random.randint(0, self._framenumber-1)
                counter2=counter2+1
                if counter2>10000:
                    break
            randomnum=newrandomnum

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            
        print(x)
        print(len(x))
        u=3
        flag2=0
        while flag2<(u-len(x)):
            randomnum=np.random.randint(0,self._framenumber-1)
            x.append(self._column['image'][randomnum])
            flag2=flag2+1
            
        for i in range(0,u):
            cv2.imwrite("%s/image%d.jpg"%(dirpath,i),x[i])


    def neckstatus(self):
        x=[]
        x=self.CalNeckAngle()
        y={}
        count=0
        a=[]
        b=[]
        c=[]
        d=[]
        for i in range(0,len(x),10):
            z=x[i:i+10]
            j=0
            total=0
            print(z)
            while j<len(z):
                print(j)
                print(z[j])


                if z[j]==0:
                    del z[j]
                else:
                    total=total+z[j]
                    j=j+1
            if j!=0:
                y[count]=total/j
                count=count+1
        for key, value in y.items():
            if value <= 20:
                a.append(key)
            elif value <30:
                b.append(key)
            elif value <45:
                c.append(key)
            else:
                d.append(key)

        print(a)
        print(b)
        print(c)
        print(d)

        repetition1=0
        repetition2=0
        repetition3=0
        repetition4=0
        if len(a)>1:
            for i in range(0, len(a)-1):
                if abs(a[i]-a[i+1])>1:
                    repetition1=repetition1+1

        if len(b)>1:
            for i in range(0, len(b)-1):
                if abs(b[i]-b[i+1])>1:
                    repetition2=repetition2+1

        if len(c)>1:
            for i in range(0, len(c)-1):
                if abs(c[i]-c[i+1])>1:
                    repetition3=repetition3+1

        if len(d)>1:
            for i in range(0, len(d)-1):
                if abs(d[i]-d[i+1])>1:
                    repetition4=repetition4+1

        u={}
        u['repetitionrange1'] = repetition1
        u['repetitionrange2'] = repetition2
        u['repetitionrange3'] = repetition3
        u['repetitionrange4'] = repetition4

        return u

    def leftelbowstatus(self):
        x=[]
        x=self.CalLeftElbow()
        y={}
        count=0
        a=[]
        b=[]
        c=[]

        for i in range(0,len(x),10):
            z=x[i:i+10]
            j=0
            total=0

            while j<len(z):
                if z[j]==0:
                    del z[j]
                else:
                    total=total+z[j]
                    j=j+1
            if j!=0:
                y[count]=total/j
                count=count+1
        for key, value in y.items():
            if value < 60:
                a.append(key)
            elif value <=100:
                b.append(key)
            else:
                c.append(key)


        repetition1=0
        repetition2=0
        repetition3=0

        if len(a)>1:
            for i in range(0, len(a)-1):
                if abs(a[i]-a[i+1])>1:
                    repetition1=repetition1+1

        if len(b)>1:
            for i in range(0, len(b)-1):
                if abs(b[i]-b[i+1])>1:
                    repetition2=repetition2+1

        if len(c)>1:
            for i in range(0, len(c)-1):
                if abs(c[i]-c[i+1])>1:
                    repetition3=repetition3+1


        u={}
        u['repetitionrange1'] = repetition1
        u['repetitionrange2'] = repetition2
        u['repetitionrange3'] = repetition3

        return u

    def rightelbowstatus(self):
        x = []
        x = self.CalRightElbow()
        y = {}
        count = 0
        a = []
        b = []
        c = []

        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            j = 0
            total = 0

            while j < len(z):
                if z[j] == 0:
                    del z[j]
                else:
                    total = total + z[j]
                    j = j + 1
            if j != 0:
                y[count] = total / j
                count = count + 1
        for key, value in y.items():
            if value < 60:
                a.append(key)
            elif value <= 100:
                b.append(key)
            else:
                c.append(key)

        repetition1 = 0
        repetition2 = 0
        repetition3 = 0

        if len(a) > 1:
            for i in range(0, len(a) - 1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1

        if len(b) > 1:
            for i in range(0, len(b) - 1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c) - 1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        u = {}
        u['repetitionrange1'] = repetition1
        u['repetitionrange2'] = repetition2
        u['repetitionrange3'] = repetition3

        return u

    def backstatus(self):
        x = []
        x = self.CalBackAngle()
        y = {}
        count = 0
        a = []
        b = []
        c = []
        d = []
        e=[]
        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            j = 0
            total = 0

            while j < len(z):


                if z[j] == 0:
                    del z[j]
                else:
                    total = total + z[j]
                    j = j + 1
            if j != 0:
                y[count] = total / j
                count = count + 1
        for key, value in y.items():
            if value < 30:
                a.append(key)
            elif value < 45:
                b.append(key)
            elif value < 60:
                c.append(key)
            elif value <90:
                d.append(key)
            else:
                e.append(key)


        repetition1 = 0
        repetition2 = 0
        repetition3 = 0
        repetition4 = 0
        repetition5 = 0
        if len(a) > 1:
            for i in range(0, len(a) - 1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1

        if len(b) > 1:
            for i in range(0, len(b) - 1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c) - 1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        if len(d) > 1:
            for i in range(0, len(d) - 1):
                if abs(d[i] - d[i + 1]) > 1:
                    repetition4 = repetition4 + 1

        if len(e) > 1:
            for i in range(0, len(e) - 1):
                if abs(e[i] - e[i + 1]) > 1:
                    repetition5 = repetition5 + 1

        u = {}
        u['repetitionrange1'] = repetition1
        u['repetitionrange2'] = repetition2
        u['repetitionrange3'] = repetition3
        u['repetitionrange4'] = repetition4
        u['repetitionrange5'] = repetition5


        return u

    def leftshoulderstatus(self):
        x = []
        x = self.CalLeftShoulderAngle()
        y = {}
        count = 0
        a = []
        b = []
        c = []
        d = []
        e = []
        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            j = 0
            total = 0

            while j < len(z):

                if z[j] == 0:
                    del z[j]
                else:
                    total = total + z[j]
                    j = j + 1
            if j != 0:
                y[count] = total / j
                count = count + 1
        for key, value in y.items():
            if value < 30:
                a.append(key)
            elif value < 60:
                b.append(key)
            elif value < 90:
                c.append(key)
            elif value < 120:
                d.append(key)
            else:
                e.append(key)

        repetition1 = 0
        repetition2 = 0
        repetition3 = 0
        repetition4 = 0
        repetition5 = 0
        if len(a) > 1:
            for i in range(0, len(a) - 1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1

        if len(b) > 1:
            for i in range(0, len(b) - 1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c) - 1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        if len(d) > 1:
            for i in range(0, len(d) - 1):
                if abs(d[i] - d[i + 1]) > 1:
                    repetition4 = repetition4 + 1

        if len(e) > 1:
            for i in range(0, len(e) - 1):
                if abs(e[i] - e[i + 1]) > 1:
                    repetition5 = repetition5 + 1

        u = {}
        u['repetitionrange1'] = repetition1
        u['repetitionrange2'] = repetition2
        u['repetitionrange3'] = repetition3
        u['repetitionrange4'] = repetition4
        u['repetitionrange5'] = repetition5

        return u

    def rightshoulderstatus(self):
        x = []
        x = self.CalRightShoulderAngle()
        y = {}
        count = 0
        a = []
        b = []
        c = []
        d = []
        e = []
        for i in range(0, len(x), 10):
            z = x[i:i + 10]
            j = 0
            total = 0

            while j < len(z):

                if z[j] == 0:
                    del z[j]
                else:
                    total = total + z[j]
                    j = j + 1
            if j != 0:
                y[count] = total / j
                count = count + 1
        for key, value in y.items():
            if value < 30:
                a.append(key)
            elif value < 60:
                b.append(key)
            elif value < 90:
                c.append(key)
            elif value < 120:
                d.append(key)
            else:
                e.append(key)

        repetition1 = 0
        repetition2 = 0
        repetition3 = 0
        repetition4 = 0
        repetition5 = 0
        if len(a) > 1:
            for i in range(0, len(a) - 1):
                if abs(a[i] - a[i + 1]) > 1:
                    repetition1 = repetition1 + 1

        if len(b) > 1:
            for i in range(0, len(b) - 1):
                if abs(b[i] - b[i + 1]) > 1:
                    repetition2 = repetition2 + 1

        if len(c) > 1:
            for i in range(0, len(c) - 1):
                if abs(c[i] - c[i + 1]) > 1:
                    repetition3 = repetition3 + 1

        if len(d) > 1:
            for i in range(0, len(d) - 1):
                if abs(d[i] - d[i + 1]) > 1:
                    repetition4 = repetition4 + 1

        if len(e) > 1:
            for i in range(0, len(e) - 1):
                if abs(e[i] - e[i + 1]) > 1:
                    repetition5 = repetition5 + 1

        u = {}
        u['repetitionrange1'] = repetition1
        u['repetitionrange2'] = repetition2
        u['repetitionrange3'] = repetition3
        u['repetitionrange4'] = repetition4
        u['repetitionrange5'] = repetition5

        return u




