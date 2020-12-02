#!/usr/bin/env python

import numpy as np
import cv2
import os
import sys, time
from scipy.ndimage import filters
import roslib
import rospy
from sensor_msgs.msg import CompressedImage

VERBOSE=False

class image_feature:
    def __init__(self, path):
        '''Initialize ros publisher, ros subscriber'''
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage, queue_size=500)
            
        self.path = path
        self.CreateMatches()
        self.subscriber = rospy.Subscriber("/camera/color/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print "subscribed to /camera/color/image_raw/compressed"

    def findDes(self, images):
        desList=[]
        kpList=[]
        for img in images:
            kp,des = self.orb.detectAndCompute(img,None)
            desList.append(des)  
            kpList.append(kp)
            imgKp = cv2.drawKeypoints(img,kp,None)            
        return kpList, desList
     
    def CreateMatches(self):
        '''Creates Matches'''        
        self.orb = cv2.ORB_create(nfeatures=1000)
        #### Import Images
        images = []
        self.classNames= []
        myList = os.listdir(self.path)        
        for cl in myList:
            imgCur = cv2.imread(self.path + '/' + cl, 0) 
            images.append(imgCur)
            self.classNames.append(os.path.splitext(cl)[0])
        
        print('Images')
        print(images)
        self.desList = self.findDes(images)
        if VERBOSE :
            print('Total Classes Detected', len(myList))
            print(self.classNames)                   
            print(len(self.desList))

    def findID(self, img, desList,thres=15):
        kp2,des2 = self.orb.detectAndCompute(img,None)
        bf = cv2.BFMatcher()
        matchList=[]
        finalVal = -1
        try:
            for des in desList:
                matches = bf.knnMatch(des, des2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                matchList.append(len(good))
        except:
            pass
        # print(matchList)
        if len(matchList)!=0:
            if max(matchList) > thres:
                finalVal = matchList.index(max(matchList))
        return finalVal

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        image_g = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        id = self.findID(image_g,self.desList)
        if id != -1:
            cv2.putText(image_np,self.classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
 
        cv2.imshow('img2',image_np)
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        cv2.waitKey(1)       
        
        #### Create and Publish CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)        
        #self.subscriber.unregister()


def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('image_feature', anonymous=True)
    ic = image_feature('/home/rosmatch/catkin_ws_hartmann/src/Mir200_Sim/mir_vision/classes')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

