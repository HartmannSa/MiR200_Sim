#!/usr/bin/env python

import numpy as np
import cv2
import os
import sys, time
from scipy.ndimage import filters
import roslib
import rospy
from sensor_msgs.msg import CompressedImage

VERBOSE=True

class object_detection:
    features = {'ORB','SIFT', 'SURF', 'STAR', 'BRISK', 'AKAZE', 'KAZE'}

    def __init__(self, path, detector='ORB'):
        '''Initialize ros publisher, ros subscriber'''
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage, queue_size=500)
            
        self.path = path            # Path to class_images/ Trainimages
        self.detector = detector    # Which Feature Detector is used
        self.CreateMatches()        # Detect Features in Trainimages
        self.subscriber = rospy.Subscriber("/camera/color/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print "subscribed to /camera/color/image_raw/compressed"

    def findDes(self, images):
        if (self.detector=='ORB'):        
            self.det = cv2.ORB_create(nfeatures=1000)
        elif (self.detector=='SIFT'):
            self.det = cv2.xfeatures2d.SIFT_create()
        elif (self.detector=='SURF'):
            self.det = cv2.xfeatures2d.SURF_create()
        else:
            print('\n########################################################')
            print('Detector ' + self.detector + ' is not implemented yet')
            print('########################################################\n')
            pass          
        desList=[]
        kpList=[]
        for img in images:
            kp,des = self.det.detectAndCompute(img,None)
            desList.append(des)  
            kpList.append(kp)            
        return kpList, desList
     
    def CreateMatches(self):
        '''To creates Matches features in Trainimages needs to be detected'''        
        # Generate classNames from Trainimages
        self.images_train = []
        self.classNames= []
        myList = os.listdir(self.path)        
        for cl in myList:
            imgCur = cv2.imread(self.path + '/' + cl, 0) 
            self.images_train.append(imgCur)
            self.classNames.append(os.path.splitext(cl)[0])
        if VERBOSE :
            print('Total Classes', len(myList))
            print(self.classNames) 
        # Generate Descriptors from Trainimages
        self.kpList_Train, self.desList_Train = self.findDes(self.images_train)
        # print(len(self.kpList_Train[0]))
        if VERBOSE :              
            print('Number of Descriptors', len(self.desList_Train))

    def findClassID(self, img,thres=20):
        kp2,des2 = self.det.detectAndCompute(img,None)
        matcher = cv2.BFMatcher()
        matchList=[]
        finalVal = -1
        try:
            for des in self.desList_Train:
                matches = matcher.knnMatch(des, des2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                matchList.append(len(good))
        except:
            pass
        if VERBOSE:
            print('Good Matches per Class')
            print(matchList)
        # Give back Id (finalVal) of matching class and show matches in image    
        if len(matchList)!=0:
            if max(matchList) > thres:
                finalVal = matchList.index(max(matchList))
                if (VERBOSE and finalVal != -1):
                    #Matchpoints erneut generieren
                    matches = matcher.knnMatch(self.desList_Train[finalVal], des2, k=2)
                    good = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good.append([m])
                    # Show the image with matches
                    img_matches = cv2.drawMatchesKnn(self.images_train[finalVal], self.kpList_Train[finalVal],img, kp2, matches[:20],None, flags=2)
                    img_matches = cv2.resize(img_matches, (1000,650)) 
                    cv2.imshow("Matches", img_matches) 
                    cv2.waitKey(1)                    
            elif (VERBOSE):
                try:
                    cv2.destroyWindow("Matches") 
                except:
                    pass
        return finalVal

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here camera-images get converted, features detected and objects classified'''
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        image_g = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        id = self.findClassID(image_g)
        if id != -1:
            cv2.putText(image_np,self.classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
        cv2.imshow('img_np',image_np)
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
    rospy.init_node('object_detection', anonymous=True)
    path_trainimg = '/home/rosmatch/catkin_ws_hartmann/src/Mir200_Sim/mir_vision/classes'
    # detectors: 'ORB','SIFT', 'SURF', 'STAR', 'BRISK', 'AKAZE', 'KAZE'
    od = object_detection(path_trainimg, 'ORB')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

