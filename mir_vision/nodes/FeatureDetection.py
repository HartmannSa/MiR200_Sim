#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import cv2
import os
import sys, time
from scipy.ndimage import filters
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
from matplotlib import pyplot as plt


VERBOSE=False

class object_detection:
    features = {'ORB','SIFT', 'SURF', 'STAR', 'BRISK', 'AKAZE', 'KAZE'}

    def __init__(self, path, detector='ORB'):
        '''Initialize ros publisher, ros subscriber'''
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage, queue_size=500)
        self.matches = []    
        self.path = path            # Path to class_images/ Trainimages
        self.detector = detector    # Which Feature Detector is used
        self.CreateMatches()        # Detect Features in Trainimages
        self.subscriber = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print("subscribed to /camera/color/image_raw/compressed")
    
    def __del__(self): 
        if VERBOSE:
            self.subscriber.unregister()
            self.image_pub.unregister()
            print("Destructor called") 

    def findDes(self, images):
        if (self.detector=='ORB'):        
            self.det = cv2.ORB_create(nfeatures=1000)
        elif (self.detector=='SIFT'):
            self.det = cv2.xfeatures2d.SIFT_create()
        elif (self.detector=='SURF'):
            self.det = cv2.xfeatures2d.SURF_create()
        else:
            print("\n########################################################")
            print("Detector ' + self.detector + ' is not implemented yet")
            print("########################################################\n")
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
            print("Total Classes", len(myList))
            print(self.classNames) 
        # Generate Descriptors from Trainimages
        self.kpList_Train, self.desList_Train = self.findDes(self.images_train)
        if VERBOSE :              
            print("Number of Descriptors", len(self.desList_Train))

    def findClassID(self, img,thres=20, match_method='best'):
        kp2,des2 = self.det.detectAndCompute(img,None)
        ## FLANN Matching
        if match_method == 'flann':            
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
            search_params = {}
            matcher_fl = cv2.FlannBasedMatcher(index_params, search_params)
        ## Brute-Force Matching
        elif match_method == 'best':
            matcher_bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # NORM_HAMMING for ORB
        elif match_method =='ratio':
            matcher_bf2 = cv2.BFMatcher()
        matchList=[]
        finalVal = -1
        try:
            for des in self.desList_Train:
                if des.shape == des2.shape:
                    good = []
                    ## Crosscheck and best matches
                    if match_method == 'best':
                        matches = matcher_bf1.match(des,des2)   # return only best match
                        good = sorted(matches, key = lambda x:x.distance)
                    else:
                        if match_method =='ratio':
                            matches = matcher_bf2.knnMatch(des, des2, k=2)   # return k best matches
                        elif match_method == 'flann':
                            matches = matcher_fl.knnMatch(des, des2, k=2)
                        ## Ratio Test Method
                        for m, n in matches:
                            if m.distance < 0.75 * n.distance:
                                good.append([m])

                    ## save number of and best matches (equal to most good matches)
                    matchList.append(len(good))  
                    if len(matchList)!=0 and len(good) != 0:       # actually not possible, even when goods = [], then ist matchList=[0]'
                       if len(good) == max(matchList):
                           self.best_matches = good 
                           print("self.best_matches wurde zugewiesen")                               
        except:
            pass
        if VERBOSE:
            print("Good Matches per Class")
            print(matchList)
        ## Give back Id (finalVal) of matching class and show matches in image    
        if len(matchList)!=0:
            if max(matchList) > thres:
                finalVal = matchList.index(max(matchList))
                ## Draw Bounding Box
                MIN_MATCH_COUNT = 10
                if len(self.best_matches)>MIN_MATCH_COUNT:
                    good_matches = self.best_matches[:MIN_MATCH_COUNT]
                    kp1 = self.kpList_Train[finalVal]
                    img1=self.images_train[finalVal]
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    matchesMask = mask.ravel().tolist()
                    h,w = img1.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)
                    dst += (w, 0)  # adding offset
                    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                singlePointColor = None,
                                matchesMask = matchesMask, # draw only inliers
                                flags = 2)
                    img_bb = cv2.drawMatches(img1,kp1,img,kp2,good_matches, None,**draw_params)
                    # Draw bounding box in Red
                    img3 = cv2.polylines(img_bb, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
                    cv2.imshow("result", img3)
                    cv2.waitKey(1)
                else:
                    print("Not enough matches are found - %d/%d" %(len(self.matches),MIN_MATCH_COUNT))
                    matchesMask = None
                ## Show the image only with matches
                if (VERBOSE and finalVal != -1): 
                    if match_method == 'best':
                        print('best is choosen')
                        img_matches = cv2.drawMatches(img1, kp1 ,img, kp2, self.best_matches[:10],None, flags=2)
                    elif match_method =='ratio' or match_method =='flann':
                        img_matches = cv2.drawMatchesKnn(self.images_train[finalVal], self.kpList_Train[finalVal],img, kp2, self.best_matches[:20],None, flags=2) # draws all the k best matches (so e.g. two lines for each keypoint) (but in good only m is appended)
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
            print("received image of type: %s" %(ros_data.format))

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        image_g = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        time1 = time.time()
        id = self.findClassID(image_g, match_method='best')
        time2 = time.time()
        if id != -1:
            cv2.putText(image_np,self.classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            if VERBOSE :
                print("%s detector found class in %.3f sec." %(self.detector,time2-time1))
        cv2.imshow('img_np',image_np)
        cv2.waitKey(1)       
        
        #### Create and Publish CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        self.image_pub.publish(msg)     

def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('object_detection', anonymous=True)
    path_trainimg = '/home/rosmatch/catkin_ws_hartmann/src/Mir200_Sim/mir_vision/classes'
    # detectors: 'ORB','SIFT', 'SURF', 'STAR', 'BRISK', 'AKAZE', 'KAZE'
    od = object_detection(path_trainimg, 'ORB')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

