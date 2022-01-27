import cv2
import numpy as np

corner_detect_params = dict(maxCorners=10, qualityLevel = 0.3, minDistance = 7, blockSize=7)

lk_params = dict(winSize = (200,200), maxLevel=2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,5,0.05))

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()

prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

#find corneers in this frame
prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_detect_params)

mask = np.zeros_like(prev_frame)

while True:
    
    ret, frame = cap.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    #using previous frame to find corners in next frame
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevPts,None,**lk_params)
    
    good_new = nextPts[status==1] # status is 1 only for points which were tracked
    good_prev = prevPts[status==1]
    
    for i,(new,prev) in enumerate(zip(good_new,good_prev)):
        
        x_new,y_new = new.ravel()
        x_prev,y_prev = prev.ravel()
        
        # Lines will be drawn using the mask created from the first frame
        mask = cv2.line(mask, ( int(x_new), int(y_new) ),( int(x_prev), int(y_prev) ), (0,255,0), 3)
        
        # Draw red circles at corner points
        frame = cv2.circle(frame,(int(x_new),int(y_new) ),8,(0,0,255),-1)
        
        
    # adding mask and original image so that we can display image with lines  tracking objects
    img = cv2.add(frame,mask)
    cv2.imshow('Object Tracking',img)
    
    prev_gray = frame_gray.copy()
    prevPts = good_new.reshape(-1,1,2)
    
    k = cv2.waitKey(2)

    if k == 27:
        break


cv2.destroyAllWindows()
cv2.release()