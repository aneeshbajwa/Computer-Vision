import cv2
#import libraries


#cascad classifiers from XML file 
eye_cascade = cv2.CascadeClassifier('../aneeshbajwa/Downloads/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_eye.xml')


#funtion to detect eyes
def detect_eyes(image):
    
    image_copy = image.copy()
    
    #return tuple of loaction and width of eyes
    eye_detect = eye_cascade.detectMultiScale(image_copy,scaleFactor=1.1,minNeighbors=5)
    
    #tuple unpacking and drawing a box around eyes
    for (x,y,w,h) in eye_detect:
        cv2.rectangle(image_copy,(x,y),(x+w,y+h),(255,255,255),10)
        
    return image_copy



cap = cv2.VideoCapture(0)

#applying function on video being captured by laptop camera
while True:
    
    ret,frames = cap.read(0)
    
    frames = detect_eyes(frames)
    
    cv2.imshow('face detection', frames)
    
    
    k = cv2.waitKey(1)
    
    if k==27:
        break
        
        
cap.release()
cv2.destroyAllWindows()
