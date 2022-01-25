import cv2
#importing libraries


# loading haar cascade XML file 
face_cascade = cv2.CascadeClassifier('../aneeshbajwa/Downloads/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_frontalface_default.xml')


#function to detect face
def detect_face(image):
    image_copy = image.copy()
    
    #returning tuple of location of face in image
    face_detect = face_cascade.detectMultiScale(image_copy)
    
    #tuple unpacking
    for (x,y,w,h) in face_detect:
        cv2.rectangle(image_copy,(x,y),(x+w,y+h),(255,0,0),5)
    
    return image_copy



  
#capturing video
cap = cv2.VideoCapture(0)

while True:
    
    #reading video and then processing on individual frames
    ret,frames = cap.read(0)
    
    frames = detect_face(frames)
    
    cv2.imshow('face detection', frames)
    
    
    k = cv2.waitKey(1)
    
    #press esc key to quit
    if k==27:
        break
        
        
cap.release()
cv2.destroyAllWindows()
