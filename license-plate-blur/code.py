import cv2
import matplotlib.pyplot as plt
#import libraries



#Function to display image larger
def display(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    
    
    
# using the xml file of haar cascade to detect russian license plates    
haar_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_russian_plate_number.xml')


#function to detect and blure license plate
def detect_and_blur_plate(img):
    img_copy = img.copy()
    
    #returning the location of top left corner of license plate along with width and height of license plate as a tuple
    number_plate_detect = haar_cascade.detectMultiScale(img_copy,scaleFactor=1.1,minNeighbors=5)
    
    roi = img_copy
    
    #unpacking tuple to get and then blur license part of image
    for (x,y,w,h) in number_plate_detect:
        #grabbing license plate from the image
        roi = img_copy[y:y+h+1,x:x+w+1]
        #bluring license plate
        roi = cv2.medianBlur(roi,35)
        
        #replacing the original image's license plate with blurrd license plate
        img_copy[y:y+h+1,x:x+w+1] = roi
    
    #returning blurred image    
    return img_copy
    
#running blur function on image of a russian car

#Read image file
img = cv2.imread('../DATA/car_plate.jpg')

#converting image color channels from BGR to ----> RGB
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
result = detect_and_blur_plate(img)

#displaying image
display(result)
