import cv2,os

cam = cv2.VideoCapture(0)  # This Command is for activating Laptop camera

# Create Haarcascade calssifier to detect frontal Face
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

i=0
sample_num = 0

print('Training Start...')
while(True):
    ret ,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(gray.shape)

    faces = detector.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5) # return the position whre face is detected

    for (x,y,w,h) in faces :
        img1 = gray[y:y+h,x:x+w]   # Crop only Face part
        img1 = cv2.resize(img1,(112,92)) # For making a standard size for all images 

        cv2.imwrite('Cam_image/u4/'+str(i)+'.jpg',img1) # For saving the cropped image in a Folder
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2) # For putting a rectangle over the Face detected

        cv2.imshow('frame',gray) # For Showing The Image
        i+=1

    if i == 100:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Training Done...')
cam.release()
cv2.destroyAllWindows()
    
        
