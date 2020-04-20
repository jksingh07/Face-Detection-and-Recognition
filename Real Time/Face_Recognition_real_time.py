#########       HOG DESCRIPTOR      #########################

import cv2,os
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import svm


names = ['Jk','Shivani','Aayushi','Jaskaran','Saran','Sonia']



hog = cv2.HOGDescriptor()

winStride = (8,8)
padding = (8,8)
locations = ((10,20),)


train_data = np.zeros((1,3780))
test_data = np.zeros((1,3780))
train_target = np.zeros((420,1))
test_target = np.zeros((180,1))
count = 0


for j in range(6):
    for i in range(70):
        img = cv2.imread('Cam_image/u%d/%d.jpg'%(j,i))
        img = np.resize(img,(200,200))
        hist = hog.compute(img,winStride,padding,locations)
        
        data1 = np.reshape(hist,(1,-1))
        
        train_data = np.concatenate((train_data,data1),axis=0)
        train_target[count] = j
        count+=1
train_data = train_data[1:,:]
count = 0
for i in range(6):
    for j in range(70,100):
        img = cv2.imread('Cam_image/u%d/%d.jpg'%(i,j))
        img = np.resize(img,(200,200))
        hist = hog.compute(img,winStride,padding,locations)
        
        data1 = np.reshape(hist,(1,-1))
        
        test_data = np.concatenate((test_data,data1),axis=0)
        test_target[count] = i
        count+=1
        
test_data = test_data[1:,:]


###########################      NEURAL NETWORKS      #############################################

#svm_model = svm.SVC()
neural_model = MLPClassifier(hidden_layer_sizes=(100,100,100),n_iter_no_change = 30)
train_neural = neural_model.fit(train_data,train_target)
predict = train_neural.predict(test_data)
score = metrics.accuracy_score(test_target, predict)
conf_matrix = metrics.confusion_matrix(test_target,predict)

print(conf_matrix,'\n\nACCURACY :',score*100)

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
xtest = np.zeros((1,3780))
xpredict = np.zeros((100,1))
i=0
try:
    while(True):
        ret,img1 = cam.read()
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5)

        for (x,y,w,h) in face:
            img2 = gray[y:y+h,x:x+w]
            img2 = cv2.resize(img2,(112,92))
            
            new_img = np.resize(img2,(200,200))
            hist = hog.compute(new_img,winStride,padding,locations)
            test = np.reshape(hist,(1,-1))
            predict = train_neural.predict(test)
        
            print(predict,np.max(train_neural.predict_proba(test)))
##            cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
##            cv2.putText(img1,names[int(predict[0])],(x,y),cv2.FONT_ITALIC,1.5,(255,0,0),2)
            threshold = train_neural.predict_proba(test)[0][int(predict)]
            if threshold > 0.8:
                cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img1,names[int(predict[0])],(x,y),cv2.FONT_ITALIC,1.5,(255,0,0),2)
            else:
                cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img1,'Unknown',(x,y),cv2.FONT_ITALIC,1.5,(255,0,0),2)
            cv2.imshow('Frame',img1)
            

            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
##        if i == 100:
##            break
finally:

    cam.release()
    cv2.destroyAllWindows()
    
