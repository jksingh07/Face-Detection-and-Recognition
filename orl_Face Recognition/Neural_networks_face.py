import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,metrics,svm
from sklearn.neural_network import MLPClassifier

#data = np.zeros((400,92))
#target = np.zeros((400,1))
train_data = np.zeros((280,112))
test_data = np.zeros((120,112))
train_target = np.zeros((280,1))
test_target = np.zeros((120,1))

count1 = 0
count2 = 0

for i in range(40):
    for j in range(7):
        img = plt.imread('orl_face/orl_face/u%d/%d.png'%(i+1,j+1))
        train_data[count1] = img.mean(axis=1)
        train_target[count1] = i+1
        count1+=1

for i in range(40):
    for j in range(7,10):
        img = plt.imread('orl_face/orl_face/u%d/%d.png'%(i+1,j+1))
        test_data[count2] = img.mean(axis=1)
        test_target[count2] = i+1
        count2+=1


        

################    CREATE A NEURAL MODEL     ########################################

neural_model =  MLPClassifier(hidden_layer_sizes=[200],verbose=0,solver='adam',max_iter=3000) # instead of using poly as a kernel we can use
                                      # linear or rbf also. FINALLY CHECK WHOSE ACCURACY IS MORE AND CHOOSE THAT ONE

#################   TRAIN YOUR NEURAL MODEL   #####################################

train_neural = neural_model.fit(train_data,train_target)

#################   TEST THE DATA (TESTING)  ############################################

predict = train_neural.predict(test_data)

################  CALCULATE THE ACCURACY  #########################################3

score = metrics.accuracy_score(test_target,predict)


##################  MAKE A CONFUSION METRICS  ######################################

conf_m = metrics.confusion_matrix(test_target,predict)

################## ALL OVER REPORT OF THIS DATA  #####################################

report = metrics.classification_report(test_target,predict)

print(report,conf_m,'\nACCURACY :',score*100)
