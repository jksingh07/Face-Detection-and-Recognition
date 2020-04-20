import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimage

im = []
feat = []
target =[]

for i in range(40):
    im.append([])
    feat.append([])
    target.append(i+1)
for i in range(40):
    for j in range(10):
        path ='./orl_face/orl_face/u%d/%d.png'%(i+1,j+1)
        im[i].append(mimage.imread(path))

for i in range(40):
    #plt.figure(0)
    for j in range(10):
        
        #plt.imshow(im[i][j],cmap='gray')
        #plt.pause(0.3)
        feat[i].append(im[i][j].reshape(1,-1))

#print(feat,len(feat))
#plt.ioff()


################  CODE STARTS BELOW ##### ABOVE CODE IS ONLY FOR TESTING PURPOSE  ########


x_train=np.zeros((280,10304))  # TRAINING DATA
x_test=np.zeros((120,10304))   # TESTING DATA
y_train=np.zeros((280,1))  # TARGET VALUES OF TRAINING  DATA
y_test=np.zeros((120,1))   # TARGET VALUES OF TESTING  DATA
count=0
count2=0
for i in range(40):
    for j in range(7):
        path1 ='./orl_face/orl_face/u%d/%d.png'%(i+1,j+1)
        c = mimage.imread(path1)
        #print(c)
        x_train[count] = c.reshape(1,-1)
        y_train[count] = i+1
        count+=1
        
for i in range(40):
    for j in range(7,10):
        path2 ='./orl_face/orl_face/u%d/%d.png'%(i+1,j+1)
        c = mimage.imread(path2)
        #print(c)
        x_test[count2] = c.reshape(1,-1)
        y_test[count2] = i+1
        count2+=1

min_d=[]
for i in range(120):
    d=[]
    for j in range(280):
        d.append(np.sum((x_test[i]-x_train[j])**2))
    min_d.append(np.argmin(d))

predict = []
for i in range(120):
    predict.append(y_train[min_d[i]][0])

correct = 0
for i in range(120):
    if y_test[i]==predict[i]:
        correct+=1
        
score = correct/120
Accuracy = score*100
print('Actual Results',y_test)
print('\n\nPREDICTED RESULTS\n',predict)
print("\n\n\t\tAccuracy :%d"%(Accuracy),'%')


###############             FACE RECOGINITION          ###################################################
###############  PASSING THE IMAGE AND RECOGNISING IT  #################################



def FaceRecognition(image_path):
    data = mimage.imread(image_path)
    img_data = data.reshape(1,-1)
    print(img_data)
    dist =[]
    for i in range(280):
        dist.append(np.sum((img_data - x_train[i])**2))
        
    index=np.argmin(dist)
    predicted_img = x_train[index].reshape(112,92)
    predicted_Target = y_train[index]
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.title('predicted image')
    plt.imshow(predicted_img,cmap = 'gray')
    plt.subplot(1,2,2)
    plt.title('Image Passed')
    plt.imshow(data,cmap='gray')
    plt.show()

if __name__ == '__main__':
    print("\n\n\t\tFACE RECOGNITION\n\n\t Select any one of the Last 3 images\
            \n\t\t8.png,9.png,10.png\nof any folder because It is not in my Training Data\
            \n\n\t\tPRESS ENTER TO EXIT")
    path =' '
    while path!='':
        path = input("\nEnter  The image Path :")
        if path!='':
            FaceRecognition(path)


#############  GUI    #######################################
from tkinter import *
from tkinter import messagebox
from PIL  import ImageTk, Image

#################  FRAMES   ###########################################
root = Tk()
root.geometry('800x600+20+20')
root.title("Face Recognition")
f = Frame(root,bg='powderblue',width=800,height=100)
f.pack(side=TOP)
f3 = Frame(root,width=800,height=50)
f3.pack(side=TOP)
f2 = Frame(root,bg='powderblue',width=800,height=150)
f2.pack(side=TOP,expand=YES)
f4 = Frame(root,bg='red',width=800,height=300)
f4.pack(side=BOTTOM,expand=YES)


########################  Functions  ##############################################

def predict():
    pos=location.get()
    FaceRecognition(pos)


###################  DESIGNINNG THE FRAME  ######################################

l = Label(f,text='Face Recognition',font =('algerian',30,'bold'),bg='powderblue').grid(row=0,column=0)

l = Label(f2,text='Image Path',bg='cyan',font =('algerian',20,'bold'),width =15,padx=5,pady=5).grid(row=0,column=0)
location =StringVar()
location.set("Enter the path of any one 8.png, 9.png ,10.png")

E = Entry(f2,textvariable = location,font=('arial',14),width=45).grid(row=0,column=1)

B =Button(f2,text='Go',command=predict,bg='cyan',font =('algerian',15,'bold'),width=10,padx=5,pady=5).grid(row=1,column=0)

img = ImageTk.PhotoImage(Image.open('C:/Users/Lenovo/Downloads/Face1.png'))
l2 = Label(f4,image=img).grid(row=0,column=0)


root.mainloop()

##
