import Face_Regognition_orl_face

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


