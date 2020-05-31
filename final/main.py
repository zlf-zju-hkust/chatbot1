# -*- coding: utf-8 -*-
# main method: LSTM, main structure: keras 

######  changing the directory ###### 
# You should change this directory below on your own computer accordingly.
working_folder = r'D:\MINE\HKUST\SecondSeminar\2-6010U - Artificial Intelligence in Finance\robot-Doraemon\final'


######  import the packages ###### 
#  if you don't succeed in this step, pls install them first.
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
# Advanced version of tensorflow-keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
nltk.download('punkt') # Use NLTK for word segmentation
# chat window
import tkinter
from tkinter import *
from PIL import ImageTk, Image



######  import the required functions from the appendix ###### 
from appendix_definition import data_preprocessing,data_lemmaztizion,word_to_number,topic_predict,get_response


######  download the corpus and processing the data ###### 
# corpus is stored in json form
data = json.loads(open('corpus.json').read())
# classify the data into words,topics,documents
words,topics,documents = data_preprocessing(data)
useless_words = ['?', '!','~',';'] 
# lemmaztizion,remove duplicates and delete the useless symbols 
words,topics = data_lemmaztizion(words,topics,useless_words)
# change the data to numbers 
train_x , train_y = word_to_number(words,topics,documents)
print("Data cleaning completed")
print ("numbers of topics:",len(topics))

######  model training ###### 
# total: 3 layers.
# First layer: 128 neurons, second layer: 64 neurons, activation='relu', dropout=0.5
# third layer: equal to number of tag in data, i.e.len(train_y[0]), activation='softmax'
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


######  model compiling and fitting ###### 
# Compile model use SGD method
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=180, batch_size=5, verbose=1)
model.save('model.h5', hist)
print("Model is established")



###### Information processing and messages sending  ###### 
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        App.config(state=NORMAL)
        # input line
        App.insert(END, "You: " + msg + '\n\n')
        App.config(foreground="black", font=("Arial", 16 ))
        # predict topic
        ints = topic_predict(msg, words,topics,model)
        # choose a response from answers randomly
        res = get_response(ints, data)
        App.insert(END, "Doraemon: " + res + '\n\n')
        App.config(state=DISABLED)
        App.yview(END)
        
        
###### define a chat window  ###### 
chatbox = Tk()
# set parameters
chatbox.title("Welcome! we are Doraemon Group!")
background_image = ImageTk.PhotoImage(Image.open('background.gif'))
w = background_image.width()
h = background_image.height()
chatbox.geometry("800x500")
background_label = Label(chatbox, image=background_image)
background_label.place(x=0, y=0, relwidth=0.4, relheight=0.8)
chatbox.resizable(width=FALSE, height=FALSE)
# box for entering message
EntryBox = Text(chatbox, bd=0, bg="#64B2FF",foreground="black",width="29", height="5", font="Arial")
# Button for sending message
SendButton = Button(chatbox, font=("Arial",18,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="white", activebackground="red",fg='black',
                    command= send )
# window for recording chat history
App = Text(chatbox, bg="#64B2FF",bd=0, height="8", width="50", font="Arial")
App.config(state=DISABLED)
# scrollbar in right side
scrollbar = Scrollbar(chatbox, command=App.yview, cursor="circle")
App['yscrollcommand'] = scrollbar.set
# screen place
EntryBox.place(x=320, y=401, height=90, width=450)
SendButton.place(x=5, y=401, height=90,width=315)
App.place(x=320,y=6, height=386, width=450)
scrollbar.place(x=780,y=6, height=386)
chatbox.mainloop()
print("End of chat, pls run the script to chat again")
