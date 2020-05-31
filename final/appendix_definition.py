# -*- coding: utf-8 -*-
# Define some functions for easy calling
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import tkinter
from tkinter import *
from PIL import ImageTk, Image

# classify the data into words,topics,documents
def data_preprocessing(data):
    #  Initialization 
    words,topics,documents =[],[],[]
    # Classify the corpus into words, topics and documents             
    for i in data['data']:
        for j in i['questions']:
            k = nltk.word_tokenize(j) 
            words.extend(k)
            documents.append((k, i['tag']))
            if i['tag'] not in topics:
                topics.append(i['tag'])
    return words,topics,documents


# lemmaztizion,remove duplicates and delete the useless symbols 
def data_lemmaztizion(words,topics,useless_words):
    lemmatizer = WordNetLemmatizer() # reduce the word to original form
    # change upper to lower
    for i in words:
        words[words.index(i)] = i.lower()
    # delete duplicates
    words = [lemmatizer.lemmatize(i) for i in words if i not in useless_words]
    # package variable, otherwise the result will be chaos
    pickle.dump(words,open('words.pkl','wb'))
    pickle.dump(topics,open('topics.pkl','wb'))
    return words,topics

 
# translate words to numbers
def word_to_number(words,topics,documents):
    lemmatizer = WordNetLemmatizer() # reduce the word to original form
    A = []
    output_v = [0] * len(topics)
    for doc in documents:
        bag = []     # bag of words for each sentence
        question_w = doc[0]
        question_w = [lemmatizer.lemmatize(word.lower()) for word in question_w]
        # if word match found in current question, bag = 1 
        for w in words:
            bag.append(1) if w in question_w else bag.append(0)
        # output = 0 for each tag; =1 for current tag for each question
        flag = list(output_v)
        flag[topics.index(doc[1])] = 1
        A.append([bag, flag])
    random.shuffle(A)    # shuffle features 
    A = np.array(A)
    # create train and test lists. X - questions, Y - data
    train_x = list(A[:,0])
    train_y = list(A[:,1])
    return train_x,train_y


# predict the topic 
def topic_predict(query, words,topics,model):
    # first, we need clean the query
    lemmatizer = WordNetLemmatizer() # reduce the word to original form
    query_words = nltk.word_tokenize(query)
    query_words = [lemmatizer.lemmatize(word.lower()) for word in query_words]
    bag = [0]*len(words)  # words matrix 
    for k in query_words:
        for i,j in enumerate(words):
            if j == k: 
                bag[i] = 1 # if current word is in the vocabulary position
    Predict = np.array(bag)
    res = model.predict(np.array([Predict]))[0]
    threshold_low = 0.2 # define a threshold
    results = [[i,r] for i,r in enumerate(res) if r>threshold_low]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"topic": topics[r[0]], "probability": str(r[1])})
    return return_list


# get response
def get_response(t, topic_j):
    tag = t[0]['topic']
    list_data = topic_j['data']
    for i in list_data:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result



