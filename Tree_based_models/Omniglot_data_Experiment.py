#!/usr/bin/env python
# coding: utf-8


import sys
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
#from scipy.misc import imread
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import numpy.random as rng
#import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle 
import pandas as pd


data_path = "/home-mscluster/szita"


train_path = os.path.join(data_path, 'images_background')
New_xtrain = os.path.join (data_path, 'images') 
#validation_path = os.path.join(data_path, 'images_evaluation')


#def load_images_from_directory(path1):

#load alphabet separately and append that to tensor 
#This forloop will be usefull when training mult decision trees
alphabet_folder_list = [f for f in listdir(train_path) if isdir(join(train_path, f))]
alphabet_folder_list.sort()
alphabet_folder_list_test = [f1 for f1 in listdir(New_xtrain) if isdir(join(New_xtrain, f1))]

alphabet_folder_list_test.sort()
#print(alphabet_folder_list)
#print(alphabet_folder_list_test)
#Dictionary to store model perfomance score 
Dict_score = {keys: [] for keys in alphabet_folder_list_test}

#print("length of alphabet list", alphabet_folder_list)
for alphabet, alphabet_new in zip(alphabet_folder_list, alphabet_folder_list_test):
    Ytrain = []
    Xtrain = []
    X_new = []
    Y_new = []
    alph_list = []
    alph_list_new = []
    letter_class = []
    letter_class_new = []

    alphabet_path = os.path.join(train_path, alphabet)
    alphabet_path_new = os.path.join(New_xtrain, alphabet_new)
    print("alphabet_path train, alphabet_path_new", alphabet_path, alphabet_path_new)
    alph_list.append(alphabet_path)
    alph_list_new.append(alphabet_new)

    character_folder_list = [cf for cf in listdir(alphabet_path) if isdir(join(alphabet_path, cf))]
    character_folder_list.sort()
    print ("char folder list len", len(character_folder_list))
    
    character_folder_list_new = [cf1 for cf1 in listdir(alphabet_path_new) if isdir(join(alphabet_path_new, cf1))]
    character_folder_list_new.sort()
    print ("char folder test list len", len(character_folder_list_new))
    
    #each character in alphabet is in different folder (char0, char1, ...)
    category_labels = []
    category_labels_new = []
    F_name = []
    F_name_new = []
    
    #count = len(join(alphabet_path, char))
    for j,char in enumerate(character_folder_list):
        print("J and char is", j, char)
        character_folder_path = join(alphabet_path, char) 
        print("char folder path is", character_folder_path)
        count = len( listdir(character_folder_path))

        print ("counts is ", count)

        letter_class.append(char)
        catergory_images = []
        category_labels.append(np.repeat(j,count))

        if not isdir(character_folder_path):
            continue 

        #read every image in this directory (char0, char1, ...)

        image_list =  [ img for img in listdir(character_folder_path) if (isfile(join(character_folder_path,img)) and img[0] != '.')]

        for filename in image_list:
            F_name.append(filename)
            image_path = join(character_folder_path, filename)
            image = imread(image_path)

            #image preprocessing 
            image = image/255
            image = 1- image

            catergory_images.append(image)

        try:
            Xtrain.append(np.stack(catergory_images))

        #edgecase - last one 
        except ValueError as e:
            #print ("error - category_images:", catergory_images)
            print ("error - category_images:")

    Ytrain.append(np.stack(category_labels))
    Xtrain = np.stack(Xtrain)
    Ytrain = np.stack(Ytrain)
    new_Xtrain = Xtrain.reshape([Xtrain.shape[0]*Xtrain.shape[1], 105*105])
    new_ylabel = Ytrain.reshape(Ytrain[0].shape[0]*Ytrain[0].shape[1])
    print(new_Xtrain.shape)
    print(new_ylabel.shape)
    
    
    for jj,char2 in enumerate(character_folder_list_new):
        print("JJ is", jj, char2)
        character_folder_path_new = join(alphabet_path_new, char2) 
        print("char2 folder path is", character_folder_path_new)
        count_new = len( listdir(character_folder_path_new))


        letter_class_new.append(char2)
        catergory_images_new = []
        category_labels_new.append(np.repeat(jj,count_new))

        if not isdir(character_folder_path_new):
            continue 

        image_list_new =  [ img for img in listdir(character_folder_path_new) if (isfile(join(character_folder_path_new,img)) and img[0] != '.')]

        for filename_new in image_list_new:
            F_name_new.append(filename_new)
            image_path_new = join(character_folder_path_new, filename_new)
            image_new = imread(image_path_new)

            #image preprocessing 
            image_new = image_new/255
            image_new = 1- image_new

            catergory_images_new.append(image_new)

        try:
            X_new.append(np.stack(catergory_images_new))

        #edgecase - last one 
        except ValueError as e:
            #print ("error - category_images:", catergory_images)
            print ("error - category_images:")

    Y_new.append(np.stack(category_labels_new))
    X_new = np.stack(X_new)
    Y_new = np.stack(Y_new)
    new_X = X_new.reshape([X_new.shape[0]*X_new.shape[1], 105*105])
    new_Y = Y_new.reshape(Y_new[0].shape[0]*Y_new[0].shape[1])

    print(new_X.shape)
    print(new_Y.shape)


    #train_pct_index = int(0.2 * len(Xtrain))
    #X_train, X_test = Xtrain[:train_pct_index], Xtrain[train_pct_index:]
    #y_train, y_test = ytrain[:train_pct_index], ytrain[train_pct_index:]  

    #Randomized search parameter optimization
    
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    # First create the base model to tune
    rf = RandomForestClassifier()
    DC_model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, 
                                   verbose=2, random_state=42, n_jobs = -1)
    
    #grid = {"n_estimators":np.arange(10,100 , 20)
    #      ,"min_samples_leaf":np.arange(10, 50, 5)}

    #Randomized search parameter optimization 
    #DC_model = RandomizedSearchCV(RandomForestClassifier(random_state = 0, oob_score = 0)
    #                      ,param_distributions = grid
    #                      ,cv = 15, n_iter=20
    #                      ,n_jobs=-1, random_state = 0)
    DC_model.fit(new_Xtrain, new_ylabel)
     
    
    y_pred =  DC_model.predict(new_X)
    #save the model to disk
    #filename_save = alphabet_new +'.sav'
    #pickle.dump(DC_model, open(filename_save, 'wb'))


    classes_new = np.array(letter_class_new)
    score2 = accuracy_score(y_pred, new_Y)
    #error 
    print ("score is", score2)
    
    
    Dict_score[alphabet_new].append(score2)


    L = len(np.where(new_Y != y_pred)[0]) 

    print("############################ len of predicted is ####################", L)
    
    
    #Alpha_name = listdir(train_path)
    F_name_new = np.array(F_name_new)
    
    #size = np.arange(0, len(F_name))

    #difference = np.setdiff1d(size, rotated_only)

    #reduced_original = F_name[difference]

    #new_y_pred = y_pred[difference]

    print("New classes", classes_new[y_pred])

    #creat dictionary with index image per class
    Dictionary_idx = {keys: [] for keys in classes_new}


    for i,j in Dictionary_idx.items():
        #print (i)
        Dictionary_idx[i].append(np.where(classes_new[y_pred] == i))


    #convert and save images into different classes from y_pred
    Dict_nimages = {keys: [] for keys in classes_new}

    for i,j in Dict_nimages.items():
        Dict_nimages[i].append(np.shape(np.where(classes_new[y_pred] == i))[1])
    print("Dict_nimages", Dict_nimages)


    for j,k in Dict_nimages.items():
        #print("k is", k)
        kk =  str(j)
        rot_character_path = join(character_folder_path, kk)

        if not os.path.exists(rot_character_path):
            os.makedirs(rot_character_path)

        #print("Here is char path", char_path)
        ggg = []
        for im in range(k[0]):
            ggg.append(im)
        new_ggg = random.sample(ggg,10)
        for im in new_ggg:
            im_arr = new_X[np.ravel(Dictionary_idx[j])[im]].reshape(105,105)
            im_arr = im_arr.astype(np.uint8)
            
            im_arr = 255 + im_arr
            arr_im = Image.fromarray(im_arr)
        
            arr_im.save(join(character_folder_path, kk,F_name_new[im]))
            
dataframe = pd.DataFrame(Dict_score)
dataframe.to_csv("Accuracy_omniglot.csv")


#if __name__ == "__main__":
#        print ("loading training set")
#        #greek 
#        #X_train, X_test, y_train, y_test,  y_pred, alph_list, letter_class, model = load_images_from_directory(train_path)
#        Xtrain, ytrain, alph_list, letter_class, y_pred, DC_model, F_name = load_images_from_directory(train_path, new_path)

ticks = (np.arange(dataframe.shape[1]))

plt.figure(figsize=(10,5))
plt.plot(dataframe.to_numpy()[0], linewidth = 3)
plt.xlabel("Alphabet")
plt.ylabel("Accuracy")
plt.title("Omniglot Tree prediction")
plt.xticks(ticks,list(dataframe.columns),rotation=90)
plt.grid()


#plot to examine data in xtrain 
#plt.figure(figsize = (15,15))
#for i in range(24):    grid = {"n_estimators":np.arange(10,100 , 20)
#          ,"min_samples_leaf":np.arange(10, 50, 5)}
##    plt.subplot(4,8,i+1)
#    plt.axis('off')
#    plt.imshow(Xtrain[i].reshape(105,105))
#    #plt.title(classes[ytrain[i]])
#    plt.title(F_name[i])

#Dicty = {keys: [] for keys in os.listdir(train_path)}
#score_list = []

"""
Header = ['Scores']
try:
    with open('scores.csv') as f:
        f.close()
        f = open('scores.csv','a')
        c = csv.writer(f, lineterminator='\n')
        c.writerow(score_list)
        f.close()
except IOError:
    f = open('scores.csv', 'a') # Create file for the first time
    w = csv.writer(f, lineterminator='\n')
    w.writerow(Header)
    w.writerow(score_list)
    f.close()
    
Header2 = ['Alphabet']    
try:
    with open('alpha_name.csv') as f:
        f.close()
        f = open('alpha_name.csv','a')
        c = csv.writer(f, lineterminator='\n')
        c.writerow(Alpha_name)# Append values without header
        f.close()
except IOError:
    f = open('alpha_name.csv', 'a') # Create file for the first time
    w = csv.writer(f, lineterminator='\n')
    w.writerow(Header2)
    w.writerow(Alpha_name)
    f.close()
"""





