import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image
import os
import random
import math
import json
import pickle
import datetime
import yagmail
import tensorflow as tf
import csv

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import load_img

############# Basic Functions #############
def save_var(var, file_name):
    '''
    Saves any type of variable with the given filename(can be a path)
    '''
    out_file = open(file_name,'wb')
    pickle.dump(var,out_file)
    out_file.close()

def read_var(file_name):
    infile = open(file_name,'rb')
    var = pickle.load(infile)
    infile.close()
    return var

def save_model(model, name):
    model_json = model.to_json()
    with open(name+'.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(name+'.h5')
    print("Saved model to disk")

def load_model(name):
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name+'.h5')
    print("Loaded model from disk")
    return loaded_model

def make_subfolder(dirname,parent_path):
    path = os.path.join(parent_path, dirname)
    os.mkdir(path)
    print("Directory '%s' created" %dirname)
    return path + '/'

def make_log_dir(parent_path = ""):
    current_date = datetime.datetime.now()
    dirname = current_date.strftime("%Y_%B_%d-%H_%M_%S")
    date = current_date.strftime("_%d")
    path = make_subfolder(dirname,parent_path)
    return path

def write_to_log(log_dir ="", log_entry = ""):
    with open(log_dir + "/log.txt", "a") as file:
        file.write(log_entry)

def send_as_mail(log_dir):
    log = log_dir + '/log.txt'
    conf_mat = log_dir + '/confusionMatrix.png'
    loss_and_acc = log_dir + '/loss_and_acc.png'
    contents = [ "Train sonuçları ve konfigürasyonu ekte yer almaktadır",
    log, loss_and_acc, conf_mat
    ]
    with yagmail.SMTP('viventedevelopment', 'yeniparrola2.1') as yag:
        yag.send('ademgunesen+viventedev@gmail.com', 'Train Sonuçları', contents)

def show_images(images: list, titles: list="Untitled    ", colorScale='gray', rows = 0, columns = 0) -> None:
    n: int = len(images)
    if rows == 0:
        rows=int(math.sqrt(n))
    if columns == 0:
        columns=(n/rows)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i], cmap=colorScale)
        plt.title(titles[i])
    plt.show(block=True)

def get_computer_info():
    f = open('computer_info.json',)
    computer_info = json.load(f)
    print("Working on "+computer_info['name'])
    return computer_info

def return_date():
    current_date = datetime.datetime.now()
    date = current_date.strftime("%Y_%B_%d")
    return date

############# Save train results to csv file #############
def initialize_metrics_csv(date):
    #initialize csv file with date and header OLD
    header = ['log_dir', 'Model Name', 'Dense Layer', 'Dropout', 'Batch Size', 'Warmup Learning Rate', 'Learning Rate', 'Loss', 'Accuracy', 'ROC AUC', 'Best Threshold']#run once 
    with open(f'out/Train_Metrics_{date}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        csv_file = (f'out/Train_Metrics_{date}.csv')
    return csv_file 

def initialize_train_csv(date, t_conf):
    #initialize csv file with date and header NEW
    header = list(vars(t_conf).keys())+['LOSS', 'ACCURACY', 'ROC_AUC', 'BEST_THRESHOLD']#run once 
    with open(f'out/Train_Metrics_{date}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        csv_file = (f'out/Train_Metrics_{date}.csv')
    return csv_file

def initialize_tasks_csv():
    #initialize csv file with date and header
    header = ['task_id', 'log_dir', 'task_state']#run once 
    with open(f'out/Tasks.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

def save_metrics(csv_file, t_conf, loss, acc, roc_auc, best_thresh):
    comp_info = get_computer_info()
    #save same inputs and outputs to cvs file 
    file = comp_info["Codes"]+(f'out/Train_Metrics.csv')
    row = list(vars(t_conf).values())+[loss, acc, roc_auc, best_thresh]
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def save_metrics_old(csv_file, conf, loss, acc, roc_auc, best_thresh):
    #save same inputs and outputs to cvs file 
    row = []
    row = [conf.log_dir, conf.MODEL_NAME, conf.DENSE_LAYER, conf.DROP_OUT, conf.BATCH_SIZE, conf.WARMUP_LEARNING_RATE, conf.LEARNING_RATE, loss, acc, roc_auc, best_thresh]
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

############# Converting ataset to csv file #############
def write_to_csv():
    #read images and labels from dataset and convert to csv file (1.)
    '''
    header = ['Images',  'Label', 'Subsets']#run once
    with open(f'C:/Users/viven/Desktop/ROPProjects/Data/Dataset/..?...csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    '''
    path = 'C:/Users/viven/Desktop/ROPProjects/Data/Dataset/'
    list = os.listdir('C:/Users/viven/Desktop/ROPProjects/Data/..?...')

    for i in range(len(list)):
        row = [list[i], '0', 'Training']#'0', list1[i], list2[i], list3[i]
        with open(path+'...?...csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def create_split_csv(path):
    #creating the subset as validation by 0.2 (2.)
    df = pd.read_csv(path+'csvDataset/...?.....csv.')

    df_sub = df.sample(frac=0.2, random_state=2)#0.2, 2
    
    for index in df.index:
        for sub_index in df_sub.index:
            if index == sub_index:
                if df.iloc[index,2] == 'Training':
                    df.iloc[index,2] = 'Validation'

    print(df['Subsets'])
    df.to_csv(path+'Dataset/...?...csv', index=False)

############# writing, reading and save in csv file #############
def read_save_pixels():
    #read images pizels and save a csv file
    path = "C:/Users/viven/Desktop/ROPProjects/Data/TrainDataset/" 
    list = os.listdir("C:/Users/viven/Desktop/ROPProjects/Data/TrainDataset")
    '''
    header = ['image_id', 'width', 'height']#run once 
    with open(f'C:/Users/viven/Desktop/ROPProjects/Data/...?....csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    '''
    for i in range(len(list)):
        img = Image.open(path+list[i])
        width, height = img.size   # Get dimensions   
        row = [list[i], width, height]
        with open('C:/Users/viven/Desktop/ROPProjects/Data/...?...csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row) 

def read_best_threshold(comp_info):
    #read best threshold from csv file for evaluate with test dataset
    file = comp_info["Codes"]+'/out/Train_Metrics.csv'
    df = pd.read_csv(comp_info["Codes"]+'/out/Train_Metrics.csv')
    with open(file) as f:
        reader = csv.reader(f, delimiter=",")
        for index in df.index:
            if df.iloc[index, 0] == 'out/2021_November_24-12_18_03/':
                best_thresh = df.iloc[index, 10]

    return best_thresh

############# Adjusting the size of images #############
def resize_image(data_path):
    #resize images
    path = data_path+"...?.."
    list = os.listdir(data_path+"...?...")
    for index in range(len(list)):
        image = Image.open(path+list[index])
        name = list[index]
        sep = '.'
        name = name.split(sep, 1)[0]
        width, height = image.size
        print(width)
        if width == 2056:             
            resized_image = image.resize((1444,1444))
            print(resized_image.size)   
            resized_image.save(data_path+"..?../asdf/"+name+".jpg")
            #show_images([image, cropped_image])
        
        elif width == 1444:
            #image = io.imread(path+list[index]) 
            image = Image.open(path+list[index])
            print(resized_image.size) 
            #show_images([image, cropped_image])
            image.save(data_path+"...?.../asdf/"+name+".jpg")
            
def crop_save_images(data_path):
    # Crop the center of the image
    path = data_path+"...?.../"
    list = os.listdir(data_path+"...?...")
    file = data_path+'...?....csv'
    df = pd.read_csv(data_path+'...?....csv')
    '''
    with open(file) as f: 
        reader = csv.reader(f, delimiter=",")
        for index in df.index:
    ''' 
    print(len(list))
    for index in range(len(list)):

        img = Image.open(path+list[index])
        print(list[index])
        name = list[index]
        sep = '.'
        name = name.split(sep, 1)[0]
        print(name)
        width, height = img.size
        x = int(width/2)
        if width == 2124:             
            image = io.imread(path+list[index]) 
            cropped_image = image[:, x-1028:x+1028]
            plt.imsave(data_path+"...?.."+name+".jpg", cropped_image)
            #show_images([image, cropped_image])
        
        elif width == 1444:
            image = io.imread(path+list[index]) 
            plt.imsave(data_path+"...?..."+name+".jpg", cropped_image)
            #show_images([image, cropped_image])
            cropped_image = image[:, :]
            
############# Check tasks cases #############
def start_task(t_conf, g_conf, d_conf, task_num):
    log_dir = make_log_dir("out/")                                                                                                                                                                                        
    t_conf.save(save_dir = log_dir)
    g_conf.save(save_dir = log_dir)
    d_conf.save(save_dir = log_dir)
    start_info = [task_num, log_dir, 'Started']
    with open(f'out/Tasks.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(start_info)
    return log_dir

def stop_task(t_conf, task_num):
    start_info = [task_num, t_conf.log_dir, 'Completed']
    with open(f'out/Tasks.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(start_info)

def check_task_state(task_id):
    fields = []
    state = 'Pending'
    print(task_id)
    with open(f'out/Tasks.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            print(row[0],row[2])
            if (task_id==int(row[0])):
                state = row[2]
                print("Match!")
    return state