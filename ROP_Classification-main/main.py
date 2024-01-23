import tensorflow as tf
import utils as util
import numpy as np
import pandas as pd
import csv

from utils import start_task, stop_task, check_task_state, save_metrics, return_date, initialize_train_csv
from train import pre_train_model, full_train_model, evaluate_model, evaluate_model_testset, tta_dataset_acc, tta_pred_one_image
from visualizer import plot_loss_and_acc, plot_full_loss_and_acc, visualize_multi_dim
from configurators import train_config, generator_config, dataset_config

#Check GPU is running
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

seed = 42
#Defining the paths of files
comp_info = util.get_computer_info()
path = comp_info["Datasets"]

g_conf = generator_config()

d_conf = dataset_config()
d_conf.train_df_path = comp_info["Datasets"]+'BadDataset/train_dataset.csv'
d_conf.train_path = comp_info["Datasets"] + 'TrainDataset'
d_conf.test_df_path = comp_info["Datasets"]+'BadDataset/test_dataset.csv'
d_conf.test_path = comp_info["Datasets"] + 'TestDataset'

t_conf_list = []
model_name = ['EfficientNetB7']
batch_size = [128]
warmup_lr = [(1e-3)]
learning_rate = [(1e-4)]
dense_layer = [256]
drop_out = [0.2]

for mn in model_name:
    for b in batch_size:
        for wlr in warmup_lr:
            for lr in learning_rate:
                for dl in dense_layer:
                    for do in drop_out: 
                        t_conf = train_config(name = "EfficientNet",
                                                    WARMUP_LEARNING_RATE = wlr,
                                                    LEARNING_RATE = lr,
                                                    BATCH_SIZE = b,
                                                    MODEL_NAME = mn,
                                                    DENSE_LAYER = dl,
                                                    DROP_OUT = do,
                                                    )

                        t_conf_list.append(t_conf)
                        
g_conf = generator_config(name = "default")

#create header&date of csv file
date = return_date()
csv_file = initialize_train_csv(date, t_conf_list[0])

task_id=0

for t_conf in t_conf_list:

    task_state = check_task_state(task_id)
    if task_state == 'Completed':
        task_id+=1
        continue
    elif task_state == 'Pending':

        log_dir = start_task(t_conf, g_conf, d_conf, task_id)
        
        #Pre_Training        
        #model, history = pre_train_model(t_conf,g_conf,d_conf)
        #loss, acc, roc_auc, best_thresh = evaluate_model(model,t_conf,g_conf,d_conf)
        #plot_loss_and_acc(history, save=True, saveDir=log_dir, fname='pretrain_')
        
        #Finetune_Training
        #model, history_finetune = finetune_model(model,t_conf,g_conf,d_conf)
        #loss, acc, roc_auc, best_thresh = evaluate_model(model,t_conf,g_conf,d_conf)
        #plot_loss_and_acc(t_conf, history_finetune, save=True, saveDir=log_dir, fname='finetune_')

        #Full Training
        #model, history, history_finetune, pre_loss, pre_acc= full_train_model(t_conf,g_conf,d_conf)   
        #loss, acc, roc_auc, best_thresh= evaluate_model(model,t_conf,g_conf,d_conf)
        #plot_full_loss_and_acc(t_conf, history, history_fine, save=True, saveDir=log_dir, fname='fulltrain_')

        #Test Dataset Evaluate
        loss, acc, roc_auc, best_thresh = evaluate_model_testset(comp_info,t_conf,g_conf,d_conf)

        #TTA Prediction
        #tta_pred_one_image(model,t_conf,g_conf,image,image_label)
        #tta_dataset_acc(comp_info,t_conf,g_conf,X_test)

        #Save results to excel
        save_metrics(csv_file, t_conf, loss, acc, roc_auc, best_thresh)#pre_loss, pre_acc
        stop_task(t_conf, task_id)
        task_id+=1

    elif task_state == 'Started':
        #complete_task(task_id)
        print("I need to complete this task")
        model.load_weights("weights.best.hdf5")
        stop_task(t_conf, task_id)
        task_id+=1
    else:
        print("unexpected task state for the task number: ", task_id)
        print(task_state)

#visualize_multi_dim(date)
