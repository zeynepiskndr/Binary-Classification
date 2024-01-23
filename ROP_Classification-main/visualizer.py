from PIL.Image import preinit
import tensorflow as tf
import numpy as np
import os
import utils as util
import matplotlib.pyplot as plt
from utils import write_to_log, read_var
import plotly.express as px
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score, precision_recall_curve, accuracy_score
from configurators import train_config, generator_config, dataset_config
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def plot_history(history):
    '''
    Takes history dictionary and plot accuracy and loss graph
    '''
    plot_acc(history)
    plot_loss(history)

def plot_loss_and_acc(history, save=False, saveDir='out/', fname=''):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 18))

    loss = history['loss']
    val_loss = history['val_loss']

    acc = history['accuracy']
    val_acc = history['val_accuracy']

    ax1.plot(loss, label='Train loss')
    ax1.plot(val_loss, label='Validation loss')
    ax1.legend(loc='best')
    ax1.set_title('Loss')

    ax2.plot(acc, label='Train accuracy')
    ax2.plot(val_acc, label='Validation accuracy')
    ax2.legend(loc='best')
    ax2.set_title('Accuracy')

    plt.xlabel('Epochs')
    if(save):
        plt.savefig(saveDir + fname + 'loss_and_acc.png')
    plt.show(block=False)
    plt.close()

def plot_full_loss_and_acc(t_conf, history, history_fine, save=False, saveDir='out/', fname=''):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 18))

    number_of_epochs_it_ran = len(history['loss'])

    pre_loss = history['loss']
    pre_val_loss = history['val_loss']

    pre_acc = history['accuracy']
    pre_val_acc = history['val_accuracy']
    
    loss = pre_loss + history_fine['loss']
    val_loss = pre_val_loss + history_fine['val_loss']

    acc = pre_acc + history_fine['accuracy']
    val_acc = pre_val_acc + history_fine['val_accuracy']
    
    ax1.plot(loss, label='Train loss')
    ax1.plot(val_loss, label='Validation loss')
    ax1.plot([number_of_epochs_it_ran-1,number_of_epochs_it_ran-1], plt.ylim(), label='Start Fine Tuning')
    ax1.legend(loc='best')
    ax1.set_title('Loss')

    ax2.plot(acc, label='Train accuracy')
    ax2.plot(val_acc, label='Validation accuracy')
    ax2.plot([number_of_epochs_it_ran-1,number_of_epochs_it_ran-1], plt.ylim(), label='Start Fine Tuning')
    ax2.legend(loc='best')
    ax2.set_title('Accuracy')

    plt.xlabel('Epochs')
    if(save):
        plt.savefig(saveDir + fname + 'full_loss_and_acc.png')
    plt.show(block=False)
    plt.close()

def plot_acc(history):
    '''
    TODO:Instead of history whats should be the input?
    Takes array of Plot training & validation accuracy values
    '''
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_loss(history):
    '''
    TODO:Instead of history whats should be the input?
    Plot training & validation loss values
    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_confusion_matrix(y_true, pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=False,
                          saveDir='out/'):
    """
    This function calculates, prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    cm = confusion_matrix(y_true, pred)
    report = classification_report(y_true, pred)
    print(report)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    #extracting true_positives, false_positives, true_negatives, false_negatives
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print("True Negatives: ",tn)
    print("False Positives: ",fp)
    print("False Negatives: ",fn)
    print("True Positives: ",tp)

    #Specificity 
    specificity = tn/(tn+fp)
    print("Specificity {:0.2f}".format(specificity))

    #Sensitivity
    sensitivity = tp/(tp+fn)
    print("Sensitivity {:0.2f}".format(sensitivity))

    if(save):
        plt.savefig(saveDir + title+ '.png')
    plt.show(block=False)
    plt.close()

    return report, sensitivity, specificity, tn, fp, fn, tp

def plot_roc_curve(y_true, y_pred,
                    title='ROC', 
                    save=False, saveDir='out/'):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    #print("Threshols: ", thresholds)
    
    roc_auc = auc(fpr, tpr)
    print('AUC: %.3f' % roc_auc)
    #Optimal Threshold for ROC Curve
    # get the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    #print('ROC Best Threshold=%f' % (best_thresh))

    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic example')
    plt.legend()

    if(save):
        plt.savefig(saveDir + title+ '.png')
    plt.show(block=False)
    plt.close()

    return roc_auc, best_thresh

def plot_precision_recall_curve(y_true, y_pred,
                            title='Presicion Recall Curve', 
                            save=False, saveDir='out/'):
                            # calculate pr-curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # calculate the precision-recall auc
    auc_score = auc(recall, precision)
    print('PR AUC: %.3f' % auc_score)

    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('PR Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    
    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='Logistic')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    if(save):
        plt.savefig(saveDir + title+ '.png')
    plt.show(block=False)

def visualise_batches(train_conf,generator_conf,dataset_conf):
    train_dest_path = dataset_conf.get_dataset()
    log_dir = train_conf.log_dir
    train_datagen = ImageDataGenerator(
                        rotation_range      = generator_conf.rotation_range,
                        width_shift_range   = generator_conf.width_shift_range,
                        height_shift_range  = generator_conf.height_shift_range,
                        shear_range         = generator_conf.shear_range,
                        zoom_range          = generator_conf.zoom_range,
                        channel_shift_range = generator_conf.channel_shift_range,
                        validation_split    = 0.2,
                        horizontal_flip     = generator_conf.horizontal_flip,
                        vertical_flip       = generator_conf.vertical_flip,
                        fill_mode           = generator_conf.fill_mode,
                        brightness_range    = generator_conf.brightness_range,
                        rescale             = generator_conf.rescale)

    train_generator = train_datagen.flow_from_directory(
            train_dest_path,
            target_size =(train_conf.IMG_HEIGHT, train_conf.IMG_WIDTH),
            batch_size  =train_conf.BATCH_SIZE,
            subset      ="training",
            class_mode  ='binary',
            shuffle     =True)
  
    for i in range(5):
        batch = train_generator.next()
        util.show_images([batch[0][0],batch[0][1],batch[0][2],batch[0][3]])

def visualize_multi_dim(comp_info, date):
    """
    file = comp_info["Codes"]+'out/2021_November_17-11_56_28/'

    g_conf_path,t_conf_path,d_conf_path = get_configs_from_dir(file)
    g_conf,t_conf,d_conf = read_configs(g_conf_path,t_conf_path,d_conf_path)

    print(g_conf)
    print(t_conf)
    print(d_conf)
    """
    df = pd.read_csv(f'out/Train_Metrics_{date}.csv')
    #df = pd.read_csv(f'out/Train_Metrics_2022_March_02.csv')
    fig = px.parallel_coordinates(df,color = "ACCURACY",
                                dimensions=['MODEL_NAME', 'IMG_HEIGHT', 'BATCH_SIZE', 'WARMUP_LEARNING_RATE', 'LEARNING_RATE', 'DENSE_LAYER', 'DROP_OUT', 'ACCURACY'],
                                color_continuous_scale=px.colors.sequential.Inferno)

    fig.show()

def get_configs_from_dir(log_dir):
    print('\nSearching for config files...')
    g_conf_path, t_conf_path, d_conf_path = "not found","not found","not found"
    entries = sorted(os.listdir(log_dir))
    for entry in entries:
        try:
            fname, format = entry.split(".")
        except:
            if("generator_config" in entry):
                print(entry)
                g_conf_path = log_dir+entry
            if("train_config" in entry):
                print(entry)
                t_conf_path = log_dir+entry
            if("dataset_config" in entry):
                print(entry)
                d_conf_path = log_dir+entry
    return g_conf_path, t_conf_path, d_conf_path

def read_configs(g_conf_path,t_conf_path,d_conf_path):
    try:
        g_conf= util.read_var(g_conf_path)
    except:
        g_conf = generator_config()
    try:
        t_conf= util.read_var(t_conf_path)
    except:
        t_conf = train_config()
    try:
        d_conf= util.read_var(d_conf_path)
    except:
        d_conf = dataset_config()
    return g_conf, t_conf, d_conf

if __name__ == "__main__":

    t_conf = train_config(name = "default",
                    BATCH_SIZE = 16,
                    EPOCHS=5)

    g_conf = generator_config(name = "default")
    d_conf = dataset_config(name = "default")
    visualise_batches(t_conf,g_conf,d_conf)
