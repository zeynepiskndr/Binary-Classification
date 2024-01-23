from matplotlib.pyplot import step
import tensorflow as tf
import tensorflow
import utils as util
import pandas as pd
import numpy as np
import time
import csv

from configurators import train_config, generator_config, dataset_config
from visualizer import plot_confusion_matrix, plot_roc_curve
from utils import write_to_log, save_var, save_model, load_model, make_subfolder, read_best_threshold

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, LearningRateScheduler
from tensorflow.keras.applications import VGG16, ResNet50
from efficientnet import tfkeras as efficientnet

def get_model(t_conf):
    if t_conf.MODEL_NAME == 'EfficientNetB0':
        base_model = efficientnet.EfficientNetB1(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )
    elif t_conf.MODEL_NAME == 'EfficientNetB1':
        base_model = efficientnet.EfficientNetB4(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )
    elif t_conf.MODEL_NAME == 'EfficientNetB2':
        base_model = efficientnet.EfficientNetB4(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )
    elif t_conf.MODEL_NAME == 'EfficientNetB3':
        base_model = efficientnet.EfficientNetB4(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )
    elif t_conf.MODEL_NAME == 'EfficientNetB4':
        base_model = efficientnet.EfficientNetB4(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )
    elif t_conf.MODEL_NAME == 'EfficientNetB5':
        base_model = efficientnet.EfficientNetB5(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )
    elif t_conf.MODEL_NAME == 'EfficientNetB6':
        base_model = efficientnet.EfficientNetB4(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )
    elif t_conf.MODEL_NAME == 'EfficientNetB7':
        base_model = efficientnet.EfficientNetB7(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )
    elif t_conf.MODEL_NAME == 'ResNet50':
        base_model = tensorflow.keras.applications.ResNet50(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )
    elif t_conf.MODEL_NAME == 'VGG16':
        base_model = tensorflow.keras.applications.VGG16(
            weights = 'imagenet',
            input_shape = (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3),
            include_top = False
            )

    base_model.trainable = False

    inputs = Input(shape=(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3))
    x = base_model(inputs, training = False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(t_conf.DENSE_LAYER, activation='relu')(x)
    x = Dropout(t_conf.DROP_OUT)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

def train_model(model,t_conf,g_conf,d_conf,seed=42):
    '''
    One stage train 
    '''
    log_dir = t_conf.log_dir
    df_train = pd.read_csv(d_conf.train_df_path)
    df_train['Label'] = df_train['Label'].astype('str')
    X_train = df_train[df_train['Subsets'] == 'Training']
    X_valid = df_train[df_train['Subsets'] == 'Validation']
    X_train = X_train.reset_index(drop = True)
    X_valid = X_valid.reset_index(drop = True)

    train_datagen = ImageDataGenerator(
                        rotation_range      = g_conf.rotation_range,
                        horizontal_flip     = g_conf.horizontal_flip,
                        vertical_flip       = g_conf.vertical_flip,
                        brightness_range    = g_conf.brightness_range,
                        rescale             = g_conf.rescale)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe   =X_train,
            directory   =d_conf.train_path,
            x_col       ='Input',
            y_col       ='Background',
            target_size =(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            class_mode  ='binary',
            batch_size  =t_conf.BATCH_SIZE,
            seed        =seed,
            shuffle     =True)

    val_datagen = ImageDataGenerator(rescale = g_conf.rescale)
    validation_generator = val_datagen.flow_from_dataframe(
            dataframe=X_valid,
            directory=d_conf.train_path,
            x_col="Input",
            y_col="Background",
            target_size=(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            class_mode='binary',
            batch_size=t_conf.BATCH_SIZE,
            seed=seed,
            shuffle=False)

    #callbacks
    weights_dir = make_subfolder("Weights", log_dir)
    my_filepath = weights_dir + 'best_weight.hdf5'
    mc = ModelCheckpoint(
        filepath=my_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    es = EarlyStopping(monitor='val_loss', mode='min', patience = t_conf.ES_PATIENCE, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    optimizer=tf.keras.optimizers.Adam(lr=t_conf.WARMUP_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    STEP_SIZE_TRAIN=train_generator.n//(train_generator.batch_size)
    STEP_SIZE_VALID=validation_generator.n//(validation_generator.batch_size)

    #Fit-pretrain model
    start_time = time.time()
    history = model.fit(
                    train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=t_conf.WARMUP_EPOCHS,
                    validation_steps=STEP_SIZE_VALID,
                    validation_data=validation_generator,
                    callbacks=[mc, es, reduce_lr]
                    ).history

    elapsed_time = time.time() - start_time
    train_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    util.write_to_log(log_dir, "\n\nElapsed time in the pre-train: " + train_time)
    start_time = time.time()

    util.save_var(history, log_dir + "history")
    #Save model and weights
    util.save_model(model, log_dir + t_conf.MODEL_NAME +  '_trained_model')

    return model, history

def pre_train_model(t_conf,g_conf,d_conf,seed=42):
    '''
    the first stage of the two-stage train 
    '''
    log_dir = t_conf.log_dir

    df_train = pd.read_csv(d_conf.train_df_path)
    df_train['Label'] = df_train['Label'].astype('str')
    X_train = df_train[df_train['Subsets'] == 'Training']
    X_valid = df_train[df_train['Subsets'] == 'Validation']
    X_train = X_train.reset_index(drop = True)
    X_valid = X_valid.reset_index(drop = True)

    train_datagen = ImageDataGenerator(
                        rotation_range      = g_conf.rotation_range,
                        horizontal_flip     = g_conf.horizontal_flip,
                        vertical_flip       = g_conf.vertical_flip,
                        brightness_range    = g_conf.brightness_range,
                        rescale             = g_conf.rescale)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe   =X_train,
            directory   =d_conf.train_path,
            x_col       ='Images',
            y_col       ='Label',
            target_size =(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            class_mode  ='binary',
            batch_size  =t_conf.BATCH_SIZE,
            seed        =seed,
            shuffle     =True)

    val_datagen = ImageDataGenerator(rescale = g_conf.rescale)
    validation_generator = val_datagen.flow_from_dataframe(
            dataframe=X_valid,
            directory=d_conf.train_path,
            x_col="Images",
            y_col="Label",
            target_size=(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            class_mode='binary',
            batch_size=t_conf.BATCH_SIZE,
            seed=seed,
            shuffle=False)

    model = get_model(t_conf)
    model.summary()

    #callbacks
    weights_dir = make_subfolder("Weights", log_dir)
    my_filepath = weights_dir + 'best_weight-{epoch:02d}.hdf5'
    mc = ModelCheckpoint(
        filepath=my_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    es = EarlyStopping(monitor='val_loss', mode='min', patience = t_conf.ES_PATIENCE, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
    optimizer=tf.keras.optimizers.Adam(lr=t_conf.WARMUP_LEARNING_RATE)
    model.layers[1].trainable = False
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    STEP_SIZE_TRAIN=train_generator.n//(train_generator.batch_size)
    STEP_SIZE_VALID=validation_generator.n//(validation_generator.batch_size)

    model.summary()
    #Fit-train model
    start_time = time.time()
    history = model.fit(
                    train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=t_conf.WARMUP_EPOCHS,
                    validation_steps=STEP_SIZE_VALID,
                    validation_data=validation_generator,
                    callbacks=[mc, es, reduce_lr]
                    ).history

    write_to_log(log_dir, "\n\nModel Name: " + t_conf.MODEL_NAME)
    write_to_log(log_dir, "\n\nDense Layer: " + str(t_conf.DENSE_LAYER))
    write_to_log(log_dir, "\n\nDropout: " + str(t_conf.DROP_OUT))

    elapsed_time = time.time() - start_time
    pretrain_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    write_to_log(log_dir, "\n\nElapsed time in the pretrain: " + pretrain_time)

    save_var(history, log_dir + "history_pretrained")
    save_model(model, log_dir + t_conf.MODEL_NAME + "model_pretrained")

    return model, history

def finetune_model(model,t_conf,g_conf,d_conf,seed=42):
    '''
    second stage of the two-stage train 
    '''
    log_dir = t_conf.log_dir
    df_train = pd.read_csv(d_conf.train_df_path)
    df_train['Label'] = df_train['Label'].astype('str')
    X_train = df_train[df_train['Subsets'] == 'Training']
    X_valid = df_train[df_train['Subsets'] == 'Validation']
    X_train = X_train.reset_index(drop = True)
    X_valid = X_valid.reset_index(drop = True)

    train_datagen = ImageDataGenerator(
                        rotation_range      = g_conf.rotation_range,
                        horizontal_flip     = g_conf.horizontal_flip,
                        vertical_flip       = g_conf.vertical_flip,
                        brightness_range    = g_conf.brightness_range,
                        rescale             = g_conf.rescale)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe   =X_train,
            directory   =d_conf.train_path,
            x_col       ='Images',
            y_col       ='Label',
            target_size =(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            class_mode  ='binary',
            batch_size  =t_conf.BATCH_SIZE,
            seed        =seed,
            shuffle     =True)

    val_datagen = ImageDataGenerator(rescale = g_conf.rescale)
    validation_generator = val_datagen.flow_from_dataframe(
            dataframe=X_valid,
            directory=d_conf.train_path,
            x_col="Images",
            y_col="Label",
            target_size=(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            class_mode='binary',
            batch_size=t_conf.BATCH_SIZE,
            seed=seed,
            shuffle=False)

    model.summary()

   #callbacks
    weights_dir = make_subfolder("Weights", log_dir)
    my_filepath = weights_dir + 'best_weight.hdf5'
    mc = ModelCheckpoint(
        filepath=my_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    es = EarlyStopping(monitor='val_loss', mode='min', patience = t_conf.ES_PATIENCE, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    optimizer=tf.keras.optimizers.Adam(lr=t_conf.LEARNING_RATE)
    model.layers[1].trainable = True
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    STEP_SIZE_TRAIN=train_generator.n//(train_generator.batch_size)
    STEP_SIZE_VALID=validation_generator.n//(validation_generator.batch_size)
    model.summary()
    #Fit-pretrain model
    start_time = time.time()
    history = model.fit(
                    train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=t_conf.EPOCHS,
                    validation_steps=STEP_SIZE_VALID,
                    validation_data=validation_generator,
                    callbacks=[mc, es, reduce_lr]
                    ).history

    elapsed_time = time.time() - start_time
    train_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    util.write_to_log(log_dir, "\n\nElapsed time in the fine-train: " + train_time)
    start_time = time.time()

    util.save_var(history, log_dir + "history_finetune")
    #Save model and weights
    util.save_model(model, log_dir + t_conf.MODEL_NAME +  '_finetuned_model')

    return model, history

def full_train_model(model,t_conf,g_conf,d_conf,seed=42):
    '''
    Two-stage train 
    '''
    model, history_pretrain = pre_train_model(model,t_conf,g_conf,d_conf)
    pre_loss, pre_acc, pre_roc_auc, pre_best_thresh = evaluate_model(model,t_conf,g_conf,d_conf)
    model, history_finetune = finetune_model(model,t_conf,g_conf,d_conf)   
    pre_loss, pre_acc, pre_roc_auc, pre_best_thresh = evaluate_model(model,t_conf,g_conf,d_conf)
    return model, history_pretrain, history_finetune, pre_loss, pre_acc

def evaluate_model(model,t_conf,g_conf,d_conf,seed=42):
    log_dir = t_conf.log_dir
    df_train = pd.read_csv(d_conf.train_df_path)
    df_train['Label'] = df_train['Label'].astype('str')
    X_train = df_train[df_train['Subsets'] == 'Training']
    X_valid = df_train[df_train['Subsets'] == 'Validation']
    X_train = X_train.reset_index(drop = True)
    X_valid = X_valid.reset_index(drop = True)

    val_datagen = ImageDataGenerator(rescale = g_conf.rescale)
    validation_generator = val_datagen.flow_from_dataframe(
            dataframe   =X_valid,
            directory   =d_conf.train_path,
            x_col       ="Images",
            y_col       ="Label",
            target_size =(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            class_mode  ='binary',
            batch_size  =t_conf.BATCH_SIZE,
            seed        =seed,
            shuffle     =False)

    scores = model.evaluate(validation_generator)
    write_to_log(log_dir, "\n\nTest Loss Trained:{}".format(scores[0]))
    write_to_log(log_dir, "\nTest Accuracy Trained:{}".format(scores[1]))
    print("Test Loss Trained:{}".format(scores[0]))
    print("Test Accuracy Pretrained:{}".format(scores[1]))
    loss = scores[0]
    acc = scores[1]

    #Save Confusion matrix
    y_pred = model.predict(validation_generator)
    y_true = validation_generator.classes

    roc_auc, best_thresh = plot_roc_curve(y_true, y_pred, save=True, saveDir=log_dir)
    write_to_log(log_dir, "\nROC AUC:{}".format(roc_auc))
    write_to_log(log_dir, "\nBest Threshold:{}".format(best_thresh))
    
    pred = np.zeros(y_pred.shape)
    pred[y_pred>best_thresh]=1

    #prc = plot_precision_recall_curve(y_true, y_pred, save=True, saveDir=log_dir)
    report, sensitivity, specificity, tn, fp, fn, tp = plot_confusion_matrix(y_true, pred, classes=[0, 1], save=True, saveDir=log_dir, normalize=True)
    write_to_log(log_dir, "\nSensitivity:{}".format(sensitivity))
    write_to_log(log_dir, "\nSpecificity:{}".format(specificity))

    write_to_log(log_dir, "\nTN:{}".format(tn))
    write_to_log(log_dir, "\nFP:{}".format(fp))
    write_to_log(log_dir, "\nFN:{}".format(fn))
    write_to_log(log_dir, "\nTP:{}".format(tp))

    return loss, acc, roc_auc, best_thresh

def evaluate_model_testset(comp_info,t_conf,g_conf,d_conf,seed=42):
    #load model  
    loaded_model = load_model(comp_info["Codes"]+'/out/2021_November_24-12_18_03/model_trained')
    log_dir = t_conf.log_dir

    df_test = pd.read_csv(d_conf.test_df_path)
    df_test['Label'] = df_test['Label'].astype('str')
    X_test = df_test
    print(X_test)
    train_datagen = ImageDataGenerator(rescale = g_conf.rescale)
    test_generator = train_datagen.flow_from_dataframe(
            dataframe   =X_test,
            directory   =d_conf.test_path,
            x_col       ="Images",
            y_col       ="Label",
            target_size =(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH),
            class_mode  ='binary',
            batch_size  =t_conf.BATCH_SIZE,
            seed        =seed,
            shuffle     =False)
    optimizer=tf.keras.optimizers.Adam(lr=t_conf.LEARNING_RATE)
    loaded_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    scores = loaded_model.evaluate(test_generator)
    write_to_log(log_dir, "\n")

    write_to_log(log_dir, "\n\nTest Loss Trained:{}".format(scores[0]))
    write_to_log(log_dir, "\nTest Accuracy Trained:{}".format(scores[1]))
    print("Test Loss Trained:{}".format(scores[0]))
    print("Test Accuracy Trained:{}".format(scores[1]))
    loss = scores[0]
    acc = scores[1]

    #Save Confusion matrix
    y_pred = loaded_model.predict(test_generator)
    y_true = test_generator.classes

    roc_auc, best_thresh = plot_roc_curve(y_true, y_pred, save=True, saveDir=log_dir)
    print(best_thresh)
    write_to_log(log_dir, "\nROC AUC:{}".format(roc_auc))
    write_to_log(log_dir, "\nBest Threshold:{}".format(best_thresh))
    '''
    best_thresh = read_best_threshold(comp_info)
    print(best_thresh)
    '''
    pred = np.zeros(y_pred.shape)
    pred[y_pred>best_thresh]=1

    #prc = plot_precision_recall_curve(y_true, y_pred, save=True, saveDir=log_dir)
    report, sensitivity, specificity, tn, fp, fn, tp = plot_confusion_matrix(y_true, pred, classes=[0, 1], save=True, saveDir=log_dir, normalize=True)
    write_to_log(log_dir, "\nSensitivity:{}".format(sensitivity))
    write_to_log(log_dir, "\nSpecificity:{}".format(specificity))

    write_to_log(log_dir, "\nTN:{}".format(tn))
    write_to_log(log_dir, "\nFP:{}".format(fp))
    write_to_log(log_dir, "\nFN:{}".format(fn))
    write_to_log(log_dir, "\nTP:{}".format(tp))

    return loss, acc, roc_auc, best_thresh

def tta_pred_one_image(model,t_conf,g_conf,image):
    log_dir = t_conf.log_dir
    datagen = ImageDataGenerator(
                        rotation_range      = g_conf.rotation_range,
                        horizontal_flip     = g_conf.horizontal_flip,
                        vertical_flip       = g_conf.vertical_flip,
                        brightness_range    = g_conf.brightness_range,
                        rescale             = g_conf.rescale)

    # convert image into dataset
    samples = np.expand_dims(image, 0)
    tta_steps = 5
    predictions = []

    for i in range(tta_steps):

        preds = model.predict_generator(datagen.flow(samples, batch_size=1, shuffle=False))
        #print(preds)
        predictions.append(preds)
        #print(predictions)

    pred = np.mean(predictions, axis=0)
    #print(pred)
    
    pred_round = pred.round()
    pred_round = pred_round.astype(np.int)
    #print(pred)

    return pred, pred_round

# make a prediction using test-time augmentation
def tta_dataset_acc(comp_info,t_conf,g_conf,X_valid):
    #load model
    loaded_model = load_model(comp_info["Codes"]+'/out/2021_November_29-16_24_01/model_trained')
    start_time = time.time()
    log_dir = t_conf.log_dir
    results = []
    preds = []
    y_true = []
    file = comp_info["Datasets"]+'/csvDataset/test_dataset.csv'
    with open(file) as f:
        reader = csv.reader(f, delimiter=",")
        for index in X_valid.index:
            my_image = load_img(comp_info["Datasets"]+'TestDataset/' + X_valid.iloc[index, 0])
            my_image = img_to_array(my_image)
            my_image_label = X_valid.iloc[index, 1]
            label = np.array(my_image_label).astype(np.int)
    
            pred, pred_round = tta_pred_one_image(loaded_model,t_conf,g_conf,my_image)
            #print(pred)
            #print(np.equal(label, pred))
            result = np.mean(np.equal(label, pred_round))
            #print(result)
            results.append(result)
            pred = pred[0].astype(np.float32)
            #print(pred)
            preds.append(pred)
            y_true.append(label)
            #print(preds)
            #print(y_true)

    #roc_auc, best_thresh = plot_roc_curve(y_true, preds, save=True, saveDir=log_dir)
    #print(best_thresh)
    uniques, counts = np.unique(results, return_counts=True)
    print(uniques)
    print(counts)
    percentages = dict(zip(uniques, counts * 100 / len(results)))
    print(percentages)

    elapsed_time = time.time() - start_time
    tta_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    write_to_log(log_dir, "\n\nElapsed time in the tta dataset acc: " + tta_time)

    write_to_log(log_dir, "\nTTA Result:{}".format(percentages))
