from keras import backend as K
import os
import time
import h5py
import sys
from keras.models import Model,model_from_json
from keras.optimizers import SGD
from keras.optimizers import Adam 
import numpy as np
from keras.utils import np_utils
from math import floor
from music_tagger_ftuning import music_tagger_wrapper
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from utils import save_data, load_dataset, save_dataset, sort_result, predict_label, load_gt, plot_confusion_matrix, extract_melgrams
''' In this file, we use the pre-trained model and by calling our new tuned model, we want to sweep to
    operate parameter tuning.
    
    
    Here are the two hyper parameter I intend to tune
    
    1. Optimization method.
        a) adam
        b)momentum
    2.Number of GRU layers at the end
        a) 3GRU
        b) 2GRU + LSTM
        
        
        NOTE: in pre-trained model there are 10 genres, ww here, train on 7 genres
'''

K.tensorflow_backend._get_available_gpus()
TRAIN = 1
TEST = 0

SAVE_MODEL = 1
SAVE_WEIGHTS = 1

LOAD_MODEL = 0
LOAD_WEIGHTS = 1

# Dataset
MULTIFRAMES = 0
SAVE_DB = 0
LOAD_DB = 1

# Model parameters
nb_classes = 7
nb_epoch = 50
batch_size = 16

time_elapsed = 0


# GTZAN Dataset Tags
tags = ['blues', 'classical', 'country', 'hiphop', 'jazz',  'pop',  'rock']
tags = np.array(tags)

# Paths to set
model_name = "adam_bs16_lr0p001_NoFreeze"
model_path = "mix_adam_trained/" + model_name + "/"
weights_path = "mix_adam_trained/" + model_name + "/weights/"

#model_name = "sgd"
#model_path = "sgd_trained/" + model_name + "/"
#weights_path = "sgd_trained/" + model_name + "/weights/"






# Create directories for the models & weights
if not os.path.exists(model_path):
    os.makedirs(model_path)
    print 'Path created: ', model_path

if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print 'Path created: ', weights_path

# Divide the song into multiple frames of 29.1s or center crop
# train_songs_list = "music/train_FMA/train_FMA_list_shuffled.txt"
# train_songs_tags = "music/train_FMA/train_FMA_labels_shuffled.txt"
train_songs_list = "lists/train_FMA_list_shuffled.txt"
train_songs_tags = "lists/train_FMA_labels_shuffled.txt"
test_songs_list = "lists/train_GTZAN_list_shuffled.txt"
test_songs_tags = "lists/train_GTZAN_labels_shuffled.txt"




# Data Loading
if LOAD_DB:
    X_train, y_train,num_frames_train = load_dataset('music_dataset_spectogram/train/music_dataset_train_mix.h5')
    X_test, y_test , num_frames_test = load_dataset('music_dataset_spectogram/train/music_dataset_test_mix.h5');
# Compute mel-spectogram for all the frames
else:
    #We dont care about MULTFRAMES and num_song_genres here, train set is 30s long each song
    X_train, num_frames_train = extract_melgrams(train_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre='')
    print('X_train shape:', X_train.shape)
#     X_test, y_test, num_frames_test = extract_melgrams(test_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre=10)

print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')
#y_train  = np.loadtxt(train_songs_tags,dtype=type(int()))
#y_test  = np.loadtxt(test_songs_tags,dtype=type(int()))


y_train = np.array(y_train)
y_test = np.array(y_test)

if SAVE_DB:
    if MULTIFRAMES:
        save_dataset('music_dataset_spectogram/train/music_dataset_multiframe_train.h5', X_train, y_train,num_frames_train)
        # save_dataset('music_dataset/music_dataset_multiframe_test.h5', X_test,y_test,num_frames_test)
    else:
        save_dataset('music_dataset_spectogram/train/music_dataset_FMA.h5', X_train,y_train ,0)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print 'Shape labels y_train: ', Y_train.shape
print 'Shape labels y_test: ', Y_test.shape



# Initialize model
model = music_tagger_wrapper(LOAD_WEIGHTS)
#model = MusicTaggerCNN(weights='msd', input_tensor=(1, 96, 1366))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.summary()
if LOAD_MODEL: 
    json_file = open(model_path+model_name+".json", 'r')
    loaded_model_json= json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.load_weights(weights_path+model_name+'_epoch_30.h5', by_name=True)
    model  = Model(model.input,model.output) 
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


# Save model architecture
if SAVE_MODEL:
    json_string = model.to_json()
    f = open(model_path+model_name+".json", 'w')
    f.write(json_string)
    f.close()
# Train model
if TRAIN:
    try:
        print ("Training the model")
        f_train = open(model_path+model_name+"_scores_training.txt", 'w')
        f_test = open(model_path+model_name+"_scores_test.txt", 'w')
        f_scores = open(model_path+model_name+"_scores.txt", 'w')
        for epoch in range(1,nb_epoch+1):
            t0 = time.time()
            print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))
            sys.stdout.flush()
            scores = model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=2, validation_data=(X_test, Y_test))
            time_elapsed = time_elapsed + time.time() - t0
            print ("Time Elapsed: " +str(time_elapsed))
            sys.stdout.flush()

            score_train = model.evaluate(X_train, Y_train, verbose=0)
            print('Train Loss:', score_train[0])
            print('Train Accuracy:', score_train[1])
            f_train.write(str(score_train)+"\n")

            score_test = model.evaluate(X_test, Y_test, verbose=0)
            print('Test Loss:', score_test[0])
            print('Test Accuracy:', score_test[1])
            f_test.write(str(score_test)+"\n")

            f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1]) + "\n")
            f_scores.write(str(score_train[0])+","+str(score_train[1])+ "\n")
            if SAVE_WEIGHTS and epoch % 5 == 0:
                model.save_weights(weights_path + model_name + "_epoch_" + str(epoch) + ".h5")
                print("Saved model to disk in: " + weights_path + model_name + "_epoch" + str(epoch) + ".h5")

        f_train.close()
        f_test.close()
        f_scores.close()

        # Save time elapsed
        f = open(model_path+model_name+"_time_elapsed.txt", 'w')
        f.write(str(time_elapsed))
        f.close()

    # Save files when an sudden close happens / ctrl C
    except:
        f_train.close()
        f_test.close()
        f_scores.close()
        # Save time elapsed
        f = open(model_path + model_name + "_time_elapsed.txt", 'w')
        f.write(str(time_elapsed))
        f.close()
    finally:
        f_train.close()
        f_test.close()
        f_scores.close()
        # Save time elapsed
        f = open(model_path + model_name + "_time_elapsed.txt", 'w')
        f.write(str(time_elapsed))
        f.close()

if TEST:
    f_test = open(model_path+model_name+"_scores_test.txt", 'w')
        
    t0 = time.time()
    print 'Predicting...','\n'

    score_test = model.evaluate(X_test, Y_test, verbose=0)
    print('Test Loss:', score_test[0])
    print('Test Accuracy:', score_test[1])
    f_test.write(str(score_test)+"\n")

            # f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1]) + "\n")
            




