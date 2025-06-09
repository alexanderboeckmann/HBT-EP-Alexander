#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import fsspec, os
import scipy as sp
import PIL
from PIL import Image
import random
import os
import glob
import re


# In[14]:


# Define shot lists and file paths
shot_list = [119591, 119599, 119601, 119646, 119648, 119653, 119654, 119658, 119659,
             119661, 119662, 119663, 119665, 119666, 119667, 119669, 119670, 119671,
             119673, 119675, 119748, 119750, 119751, 119752, 119754, 119755, 119756,
             119757, 119760, 119761, 119762, 119763, 119764, 119766, 119767, 119768, 119769]

file_path = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HighFreqMLModeling/Training/Input Data/Shots/'
#file_path_hbt = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HighFreqMLModeling/Training/python_hbteplib_data/' #newer data
file_path_hbt = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HighFreqMLModeling/Training/Old Shots/' #older data


TARGET_FRAME_COUNT = 800
CAMERA_DEPTH = 65535.0 # 2^16


# In[16]:


# Helper functions for data processing
def determine_frame_ratio(num_frames, target_frames=TARGET_FRAME_COUNT):
    """
    Determines the frame ratio needed to downsample the data to target_frames.
    Returns the ratio and the actual number of frames after downsampling.
    """
    ratio = max(1, num_frames // target_frames)
    actual_frames = num_frames // ratio
    return ratio, actual_frames

def process_shot_data(folder_path, target_frame_count=TARGET_FRAME_COUNT, max_pixel_value=CAMERA_DEPTH):
    """
    Process a single shot's data with automatic frame rate handling.
    Returns: 2D data, cut 2D data, and flat data for the shot
    """
    tiff_files = sorted(glob.glob(os.path.join(folder_path, "*.tiff")))
    num_frames = len(tiff_files)
    
    if num_frames == 0:
        raise ValueError(f"No TIFF files found in {folder_path}")
    
    frame_ratio, actual_frames = determine_frame_ratio(num_frames, target_frame_count)
    
    # Initialize shot lists
    flat_shot = []
    shot_2d = []
    cut_shot = []
    
    # Process TIFF files with dynamic frame ratio
    for j, tiff_file in enumerate(tiff_files):
        if j % frame_ratio == 0 and len(shot_2d) < target_frame_count:
            try:
                im = Image.open(tiff_file)
                im = np.array(im, dtype=np.float32)
                im = im / max_pixel_value
                flat_im = im.reshape(-1)
                cut_2d = im[:, 48:-48]
                
                shot_2d.append(im)
                flat_shot.append(flat_im)
                cut_shot.append(cut_2d)
                
            except Exception as e:
                print(f"Error loading {tiff_file}: {e}")
                continue
    
    return np.array(shot_2d), np.array(cut_shot), np.array(flat_shot)

def process_all_shots(shot_list, base_path, target_frame_count=TARGET_FRAME_COUNT):
    """
    Process multiple shots with automatic frame rate handling
    """
    training_data_2D = []
    cut_training_data_2D = []
    flat_training_data = []
    
    for shot in shot_list:
        folder_path = os.path.join(base_path, str(shot), 'CAM-26731/tiff/')
        try:
            shot_2d, cut_2d, flat_data = process_shot_data(folder_path, target_frame_count)
            
            if len(shot_2d) == target_frame_count:
                training_data_2D.append(shot_2d)
                cut_training_data_2D.append(cut_2d)
                flat_training_data.append(flat_data)
            else:
                print(f"Shot {shot} produced {len(shot_2d)} frames, expected {target_frame_count}. Skipping.")
                
        except Exception as e:
            print(f"Error processing shot {shot}: {e}")
            continue
    
    return (np.array(training_data_2D), 
            np.array(cut_training_data_2D), 
            np.array(flat_training_data))


# In[18]:


# Load and format HBT data
def format_hbt_data(data, mode_num):
    # Determine frame ratio
    original_length = data[0][0].shape[0] # 5000
    target_length = TARGET_FRAME_COUNT
    frame_ratio = original_length // target_length  # generally 5 (5000/TARGET_FRAME_COUNT)
    
    data = np.asarray(data, dtype=float)
    data = np.reshape(data[:,mode_num-1,:], (len(shot_list), original_length, 1))
    data = data[:,::frame_ratio,:]
    data = data[:,:target_length,:]
    return data

# Load HBT data
hbt_ma_data = []
hbt_mp_data = []
hbt_time_data = []
for i in range(len(shot_list)):
    ma_list = []
    mp_list = []
    for j in range(1,5):
        ma_data = np.load(file_path_hbt+str(shot_list[i])+'m'+str(j)+'Amp.npy')
        mp_data = np.load(file_path_hbt+str(shot_list[i])+'m'+str(j)+'Phase.npy')
        ma_list.append(ma_data)
        mp_list.append(mp_data)
    
    hbt_ma_data.append(ma_list)
    hbt_mp_data.append(mp_list)
    time_data = np.load(file_path_hbt+str(shot_list[i])+'time.npy')
    hbt_time_data.append(time_data)

# Format HBT data
hbt_ma1_data = format_hbt_data(hbt_ma_data, 1)
hbt_ma2_data = format_hbt_data(hbt_ma_data, 2)
hbt_ma3_data = format_hbt_data(hbt_ma_data, 3)
hbt_ma4_data = format_hbt_data(hbt_ma_data, 4)

hbt_mp1_data = format_hbt_data(hbt_mp_data, 1)
hbt_mp2_data = format_hbt_data(hbt_mp_data, 2)
hbt_mp3_data = format_hbt_data(hbt_mp_data, 3)
hbt_mp4_data = format_hbt_data(hbt_mp_data, 4)

# Format time data
hbt_time_data = np.asarray(hbt_time_data, dtype=float)
hbt_time_data = np.reshape(hbt_time_data, (37, hbt_ma_data[0][0].shape[0]))
original_length = hbt_time_data.shape[1]
frame_ratio = original_length // TARGET_FRAME_COUNT
hbt_time_data = hbt_time_data[:,::frame_ratio]
hbt_time_data = hbt_time_data[:,:TARGET_FRAME_COUNT]

print("HBT data shapes:")
print(f"Mode amplitude 1: {hbt_ma1_data.shape}")
print(f"Mode phase 1: {hbt_mp1_data.shape}")
print(f"Time data: {hbt_time_data.shape}")


# Process original shot list
training_data_2D, cut_training_data_2D, flat_training_data = process_all_shots(shot_list, file_path)

print(f"training_data_2D shape: {training_data_2D.shape}")
print(f"cut_training_data_2D shape: {cut_training_data_2D.shape}")


# In[20]:


# Prepare data for HBT prediction model
target_data = hbt_ma2_data  # Using mode 2 amplitude as target
training_data = cut_training_data_2D

# Normalization factors
camera_norm = 1  # Camera data already normalized by max_pixel_value
ma_norm = 1  # Mode amplitude normalization factor

# Reshape the training data and labels
target_vector = []
training_vector = []
tot_frames = TARGET_FRAME_COUNT

for i in range(len(shot_list)):
    for j in range(tot_frames):
        target_vector.append(target_data[i][j])
        training_vector.append(training_data[i][j])

# Shuffle the data
random.seed(123)
zip_list = list(zip(target_vector, training_vector))
random.shuffle(zip_list)
target_vector, training_vector = zip(*zip_list)

# Convert to numpy arrays and normalize
target_vector = np.asarray(target_vector, dtype=np.float32) / ma_norm
training_vector = np.asarray(training_vector, dtype=np.float32)

# Split into training and testing sets
testing_inputs = training_vector[-201:-1]
testing_labels = target_vector[-201:-1]
training_vector = training_vector[0:-200]
target_vector = target_vector[0:-200]

print('Training shape: ', training_vector.shape, 'Target shape: ', target_vector.shape)
print('Testing shape: ', testing_inputs.shape, 'Testing label shape: ', testing_labels.shape)
plt.plot(testing_labels)


# In[22]:


# Define model architecture
num_conv2d_layers = 2
num_dense_layers = 1

conv2d_neurons = [16, 8]
conv2d_size = [(8, 8), (8, 8)]
dense_layer_neurons = [24]
max_pooling_size = (4, 4)
activation_func = 'relu'
loss_func = 'mean_squared_error'
optimizer_func = 'adam'

# Create the model
william_model = tf.keras.models.Sequential()

# Add input layer
william_model.add(tf.keras.layers.InputLayer(shape=(32, 32, 1)))

# Add Conv2D layers
for i in range(num_conv2d_layers):
    william_model.add(tf.keras.layers.Conv2D(conv2d_neurons[i], conv2d_size[i], 1, activation=activation_func))
    william_model.add(tf.keras.layers.MaxPooling2D(max_pooling_size, 1))

# Flatten the output
william_model.add(tf.keras.layers.Flatten())

# Add dense layers
for i in range(num_dense_layers):
    william_model.add(tf.keras.layers.Dense(dense_layer_neurons[i], activation=activation_func))

# Output layer
william_model.add(tf.keras.layers.Dense(1))

# Compile the model
william_model.compile(optimizer=optimizer_func, loss=loss_func)

# Display model summary
william_model.summary()


# In[24]:


# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
Model = william_model
history = Model.fit(training_vector, target_vector,
                       epochs=15,
                       validation_split=0.2,
                       verbose=1,
                       callbacks=[early_stop])


# In[26]:


# Evaluate the model on test data
predictions = Model.predict(testing_inputs)

# Calculate prediction errors
prediction_errors = abs((testing_labels[:,0]/np.max(testing_labels[:,0]) - 
                        predictions[:,0]/np.max(predictions[:,0]))) * 100

# Create visualization plots
fig, axes = plt.subplots(figsize=(15,5), ncols=3, nrows=1)

# Plot 1: Training History
axes[0].plot(range(1,len(history.history['val_loss'])+1), 
             history.history['val_loss'], 
             label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('HBT Model Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Error Distribution
weights = np.ones_like(prediction_errors) / len(prediction_errors) * 100
axes[1].hist(prediction_errors, 20, weights=weights)
axes[1].set_xlabel('% error')
axes[1].set_ylabel('% Count')
axes[1].set_title('Normalized Testing Error (n=200)')

# Plot 3: Error by Sample
axes[2].plot(prediction_errors, '.')
axes[2].set_xlabel('Test Sample Number')
axes[2].set_ylabel('% error')
axes[2].set_title('Normalized Testing Error')

plt.tight_layout()
plt.show()

# Plot 4: actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(testing_labels[:,0]/np.max(testing_labels[:,0]), '.', label='Actual Mode Amplitude')
plt.plot(predictions[:,0]/np.max(predictions[:,0]), '.', label='Predicted Mode Amplitude')
plt.plot(-(testing_labels[:,0]/np.max(testing_labels[:,0]) - 
           predictions[:,0]/np.max(predictions[:,0])), '*', label='Difference')
plt.xlabel('Sample Number')
plt.ylabel('Normalized Mode Amplitude')
plt.title('HBT Mode Amplitude Prediction Results')
plt.legend()
plt.grid(True)
plt.show()

print(f"Maximum actual mode amplitude: {np.max(testing_labels[:,0])}")
print(f"Maximum predicted mode amplitude: {np.max(predictions[:,0])}")
print(f"Mean absolute percentage error: {np.mean(prediction_errors):.2f}%")


# In[ ]:


# YOU MUST change name before saving!!!
#Model.save_weights('Potential Models/model_agb001.weights.h5')


# In[ ]:




