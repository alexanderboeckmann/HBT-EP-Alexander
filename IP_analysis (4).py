#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Discription
# This is designed to predict plasma current based off of highspeed image data


# In[2]:


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


# In[3]:


# Define shot list
shot_list = [119591, 119599, 119601, 119646, 119648, 119653, 119654, 119658, 119659,
             119661, 119662, 119663, 119665, 119666, 119667, 119669, 119670, 119671,
             119673, 119675, 119748, 119750, 119751, 119752, 119754, 119755, 119756,
             119757, 119760, 119761, 119762, 119763, 119764, 119766, 119767, 119768,
             119769]

# Load IP data from .npy files
file_path_ip = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HighFreqMLModeling/ip_Data/'
file_path_tiff = '/Users/aboeckmann/Documents/Columbia/PlasmaLab/HighFreqMLModeling/Training/Input Data/Shots/'

TARGET_FRAME_COUNT = 800
CAMERA_DEPTH = 65535.0 # 2^16


# In[4]:


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
                # Normalize to [0,1] range
                im = (im - im.min()) / (im.max() - im.min())
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
def load_ip_data(shot_list, file_path):
    ip_data = []
    for shot in shot_list:
        try:
            shot_data = np.load(os.path.join(file_path, f'{shot}ip.npy'))
            ip_data.append(shot_data)
        except FileNotFoundError:
            print(f"Warning: Could not find IP data for shot {shot}")
    return np.array(ip_data)


# In[5]:


# Function to format IP data
def format_ip_data(data, target_length=TARGET_FRAME_COUNT):
    """Format plasma current data to match target length through downsampling"""
    # Convert to array and get frame ratio
    data = np.asarray(data, dtype=float)
    frame_ratio = data[0].shape[0] // target_length
    
    # Reshape and downsample in one go
    data = np.reshape(data, (len(data), -1, 1))
    data = data[:,::frame_ratio,:]
    data = data[:,:target_length,:]
    return data

# Load IP data
ip_data = load_ip_data(shot_list, file_path_ip)
formatted_ip_data = format_ip_data(ip_data)

print("IP data shape:", formatted_ip_data.shape)

# Plot all shots overlaid
plt.figure(figsize=(15, 6))
for i, shot_num in enumerate(shot_list):
    plt.plot(formatted_ip_data[i, :, 0], alpha=0.5, label=f'Shot {shot_num}')
plt.title('IP Data - All Shots Overlay')
plt.xlabel('Time Index')
plt.ylabel('IP Value')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()

# Create plot showing mean and standard deviation
plt.figure(figsize=(15, 6))
mean_ip = np.mean(formatted_ip_data[:, :, 0], axis=0)
std_ip = np.std(formatted_ip_data[:, :, 0], axis=0)

plt.plot(mean_ip, 'b-', label='Mean IP', linewidth=2)
plt.fill_between(range(len(mean_ip)), 
                mean_ip - std_ip, 
                mean_ip + std_ip, 
                color='b', alpha=0.2, 
                label='±1 Std Dev')
plt.title('IP Data - Mean and Standard Deviation Across All Shots')
plt.xlabel('Time Index')
plt.ylabel('IP Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[21]:


training_data_2D, cut_training_data_2D, flat_training_data = process_all_shots(shot_list, file_path_tiff)

# Look at cut_training_data_2D to check size
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(cut_training_data_2D[0, i, :, :], cmap='gray')  # Removed the extra channel index
    plt.title(f'Frame {i}')
    plt.axis('off')
plt.suptitle(f'Sample Images from Shot {shot_list[0]}')
plt.tight_layout()
plt.show()


# Prepare data for IP prediction model
ip_target_data = formatted_ip_data
ip_training_data = cut_training_data_2D

# Calculate normalization factor for IP data
ip_norm_factor = np.max(np.abs(formatted_ip_data))
print(f"IP normalization factor: {ip_norm_factor}")

# Reshape and normalize the data
ip_target_vector = []
ip_training_vector = []
tot_frames = TARGET_FRAME_COUNT

for i in range(len(shot_list)):
    for j in range(tot_frames):
        # Normalize IP data
        ip_target_vector.append(ip_target_data[i][j] / ip_norm_factor)
        ip_training_vector.append(ip_training_data[i][j])

# Shuffle the data
random.seed(123)
zip_list = list(zip(ip_target_vector, ip_training_vector))
random.shuffle(zip_list)
ip_target_vector, ip_training_vector = zip(*zip_list)

# Convert to numpy arrays with explicit dtypes and shapes
ip_target_vector = np.asarray(ip_target_vector, dtype=np.float32)
ip_training_vector = np.asarray(ip_training_vector, dtype=np.float32)

# Ensure consistent shapes
ip_training_vector = ip_training_vector.reshape(-1, 32, 32, 1)
ip_target_vector = ip_target_vector.reshape(-1, 1)

# Split into training and testing sets
ip_testing_inputs = ip_training_vector[-201:-1]
ip_testing_labels = ip_target_vector[-201:-1]
ip_training_vector = ip_training_vector[0:-200]
ip_target_vector = ip_target_vector[0:-200]

# Convert to TensorFlow tensors with specified shapes
ip_training_vector = tf.convert_to_tensor(ip_training_vector, dtype=tf.float32)
ip_target_vector = tf.convert_to_tensor(ip_target_vector, dtype=tf.float32)
ip_testing_inputs = tf.convert_to_tensor(ip_testing_inputs, dtype=tf.float32)
ip_testing_labels = tf.convert_to_tensor(ip_testing_labels, dtype=tf.float32)

print('Training shape: ', ip_training_vector.shape, 'Target shape: ', ip_target_vector.shape)
print('Testing shape: ', ip_testing_inputs.shape, 'Testing label shape: ', ip_testing_labels.shape)


# In[7]:


import tensorflow as tf

# Define model architecture parameters
num_conv2d_layers = 2
num_dense_layers = 1
conv2d_neurons = [16, 8]  # Number of filters in each Conv2D layer
conv2d_size = [(8, 8), (8, 8)]  # Kernel sizes for Conv2D layers
dense_layer_neurons = [24]  # Neurons in the dense layer
max_pooling_size = (4, 4)  # Pooling size for MaxPooling2D
activation_func = 'relu'  # Activation function for hidden layers
loss_func = 'mean_squared_error'  # Loss function for regression
optimizer_func = 'adam'  # Optimizer

# Create the model
ip_model = tf.keras.models.Sequential()

# Add input layer
ip_model.add(tf.keras.layers.InputLayer(shape=(32, 32, 1)))

# Add Conv2D and MaxPooling layers
for i in range(num_conv2d_layers):
    ip_model.add(tf.keras.layers.Conv2D(
        filters=conv2d_neurons[i],
        kernel_size=conv2d_size[i],
        strides=1,
        activation=activation_func,
        padding='same'  # Added to preserve spatial dimensions
    ))
    ip_model.add(tf.keras.layers.MaxPooling2D(
        pool_size=max_pooling_size,
        strides=1,
        padding='same'  # Added to preserve spatial dimensions
    ))

# Flatten the output
ip_model.add(tf.keras.layers.Flatten())

# Add dense layers
for i in range(num_dense_layers):
    ip_model.add(tf.keras.layers.Dense(
        units=dense_layer_neurons[i],
        activation=activation_func
    ))

# Output layer (single value for normalized IP prediction)
ip_model.add(tf.keras.layers.Dense(1))

# Compile the model
ip_model.compile(optimizer=optimizer_func, loss=loss_func, metrics=['mae'])

# Display model summary
ip_model.summary()


# In[8]:


# Training settings
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)

# Train the model with tf.data.Dataset
Model = ip_model
ip_history = Model.fit(ip_training_vector, ip_target_vector,
                    epochs=15,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[early_stop])


# In[9]:


# Evaluate the model on test data
ip_predictions = Model.predict(ip_testing_inputs)

# Calculate prediction errors
ip_prediction_errors = abs((ip_testing_labels[:,0] - ip_predictions[:,0])) / tf.reduce_max(ip_testing_labels[:,0]) * 100

# Create visualization plots
fig, axes = plt.subplots(figsize=(15,5), ncols=3, nrows=1)

# Plot 1: Training History
axes[0].plot(range(1,len(ip_history.history['val_loss'])+1), 
             ip_history.history['val_loss'], 
             label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('IP Model Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Error Distribution
weights = np.ones_like(ip_prediction_errors) / len(ip_prediction_errors) * 100
axes[1].hist(ip_prediction_errors, 20, weights=weights)
axes[1].set_xlabel('% error')
axes[1].set_ylabel('% Count')
axes[1].set_title('Normalized IP Testing Error (n=200)')

# Plot 3: Error by Sample
axes[2].plot(ip_prediction_errors, '.')
axes[2].set_xlabel('Test Sample Number')
axes[2].set_ylabel('% error')
axes[2].set_title('Normalized IP Testing Error')

plt.tight_layout()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(ip_testing_labels[:,0]/tf.reduce_max(ip_testing_labels[:,0]), '.', label='Actual IP')
plt.plot(ip_predictions[:,0]/tf.reduce_max(ip_predictions[:,0]), '.', label='Predicted IP')
plt.plot(-(ip_testing_labels[:,0]/tf.reduce_max(ip_testing_labels[:,0]) - 
           ip_predictions[:,0]/tf.reduce_max(ip_predictions[:,0])), '*', label='Difference')
plt.xlabel('Sample Number')
plt.ylabel('Normalized IP')
plt.title('IP Prediction Results')
plt.legend()
plt.grid(True)
plt.show()

print(f"Maximum actual IP: {tf.reduce_max(ip_testing_labels[:,0]) * ip_norm_factor}")
print(f"Maximum predicted IP: {tf.reduce_max(ip_predictions[:,0]) * ip_norm_factor}")
print(f"Mean absolute percentage error: {tf.reduce_mean(ip_prediction_errors):.2f}%")


# In[ ]:




