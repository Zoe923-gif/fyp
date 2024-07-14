import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = "C:/Users/zoezh/Downloads/fabric_defect_dataset"
train_dir = "C:/Users/zoezh/Downloads/dataset_split__/train"
val_dir = "C:/Users/zoezh/Downloads/dataset_split__/val"
test_dir = "C:/Users/zoezh/Downloads/dataset_split__/test"

# Create directories for train, validation, and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create defect and non_defect subdirectories in train, val, and test directories
for subdir in ['defect', 'non_defect']:
    os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(val_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)

# Function to split dataset into train, val, and test sets
def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.2, 0.1)):
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return
    
    filenames = os.listdir(source_dir)
    if not filenames:
        print(f"No files found in {source_dir}.")
        return

    train_files, val_test_files = train_test_split(filenames, test_size=split_ratio[1] + split_ratio[2])
    val_files, test_files = train_test_split(val_test_files, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]))

    print(f"Total files: {len(filenames)}")
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    for filename in train_files:
        src_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(train_dir, os.path.basename(source_dir), filename)
        if not os.path.exists(src_file):
            print(f"Source file {src_file} does not exist.")
            continue
        shutil.copy(src_file, dest_file)
    
    for filename in val_files:
        src_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(val_dir, os.path.basename(source_dir), filename)
        if not os.path.exists(src_file):
            print(f"Source file {src_file} does not exist.")
            continue
        shutil.copy(src_file, dest_file)
    
    for filename in test_files:
        src_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(test_dir, os.path.basename(source_dir), filename)
        if not os.path.exists(src_file):
            print(f"Source file {src_file} does not exist.")
            continue
        shutil.copy(src_file, dest_file)

# Split defect data
split_data(os.path.join(dataset_dir, 'defect'), train_dir, val_dir, test_dir)

# Split non_defect data
split_data(os.path.join(dataset_dir, 'non_defect'), train_dir, val_dir, test_dir)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image Data Generators with updated target size
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Updated to match the model's input shape
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),  # Updated to match the model's input shape
    batch_size=32,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),  # Updated to match the model's input shape
    batch_size=32,
    class_mode='binary'
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def build_mscnn(input_shape):
    inputs = Input(shape=input_shape)

    # Scale 1
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = MaxPooling2D((2, 2))(x1)
    
    # Scale 2
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x2 = MaxPooling2D((2, 2))(x2)
    
    # Scale 3
    x3 = Conv2D(128, (7, 7), activation='relu', padding='same')(inputs)
    x3 = MaxPooling2D((2, 2))(x3)
    
    # Concatenate
    concatenated = concatenate([x1, x2, x3], axis=-1)
    
    # Further Convolution and Pooling
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(concatenated)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    
    # Fully Connected Layers
    x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, x)
    
    return model

input_shape = (128, 128, 3)  # Ensure this matches the data shape
model = build_mscnn(input_shape)

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=20, validation_data=val_generator)

# Save the model
model.save('C:/Users/zoezh/Downloads/mscnn_fabric_defect_detector.keras')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')
