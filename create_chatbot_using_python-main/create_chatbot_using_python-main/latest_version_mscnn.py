import os
import shutil
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, concatenate, GlobalAveragePooling2D

# Define paths
dataset_dir = "C:/Users/zoezh/Downloads/fabric_defect_dataset"
train_dir = "C:/Users/zoezh/Downloads/dataset_split__/train"
val_dir = "C:/Users/zoezh/Downloads/dataset_split__/val"
test_dir = "C:/Users/zoezh/Downloads/dataset_split__/test"

# Function to create directories
def create_directories():
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
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
        shutil.copy(os.path.join(source_dir, filename), os.path.join(train_dir, os.path.basename(source_dir), filename))
    
    for filename in val_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(val_dir, os.path.basename(source_dir), filename))
    
    for filename in test_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(test_dir, os.path.basename(source_dir), filename))

# Function to build the Multi-Scale CNN model
def build_mscnn(input_shape):
    inputs = Input(shape=input_shape)

    # Scale 1
    x1 = Conv2D(32, (3, 3), activation='tanh', padding='same')(inputs)
    x1 = MaxPooling2D((2, 2))(x1)
    
    # Scale 2
    x2 = Conv2D(64, (5, 5), activation='tanh', padding='same')(inputs)
    x2 = MaxPooling2D((2, 2))(x2)
    
    # Scale 3
    x3 = Conv2D(128, (7, 7), activation='tanh', padding='same')(inputs)
    x3 = MaxPooling2D((2, 2))(x3)
    
    # Concatenate
    concatenated = concatenate([x1, x2, x3], axis=-1)

    # Dropout Layer
    # dropout = Dropout(0.5)(concatenated)

    # Apply Global Average Pooling
    globalAvgPool = GlobalAveragePooling2D()(concatenated)

    # convo_layer = Conv2D(256, (3, 3), activation='tanh', padding='same')(concatenated)
    # pool_layer = MaxPooling2D((2, 2))(convo_layer)
    # flatten_layer = Flatten()(pool_layer)
    
    # Flatten and Fully Connected Layers
    # flatten_layer = Flatten()(globalAvgPool)
    dense512 = Dense(512, activation='tanh')(globalAvgPool)
    # x = Dropout(0.5)(dense512)
    output_layer = Dense(1, activation='sigmoid')(dense512)
    
    model = Model(inputs, output_layer)
    return model

# Function to plot training and validation metrics
def plot_metrics(history):
    # Accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Main process
if __name__ == "__main__":
    # Create directories
    create_directories()

    # Split dataset
    split_data(os.path.join(dataset_dir, 'defect'), train_dir, val_dir, test_dir)
    split_data(os.path.join(dataset_dir, 'non_defect'), train_dir, val_dir, test_dir)

    # Image Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=20, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True,
        vertical_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128), 
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128), 
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary'
    )

    # Build, compile, and train the model
    input_shape = (128, 128, 1)  
    model = build_mscnn(input_shape)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator, epochs=20, validation_data=val_generator)

    # Plot training and validation metrics
    plot_metrics(history)

    # Save the model
    model.save('C:/Users/zoezh/Downloads/mscnn_fabric_defect_detector.keras')

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_acc}')
    print(f'Test Loss: {test_loss}')

    # Save a summary of the results
    summary_df = pd.DataFrame({
        'Metric': ['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss', 'Test Accuracy', 'Test Loss'],
        'Value': [
            history.history['accuracy'][-1],
            history.history['val_accuracy'][-1],
            history.history['loss'][-1],
            history.history['val_loss'][-1],
            test_acc,
            test_loss
        ]
    })
    summary_df.to_csv('C:/Users/zoezh/Downloads/training_testing_summary.csv', index=False)
    print(summary_df)
