import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
import pandas as pd

# -------------------- Data Loading --------------------
def load_nifti_data(data_path, target_shape=(64, 64, 64)):
    images, labels = [], []
    class_labels = sorted([
    d for d in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, d))
    ])  # Sorted for consistent label assignment

    for idx, class_name in enumerate(class_labels):
        class_path = os.path.join(data_path, class_name)
        nifti_files = [f for f in os.listdir(class_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
        
        for file in nifti_files:
            file_path = os.path.join(class_path, file)
            nifti_img = nib.load(file_path).get_fdata()
            nifti_img = nifti_img / np.max(nifti_img)  # Normalize
            cropped_img = center_crop(nifti_img, target_shape)
            images.append(cropped_img)
            labels.append(idx)

    images = np.array(images)
    labels = to_categorical(labels)
    return images, labels

def center_crop(img, target_shape):
    start = [(img.shape[i] - target_shape[i]) // 2 for i in range(3)]
    end = [start[i] + target_shape[i] for i in range(3)]
    return img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

# -------------------- Model --------------------
def build_3d_cnn(input_shape, num_classes):
    model = Sequential([
        Conv3D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling3D(pool_size=2),
        BatchNormalization(),

        Conv3D(64, kernel_size=3, activation='relu'),
        MaxPooling3D(pool_size=2),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------- Pipeline --------------------
if __name__ == "__main__":
    # Step 1: Load and preprocess data
    data_dir = "data"  # Replace with your dataset directory
    images, labels = load_nifti_data(data_dir)
    images = images[..., np.newaxis]  # Add channel dim

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Step 3: Build and train model
    model = build_3d_cnn(X_train.shape[1:], y_train.shape[1])
    model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=4)

    # Step 4: Predict new labels on test data
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Step 5: Save predictions
    df = pd.DataFrame({'True': true_classes, 'Predicted': predicted_classes})
    df.to_csv("predicted_test_labels.csv", index=False)

    print("Training complete. Predictions saved to 'predicted_test_labels.csv'.")

# -------------------- Step 6: Create Dummy NIfTI File with 25 Labels --------------------
# Load the original file to match its shape/affine
original_nifti = nib.load("./data/3_month_patient1/3Month25Labels_Stealth_controlled.nii")
original_shape = original_nifti.get_fdata().shape
affine = original_nifti.affine

# Create a dummy label map: fill with zeros, then add artificial regions
label_volume = np.zeros(original_shape, dtype=np.uint8)

# Example: Create 25 regions by marking isolated points or cubes
for label in range(1, 26):
    # Create a cube centered at a random location
    z, y, x = np.random.randint(10, original_shape[0]-10), np.random.randint(10, original_shape[1]-10), np.random.randint(10, original_shape[2]-10)
    label_volume[z-2:z+2, y-2:y+2, x-2:x+2] = label

# Save as a new NIfTI file
label_nifti = nib.Nifti1Image(label_volume, affine)
nib.save(label_nifti, "generated_25_labels.nii.gz")
print("✅ New label NIfTI file saved: generated_25_labels.nii.gz")
