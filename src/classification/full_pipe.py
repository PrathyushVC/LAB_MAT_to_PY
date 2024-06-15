
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score

# Load medical images and masks
def load_images(image_paths, mask_paths):
    """
    Load medical images and their corresponding masks.
    
    Args:
    image_paths (list of str): List of file paths to the medical images.
    mask_paths (list of str): List of file paths to the masks.
    
    Returns:
    images (list of SimpleITK.Image): Loaded medical images.
    masks (list of SimpleITK.Image): Loaded masks.
    """
    supported_formats = ['.nii', '.nii.gz', '.mha']
    
    def read_image(file_path):
        _, ext = os.path.splitext(file_path)
        if ext.lower() in supported_formats:
            return sitk.ReadImage(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    images = [read_image(p) for p in image_paths]
    masks = [read_image(p) for p in mask_paths]
    
    return images, masks

# Adjust resolution
def adjust_resolution(image, target_resolution=(1.0, 1.0, 1.0)):
    """
    Adjust the resolution of a medical image.
    
    Args:
    image (SimpleITK.Image): Input medical image.
    target_resolution (tuple of float): Target resolution (spacing) for the image.
    
    Returns:
    SimpleITK.Image: Image resampled to the target resolution.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_resolution)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    
    new_size = np.array(image.GetSize()) * np.array(image.GetSpacing()) / np.array(target_resolution)
    new_size = new_size.astype(int).tolist()
    resampler.SetSize(new_size)
    
    return resampler.Execute(image)

# Bias field correction
def bias_field_correction(image):
    """
    Apply bias field correction to a medical image.
    
    Args:
    image (SimpleITK.Image): Input medical image.
    
    Returns:
    SimpleITK.Image: Bias field corrected image.
    """
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(image)
    return corrected_image

# Preprocess images
def preprocess_images(images, target_resolution=None, apply_bias_field_correction=False):
    """
    Preprocess a list of medical images by adjusting resolution and applying bias field correction.
    
    Args:
    images (list of SimpleITK.Image): List of input medical images.
    target_resolution (tuple of float, optional): Target resolution for resampling. Default is None.
    apply_bias_field_correction (bool, optional): Whether to apply bias field correction. Default is False.
    
    Returns:
    list of numpy.ndarray: Preprocessed images.
    """
    processed_images = []
    for img in images:
        if target_resolution is not None:
            img = adjust_resolution(img, target_resolution)
        if apply_bias_field_correction:
            img = bias_field_correction(img)
        processed_images.append(sitk.GetArrayFromImage(img))
    return processed_images

# Extract radiomic features
def extract_radiomic_features(images, masks):
    """
    Extract radiomic features from medical images and their corresponding masks.
    Note this currently only handles the features extracted by pyradiomics
    
    Args:
    images (list of SimpleITK.Image): List of medical images.
    masks (list of SimpleITK.Image): List of masks corresponding to the images.
    
    Returns:
    pandas.DataFrame: DataFrame containing the extracted radiomic features.
    """

    extractor = featureextractor.RadiomicsFeatureExtractor()
    print("Extracting the following radiomic features:")
    print(extractor.getFeatureNames())
    features = []
    for img, mask in zip(images, masks):
        result = extractor.execute(img, mask)
        features.append(result)
    return pd.DataFrame(features)

# Evaluate the model
def evaluate_model(X, y, model_name='RandomForest', n_features_to_select=10):
    """
    Evaluate a machine learning model with feature selection and hyperparameter tuning.
    Note this code is single shot and does not handle K fold or patient_ids being related to multiple samples. 
    
    Args:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Target vector.
    model_name (str, optional): Name of the model to use ('RandomForest', 'XGBoost', 'CatBoost', 'SVM'). Default is 'RandomForest'.
    n_features_to_select (int, optional): Number of features to select. Default is 10.
    
    Returns:
    float: AUC score.
    float: F1 score.
    float: Sensitivity.
    float: Specificity.
    float: Accuracy.
    dict: Best hyperparameters found during tuning.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Feature selection
    
    base_model = None
    param_grid = {}
    if model_name == 'RandomForest':
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30]
        }
    elif model_name == 'XGBoost':
        base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif model_name == 'CatBoost':
        base_model = CatBoostClassifier(random_state=42, silent=True)
        param_grid = {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif model_name == 'SVM':
        base_model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Hyperparameter tuning
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='roc_auc')
    selector = RFE(grid_search, n_features_to_select=n_features_to_select, step=1)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Compute evaluation metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return auc, f1, sensitivity, specificity, accuracy, grid_search.best_params_

# Example usage
if __name__ == "__main__":
    image_paths = ['path/to/image1.nii', 'path/to/image2.nii']  # Replace with actual paths
    mask_paths = ['path/to/mask1.nii', 'path/to/mask2.nii']    # Replace with actual paths
    target_resolution = (1.0, 1.0, 1.0)  # Adjust as needed
    
    images, masks = load_images(image_paths, mask_paths)
    images = preprocess_images(images, target_resolution=target_resolution, apply_bias_field_correction=True)
    masks = [sitk.GetArrayFromImage(mask) for mask in masks]
    
    features_df = extract_radiomic_features(images, masks)
    
    # Assuming 'label' column contains the target labels
    X = features_df.drop(columns=['label']).values
    y = features_df['label'].values
    
    model_name = 'RandomForest'  # Change to 'XGBoost', 'CatBoost', 'SVM' as needed
    auc, f1, sensitivity, specificity, accuracy, best_params = evaluate_model(X, y, model_name=model_name)
    
    print(f"Model: {model_name}")
    print(f"Best Parameters: {best_params}")
    print(f"AUC: {auc}")
    print(f"F1 Score: {f1}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Accuracy: {accuracy}")

