## Cat vs Dog Image Classification using HOG + SVM

## Project Overview
This project builds a binary image classifier for cats and dogs using a traditional computer vision pipeline.

Instead of training directly on raw pixels, the notebook extracts Histogram of Oriented Gradients (HOG) features from each image and trains a Support Vector Machine (SVM) with an RBF kernel. This improves robustness to background noise and reduces dimensionality.

Notebook file: `SCT_ML_3.ipynb`

## Dataset Structure
The project expects the dataset in the following structure:

```text
data/
  cat/
  dog/
```

Each folder should contain images for the respective class.

## Pipeline
1. Load images from `data/cat` and `data/dog`.
2. Convert images to grayscale.
3. Resize all images to `64x64`.
4. Extract HOG features with:
   - `orientations=9`
   - `pixels_per_cell=(8, 8)`
   - `cells_per_block=(2, 2)`
5. Split data into train/test using an 80/20 stratified split.
6. Train `sklearn.svm.SVC` with:
   - `kernel='rbf'`
   - `C=5.0`
   - `gamma='scale'`
7. Evaluate using accuracy and a classification report.

## Sample Performance
Observed test accuracy from the notebook run is approximately `86%`.

Example class-wise metrics:

```text
              precision    recall  f1-score   support

cat              0.88      0.84      0.86       100
dog              0.85      0.89      0.87       100

accuracy                               0.86       200
macro avg        0.87      0.86      0.86       200
weighted avg     0.87      0.86      0.86       200
```

## Requirements
Install dependencies with:

```bash
pip install numpy opencv-python scikit-image scikit-learn
```

## How to Run
### Option 1: Local Jupyter
1. Ensure the `data/` folder contains `cat/` and `dog/` subfolders.
2. Open `SCT_ML_3.ipynb`.
3. Run all cells in order.

### Option 2: Google Colab
1. Upload `data.zip`.
2. Unzip using the first notebook cell.
3. Run remaining cells sequentially.

## Notes
- The current approach is lightweight and interpretable compared to deep CNN models.
- You can improve performance further with data augmentation, hyperparameter tuning, or a CNN baseline for comparison.
