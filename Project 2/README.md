# Breast Cancer Detection using Logistic Regression

A machine learning project that implements a binary classification model to detect breast cancer (malignant vs benign) using the Wisconsin Breast Cancer dataset.

## Project Overview

This project demonstrates the application of Logistic Regression for medical diagnosis, specifically classifying breast tumors as malignant or benign based on various cell nucleus features extracted from digitized images of fine needle aspirate (FNA) of breast mass.

## Dataset

- **Source**: Wisconsin Breast Cancer dataset (available via scikit-learn)
- **Samples**: 569 instances
- **Features**: 30 numeric features describing characteristics of cell nuclei
- **Classes**: 
  - Malignant (0)
  - Benign (1)

### Feature Categories

The features include measurements such as:
- Mean radius, texture, perimeter, area
- Mean smoothness, compactness, concavity
- Mean concave points, symmetry, fractal dimension
- Similar "worst" (largest) measurements for each category

## Technologies Used

- **Python 3.x**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **scikit-learn**: Machine learning algorithms and utilities

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Project Structure

```
breast-cancer-detection/
│
├── notebook.ipynb          # Main Jupyter notebook
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Implementation Steps

### 1. Data Loading
Load the breast cancer dataset from scikit-learn's built-in datasets.

### 2. Data Exploration
- Examine dataset shape and structure
- Inspect feature names and target classes
- Create a pandas DataFrame for better visualization

### 3. Data Preprocessing
- **Train-Test Split**: 80% training, 20% testing with stratification
- **Feature Scaling**: StandardScaler normalization (mean ≈ 0, std ≈ 1)

### 4. Model Training
Train a Logistic Regression classifier with the following parameters:
- `max_iter=1000`: Maximum iterations for convergence

### 5. Model Evaluation
Evaluate model performance using:
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Precision, recall, and F1-score
- **Confusion Matrix**: Visual representation of predictions

### 6. Visualization
Generate a confusion matrix heatmap to visualize model performance.

## Results

The model achieves:
- **Accuracy**: ~98.25%
- **Precision**: 
  - Malignant: 0.98
  - Benign: 0.99
- **Recall**: 
  - Malignant: 0.98
  - Benign: 0.99

### Confusion Matrix
```
              Predicted
              Malignant  Benign
Actual
Malignant        41        1
Benign            1       71
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook notebook.ipynb
```

2. Run all cells sequentially to:
   - Load and explore the data
   - Train the model
   - Evaluate performance
   - Visualize results

## Key Insights

1. **High Accuracy**: The model achieves excellent performance with minimal misclassifications
2. **Balanced Performance**: Good precision and recall for both classes
3. **Feature Scaling**: Critical for Logistic Regression performance
4. **Minimal False Negatives**: Only 1 malignant case misclassified (critical for medical diagnosis)

## How Logistic Regression Works

Logistic Regression is a linear model that:
1. Computes a weighted sum of input features
2. Applies the sigmoid function to produce a probability (0 to 1)
3. Classifies based on a threshold (typically 0.5)
4. Uses gradient descent to optimize weights during training

## Limitations & Future Work

### Current Limitations
- Simple linear model may not capture complex non-linear relationships
- No hyperparameter tuning performed
- Single model evaluation (no cross-validation)

### Potential Improvements
- Implement cross-validation for robust performance estimation
- Try ensemble methods (Random Forest, Gradient Boosting)
- Perform feature importance analysis
- Explore deep learning approaches
- Add model explainability (SHAP, LIME)
- Implement hyperparameter optimization

## Medical Disclaimer

⚠️ **Important**: This is an educational project for demonstration purposes only. It should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## License

This project is open source and available for educational purposes.

## References

- [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## Author

Kazi Shaheed Rahman
