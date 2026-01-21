# Machine Learning Practice Notebook

A beginner-friendly Jupyter notebook demonstrating fundamental machine learning concepts using Python.

## 📋 Overview

This notebook provides hands-on examples of:
- **Data Visualization** with Matplotlib
- **Linear Regression** for predictive modeling
- **K-Nearest Neighbors (KNN)** classification

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries:**
  - `numpy` - Numerical computing
  - `pandas` - Data manipulation
  - `matplotlib` - Data visualization
  - `scikit-learn` - Machine learning algorithms
  - `tensorflow` - Deep learning framework

## 📊 Projects Included

### 1. Data Visualization
- Bar chart showing average marks across subjects (Maths, DBMS, OS, ML, Python)
- Demonstrates basic plotting with Matplotlib

### 2. Linear Regression: Hours Studied vs. Marks
- **Objective:** Predict exam marks based on study hours
- **Dataset:** 8 data points (1-8 hours studied)
- **Model:** Simple linear regression
- **Features:**
  - Scatter plot of raw data
  - Fitted regression line visualization
  - Interactive prediction (change study hours to see predicted marks)

**Key Formula:** `marks ≈ m * hours + b`
- Slope (m): 7.08
- Intercept (b): 27.50

### 3. Classification: Iris Flower Dataset
- **Objective:** Classify iris flower species
- **Dataset:** Iris dataset (built-in scikit-learn)
- **Features Used:** Sepal length and sepal width (2D for easy visualization)
- **Model:** K-Nearest Neighbors (KNN) with k=5
- **Classes:** Setosa, Versicolor, Virginica
- **Test Accuracy:** 80%

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### Running the Notebook
1. Open in Google Colab or Jupyter Notebook
2. Run cells sequentially (Shift + Enter)
3. Experiment with different values in prediction cells

## 📈 Key Concepts Demonstrated

### Supervised Learning
- **Regression:** Predicting continuous values (marks)
- **Classification:** Predicting categories (flower species)

### Model Evaluation
- Visual inspection of fitted models
- Accuracy metrics for classification

### Real-World Applications
- **Regression:** Price prediction, sales forecasting, demand estimation
- **Classification:** Customer segmentation, image recognition, disease diagnosis

## 🎯 Learning Outcomes

After working through this notebook, you'll understand:
- How to load and visualize data
- The difference between regression and classification
- How to train and evaluate simple ML models
- How to interpret model predictions

## 📝 Notes

- The notebook includes intentional errors (e.g., concatenating strings with integers) to demonstrate debugging
- Interactive prediction cells allow hands-on experimentation
- Comments explain each step for educational purposes

## 🔮 Future Enhancements

- Add more features to improve Iris classification accuracy
- Implement cross-validation
- Compare multiple classification algorithms
- Add feature engineering examples

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Machine Learning Basics](https://developers.google.com/machine-learning/crash-course)

## ⚠️ Important Notes

- **Browser Storage:** This notebook does NOT use `localStorage` or `sessionStorage` as these APIs are not supported in the notebook environment
- All data is stored in memory during the session
- Results are reset when the kernel is restarted

## 👨‍💻 Author

Created as an educational resource for machine learning beginners.

---

**Happy Learning! 🎓**
