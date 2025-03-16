### **Sentiment Analysis with SVM and NTK**

This repository contains a Sentiment Analysis model that leverages **SVM** with the **Neural Tangent Kernel (NTK)** to improve performance on high-dimensional data. The NTK approximates neural network behavior, enhancing SVM's classification accuracy.

---

## **Dependencies**
Ensure you have the following libraries installed:  
- `jax`  
- `flax`  
- `optax`  
- `scikit-learn`  
- `matplotlib`  
- `seaborn`  
- `pandas`  
- `numpy`  

You can install the dependencies using:

```bash
pip install jax flax optax scikit-learn matplotlib seaborn pandas numpy
```

---

## **Dataset**
- The dataset used for sentiment analysis should contain labeled text data for binary classification (e.g., positive or negative sentiments).
- Preprocess your data into feature tensors (`X_train`, `X_test`) and corresponding labels (`y_train`, `y_test`).

---

## **How to Run**
1. Clone this repository:  
 
2. Prepare your data and load it in the format required (with `X_train`, `X_test`, `y_train`, and `y_test`).

3. Execute the ipynb file
