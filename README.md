
# Yunbase

Yunbase is a comprehensive Python class designed to streamline repetitive tasks in data mining and machine learning competitions. It handles common operations such as data preprocessing, k-fold cross-validation, model training, and prediction, allowing you to focus on crafting winning solutions. The name "Yunbase" combines "Yun" from my online alias and "base" to signify its role as a baseline framework for algorithm competitions.

## Quick Start

### 1. Clone the Project

Clone the Yunbase repository to your local machine:

```bash
git clone https://github.com/yunsuxiaozi/Yunbase.git
```

### 2. Import Yunbase

Import the `Yunbase` class from the `baseline` module:

```python
from Yunbase.baseline import Yunbase
```

### 3. Create a Yunbase Instance

Initialize a `Yunbase` object with your desired configuration:

```python
yunbase = Yunbase(
    num_folds=5,
    models=[],           # Optional: List of custom models
    FE=None,             # Optional: Custom feature engineering function
    seed=2024,
    objective='regression',
    metric='rmse',
    nan_margin=0.95,
    group_col='p_num',   # Optional: Column name for group k-fold
    target_col='bg+1:00',
)
```

#### Parameters:

- **num_folds**: *(int)* Number of folds for cross-validation.
- **models**: *(list)* List of custom models to use. If empty, default models are used.
- **FE**: *(function)* Custom feature engineering function. Should accept and return a DataFrame (`df = FE(df)`).
- **seed**: *(int)* Random seed for reproducibility.
- **objective**: *(str)* Type of task. Options are `'binary'`, `'multi_class'`, or `'regression'`.
- **metric**: *(str)* Evaluation metric. Supported metrics include `'rmse'`, `'mse'`, `'accuracy'`, `'logloss'`, `'auc'`, and `'f1'`.
- **nan_margin**: *(float)* Threshold for dropping columns with missing values.
- **group_col**: *(str)* Column name for grouping in group k-fold cross-validation.
- **target_col**: *(str)* Name of the target column to predict.

### 4. Train the Models

Train the models using your training data. You can provide the path to a CSV file or a DataFrame:

```python
yunbase.fit(train_path_or_file="train.csv")
```

### 5. Make Predictions

Generate predictions on the test set:

```python
test_preds = yunbase.predict(test_path_or_file="test.csv")
```

### 6. Save Prediction Results

Create a submission file by replacing the `target_col` in your sample submission file with the predictions:

```python
yunbase.submit(submission_path='sample_submission.csv', test_preds=test_preds)
```

This will generate a file named `yunbase_submission.csv` with the updated predictions.

## Detailed Usage

### Custom Feature Engineering

You can define your own feature engineering function and pass it to the `FE` parameter:

```python
def custom_fe(df):
    # Example: Create interaction features
    df['new_feature'] = df['feature1'] * df['feature2']
    # Additional transformations
    return df

yunbase = Yunbase(
    FE=custom_fe,
    # Other parameters...
)
```

### Custom Models

You can supply your own models by passing a list of tuples containing the model instances and their names:

```python
from sklearn.linear_model import LinearRegression

custom_models = [
    (LinearRegression(), 'linear_regression'),
    # Add other models
]

yunbase = Yunbase(
    models=custom_models,
    # Other parameters...
)
```

### Supported Evaluation Metrics

The following evaluation metrics are supported:

- **Regression**: `'rmse'`, `'mse'`
- **Classification**: `'accuracy'`, `'logloss'`, `'auc'`, `'f1'`

### Supported Cross-Validation Methods

Depending on the task and data, Yunbase automatically selects an appropriate cross-validation method:

- **KFold**
- **StratifiedKFold**
- **GroupKFold**
- **StratifiedGroupKFold**

### Example Workflow

```python
# Import Yunbase
from Yunbase.baseline import Yunbase

# Initialize Yunbase
yunbase = Yunbase(
    num_folds=5,
    seed=42,
    objective='binary',
    metric='auc',
    target_col='target'
)

# Train models
yunbase.fit('train.csv')

# Make predictions
test_preds = yunbase.predict('test.csv')

# Submit predictions
yunbase.submit('sample_submission.csv', test_preds)
```

## Future Work

The current version of Yunbase provides a solid framework for handling common tasks in machine learning competitions. Future improvements will focus on:

- **Bug Fixes**: Ongoing efforts to identify and resolve any issues.
- **Feature Expansion**: Adding support for more models, metrics, and preprocessing techniques.
- **Hyperparameter Optimization**: Integrating automated hyperparameter tuning.
- **Ensemble Techniques**: Implementing stacking and blending methods for better performance.

Stay tuned for updates!

*2024/9/27*

## Reference Tutorial

For a practical demonstration of how to use Yunbase, check out the Kaggle notebook:

[Yunbase Tutorial on Kaggle](https://www.kaggle.com/code/yunsuxiaozi/brist1d-yunbase)

---

Feel free to contribute to the project or raise issues on [GitHub](https://github.com/yunsuxiaozi/Yunbase).
