"""
@author:yunsuxiaozi n twobob
@start_time:2024/9/27
@update_time:2024/9/28
"""
import polars as pl  # Similar to pandas but offers better performance on large datasets.
import pandas as pd  # Library for data manipulation and analysis.
import numpy as np   # Library for numerical computations.

from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, GroupKFold
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier, log_evaluation, early_stopping
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
import warnings  # To suppress warnings.
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output.

class Yunbase():
    def __init__(self, num_folds=5,
                 models=[],
                 FE=None,
                 seed=1337,
                 objective='regression',
                 metric='mse',
                 nan_margin=0.95,
                 group_col=None,
                 num_classes=None,
                 target_col='target',
                 ):
        """
        Parameters:
        - num_folds: Number of folds for cross-validation.
        - models: List of models to use. Default models are provided as baseline, but you can specify your own.
        - FE: Custom feature engineering function in addition to the basic feature engineering provided.
        - seed: Random seed for reproducibility.
        - objective: Task type: 'regression', 'binary', or 'multi_class'.
        - metric: Evaluation metric to use.
        - nan_margin: Threshold for dropping columns with missing values.
        - group_col: Column to use for grouping in GroupKFold.
        - num_classes: Number of classes for classification tasks.
        - target_col: Name of the target column to predict.
        """
        self.num_folds = num_folds
        self.seed = seed
        self.models = models
        self.FE = FE
        self.objective = objective
        self.metric = metric
        self.nan_margin = nan_margin
        self.group_col = group_col
        self.target_col = target_col
        self.num_classes = num_classes
        self.pretrained_models = {}  # Dictionary to save trained models

    def get_details(self):
        # Currently supported evaluation metrics
        metrics = ['rmse', 'mse', 'accuracy', 'logloss', 'auc', 'f1']
        # Currently supported models
        models = ['lgb', 'xgb', 'catboost', 'linear', 'logistic']
        # Currently supported cross-validation methods
        kfolds = ['KFold', 'GroupKFold', 'StratifiedKFold', 'StratifiedGroupKFold']
        # Currently supported objectives
        objectives = ['binary', 'multi_class', 'regression']
        print(f"Currently supported metrics: {metrics}")
        print(f"Currently supported models: {models}")
        print(f"Currently supported cross-validation methods: {kfolds}")
        print(f"Currently supported objectives: {objectives}")

    # Perform feature engineering on training or test data, mode='train' or 'test'
    def Feature_Engineer(self, df, mode='train'):
        if self.FE is not None:
            # Apply custom feature engineering function
            df = self.FE(df)
        if mode == 'train':
            # Drop columns with too many missing values
            self.nan_cols = [col for col in df.columns if df[col].isna().mean() > self.nan_margin]
            # Drop columns with only one unique value
            self.unique_cols = [col for col in df.columns if df[col].nunique() == 1]
            # Identify categorical columns (strings), except for group column
            self.object_cols = [col for col in df.columns if (df[col].dtype == 'object') and (col != self.group_col)]
            # Label encoding for object columns
            self.label_encoders = {}
            for col in self.object_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            # One-hot encode columns with few unique values
            self.one_hot_cols = []
            for col in df.columns:
                if col != self.target_col and col != self.group_col:
                    if (df[col].nunique() < 20) and (df[col].nunique() > 2):
                        self.one_hot_cols.append([col, list(df[col].unique())]) 
            # Save numerical columns for scaling
            self.numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if self.target_col in self.numeric_cols:
                self.numeric_cols.remove(self.target_col)
            if self.group_col in self.numeric_cols:
                self.numeric_cols.remove(self.group_col)
            # Fit scaler on numerical columns
            self.scaler = StandardScaler()
            df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])
        else:
            # For test data, use the saved encoders and scalers
            # Label encoding
            for col in self.object_cols:
                if col in df.columns:
                    le = self.label_encoders.get(col)
                    if le:
                        df[col] = df[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
                        le.classes_ = np.append(le.classes_, '<unknown>')
                        df[col] = le.transform(df[col].astype(str))
            # Scaling numerical columns
            df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        # One-hot encoding
        for i in range(len(self.one_hot_cols)):
            col, unique_values = self.one_hot_cols[i]
            for u in unique_values:
                df[f"{col}_{u}"] = (df[col] == u).astype(np.int8)
        # Drop unnecessary columns
        df.drop(self.nan_cols + self.unique_cols, axis=1, inplace=True, errors='ignore')
        return df

    def Metric(self, y_true, y_pred):
        if self.metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'logloss':
            return log_loss(y_true, y_pred)
        elif self.metric == 'auc':
            return roc_auc_score(y_true, y_pred)
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred, average='macro')
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def fit(self, train_path_or_file='train.csv'):
        try:  # Try to read from path
            self.train = pl.read_csv(train_path_or_file)
            self.train = self.train.to_pandas()
        except:  # Assume it's a DataFrame
            self.train = train_path_or_file
        # Check if provided training data is a DataFrame
        if not isinstance(self.train, pd.DataFrame):
            raise ValueError("train_path_or_file is not a pandas DataFrame")
        self.train = self.Feature_Engineer(self.train, mode='train')
        
        # Check if the objective is supported
        if self.objective.lower() not in ['binary', 'multi_class', 'regression']:
            raise ValueError("Unsupported or currently unsupported objective")
        
        # Choose the cross-validation method
        if self.objective.lower() in ['binary', 'multi_class']:
            if self.group_col is not None:  # Grouped
                kf = StratifiedGroupKFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
            else:
                kf = StratifiedKFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
        else:  # Regression task
            if self.group_col is not None:  # Grouped
                kf = GroupKFold(n_splits=self.num_folds)
            else:
                kf = KFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
        
        # Prepare models: If you have prepared models, use them; otherwise, use default models
        if len(self.models) == 0:
            metric = self.metric.lower()
            if self.objective.lower() == 'multi_class':
                metric = 'multi_logloss'
            # Define default models
            default_models = []
            lgb_params = {
                "boosting_type": "gbdt",
                "metric": metric,
                'random_state': self.seed,
                "max_depth": 10,
                "learning_rate": 0.05,
                "n_estimators": 10000,
                "colsample_bytree": 0.6,
                "colsample_bynode": 0.6,
                "verbose": -1,
                "reg_alpha": 0.2,
                "reg_lambda": 5,
                "extra_trees": True,
                'num_leaves': 64,
                "max_bin": 255,
            }
            if self.objective.lower() == 'regression':
                default_models.append((LGBMRegressor(**lgb_params), 'lgb'))
                default_models.append((XGBRegressor(random_state=self.seed, n_estimators=10000, learning_rate=0.05), 'xgb'))
                default_models.append((CatBoostRegressor(random_state=self.seed, verbose=False), 'catboost'))
            else:
                default_models.append((LGBMClassifier(**lgb_params), 'lgb'))
                default_models.append((XGBClassifier(random_state=self.seed, n_estimators=10000, learning_rate=0.05), 'xgb'))
                default_models.append((CatBoostClassifier(random_state=self.seed, verbose=False), 'catboost'))
            self.models = default_models
        
        X = self.train.drop([self.group_col, self.target_col], axis=1, errors='ignore')
        y = self.train[self.target_col]
        
        # Map column names to standardized names
        self.col2name = {}
        for i, col in enumerate(list(X.columns)):
            self.col2name[col] = f'col_{i}'
        X = X.rename(columns=self.col2name)
        
        # Convert object columns to float32
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype(np.float32)
        
        # For classification tasks, create target2idx and idx2target mappings
        if self.objective.lower() != 'regression':
            self.target2idx = {}
            self.idx2target = {}
            y_unique = sorted(list(y.unique()))
            for i, val in enumerate(y_unique):
                self.target2idx[val] = i
                self.idx2target[i] = val
            y = y.apply(lambda k: self.target2idx[k])
        
        if self.group_col is not None:
            group = self.train[self.group_col]
        else:
            group = None
        
        for model, model_name in self.models:
            oof = np.zeros(len(y))
            for fold, (train_index, valid_index) in enumerate(kf.split(X, y, groups=group)):
                print(f"Model: {model_name}, Fold: {fold}")
                X_train, X_valid = X.iloc[train_index].reset_index(drop=True), X.iloc[valid_index].reset_index(drop=True)
                y_train, y_valid = y.iloc[train_index].reset_index(drop=True), y.iloc[valid_index].reset_index(drop=True)
                
                if model_name == 'lgb':
                    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                              callbacks=[log_evaluation(100), early_stopping(200)])
                elif model_name == 'xgb':
                    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False,
                              early_stopping_rounds=200)
                elif model_name == 'catboost':
                    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], use_best_model=True, verbose=False)
                else:
                    model.fit(X_train, y_train)
                
                if self.objective.lower() == 'regression':
                    oof[valid_index] = model.predict(X_valid)
                else:
                    if self.num_classes == 2:
                        oof[valid_index] = model.predict_proba(X_valid)[:, 1]
                    else:
                        oof[valid_index] = np.argmax(model.predict_proba(X_valid), axis=1)
                
                self.pretrained_models[f'{model_name}_fold{fold}'] = model
            print(f"{self.metric}: {self.Metric(y.values, oof)}")

    def predict(self, test_path_or_file='test.csv'):
        try:  # Try to read from path
            self.test = pl.read_csv(test_path_or_file)
            self.test = self.test.to_pandas()
        except:  # Assume it's a DataFrame
            self.test = test_path_or_file
        # Check if provided test data is a DataFrame
        if not isinstance(self.test, pd.DataFrame):
            raise ValueError("test_path_or_file is not a pandas DataFrame")
        self.test = self.Feature_Engineer(self.test, mode='test')
        self.test = self.test.drop([self.group_col, self.target_col], axis=1, errors='ignore')
        self.test = self.test.rename(columns=self.col2name)
        for col in self.test.columns:
            if self.test[col].dtype == 'object':
                self.test[col] = self.test[col].astype(np.float32)
        if self.objective.lower() == 'regression':
            test_preds = np.zeros((len(self.pretrained_models), len(self.test)))
            fold = 0
            for model_name, model in self.pretrained_models.items():
                test_preds[fold] = model.predict(self.test)
                fold += 1
            return test_preds.mean(axis=0)
        else:
            if self.num_classes == 2:
                test_preds = np.zeros((len(self.pretrained_models), len(self.test)))
                fold = 0
                for model_name, model in self.pretrained_models.items():
                    test_preds[fold] = model.predict_proba(self.test)[:, 1]
                    fold += 1
                avg_preds = test_preds.mean(axis=0)
                return (avg_preds > 0.5).astype(int)
            else:
                test_preds = np.zeros((len(self.pretrained_models), len(self.test), self.num_classes))
                fold = 0
                for model_name, model in self.pretrained_models.items():
                    test_preds[fold] = model.predict_proba(self.test)
                    fold += 1
                avg_preds = test_preds.mean(axis=0)
                return np.argmax(avg_preds, axis=1)

    def submit(self, submission_path='submission.csv', test_preds=None):
        submission = pd.read_csv(submission_path)
        submission[self.target_col] = test_preds
        if self.objective.lower() != 'regression':
            submission[self.target_col] = submission[self.target_col].apply(lambda x: self.idx2target.get(x, x))
        submission.to_csv("yunbase_submission.csv", index=False)
        print(submission.head())
