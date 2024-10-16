
# IMPORTS

#---------------------------------------------------------------

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.isotonic import IsotonicRegression
from venn_abers import VennAbers, VennAbersCalibrator
import h2o

class BooleanToIntWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator
        
    def fit(self, X, y):
        return self.estimator.fit(X, y.astype(int))
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

class OriginalAndCalibratedWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, inductive=False, n_splits=5, random_state=None):
        self.base_estimator = base_estimator
        self.inductive = inductive
        self.n_splits = n_splits
        self.random_state = random_state
        self.vac = None
        self.le = LabelEncoder()

    def fit(self, X, y):
        self.le.fit(y)
        y_encoded = self.le.transform(y)
        self.classes_ = self.le.classes_

        if 'CatBoost' in self.base_estimator.__class__.__name__:
            self.base_estimator.fit(X, y_encoded)
        else:
            self.base_estimator.fit(X, y)

        wrapped_estimator = BooleanToIntWrapper(self.base_estimator)

        self.vac = VennAbersCalibrator(
            estimator=wrapped_estimator,
            inductive=self.inductive,
            n_splits=self.n_splits,
            random_state=self.random_state
        )
        self.vac.fit(X, y_encoded)
        return self

    def predict(self, X):
        check_is_fitted(self, ['base_estimator', 'vac'])
        return self.le.inverse_transform(self.base_estimator.predict(X))

    def predict_proba(self, X):
        check_is_fitted(self, ['base_estimator', 'vac'])
        return self.vac.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.base_estimator, 'decision_function'):
            return self.base_estimator.decision_function(X)
        else:
            proba = self.base_estimator.predict_proba(X)
            return np.log(proba[:, 1] / proba[:, 0])

class CalibrationWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, increasing=True):
        self.base_estimator = base_estimator
        self.increasing = increasing
        self.isotonic_regressor = IsotonicRegression(increasing=self.increasing)
    
    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        y_pred = self.base_estimator.predict(X)
        self.isotonic_regressor.fit(y_pred, y)
        return self
    
    def predict(self, X):
        y_pred = self.base_estimator.predict(X)
        y_calibrated = self.isotonic_regressor.transform(y_pred)
        return y_calibrated
    
    def predict_raw(self, X):
        return self.base_estimator.predict(X)

class CalibratedModelClassification:
    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, data):
        if isinstance(data, pd.DataFrame):
            data_h2o = h2o.H2OFrame(data)
            for col in data.columns:
                if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                    data_h2o[col] = data_h2o[col].asfactor()
        elif isinstance(data, h2o.H2OFrame):
            data_h2o = data
        else:
            raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

        preds = self.model.predict(data_h2o)
        preds_df = preds.as_data_frame()

        if 'p0' in preds_df.columns and 'p1' in preds_df.columns:
            p_raw = preds_df[['p0', 'p1']].values
        else:
            raise KeyError("Class probabilities not found in model output.")

        calibrated_probs = self.calibrator.predict_proba(p_raw)

        prob_lower = calibrated_probs[0]
        prob_upper = calibrated_probs[1]
        prob_mean = (prob_lower + prob_upper) / 2

        return prob_mean[:, 1]

    def predict_dataframe(self, data, include_data=True, threshold=0.5):
        if isinstance(data, pd.DataFrame):
            data_h2o = h2o.H2OFrame(data)
            for col in data.columns:
                if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                    data_h2o[col] = data_h2o[col].asfactor()
        elif isinstance(data, h2o.H2OFrame):
            data_h2o = data
        else:
            raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

        preds = self.model.predict(data_h2o)
        preds_df = preds.as_data_frame()

        p0 = preds_df['p0'].values
        p1 = preds_df['p1'].values
        p_raw = np.column_stack((p0, p1))

        calibrated_probs = self.calibrator.predict_proba(p_raw)

        prob_lower = calibrated_probs[0]
        prob_upper = calibrated_probs[1]
        prob_mean = (prob_lower + prob_upper) / 2

        prob_sum = prob_mean.sum(axis=1).reshape(-1, 1)
        normalized_probs = prob_mean / prob_sum

        preds_df['Score_0'] = normalized_probs[:, 0]
        preds_df['Score_1'] = normalized_probs[:, 1]
        preds_df['Label'] = (preds_df['Score_1'] > threshold).astype(int)

        if include_data:
            return pd.concat([data.reset_index(drop=True), preds_df[['Score_0', 'Score_1', 'Label']]], axis=1)
        else:
            return preds_df[['Score_0', 'Score_1', 'Label']]

    def save(self, directory):
        self.model_path = h2o.save_model(model=self.model, path=directory, force=True)
        save_path = os.path.join(directory, "calibrated_model.pkl")
        joblib.dump({'calibrator': self.calibrator, 'model_path': self.model_path}, save_path)
        print(f"CalibratedModel saved to: {save_path}")

    @classmethod
    def load(cls, directory):
        load_path = os.path.join(directory, "calibrated_model.pkl")
        data = joblib.load(load_path)
        model = h2o.load_model(data['model_path'])
        calibrator = data['calibrator']
        return cls(model, calibrator)

class CalibratedModelWithOptimizedThreshold:
    def __init__(self, calibrated_model, threshold=0.5):
        self.calibrated_model = calibrated_model  # calibrated model (Venn-Abers or similar)
        self.threshold = threshold                # optimized threshold

    def predict_proba(self, h2o_test):
        return self.calibrated_model.predict_proba(h2o_test)
    
    def predict_dataframe(self, data, include_data=True):
        preds_df = self.calibrated_model.predict_dataframe(data, include_data=include_data, threshold=self.threshold)
        return preds_df

    def predict(self, h2o_test):
        y_pred_proba = self.predict_proba(h2o_test)
        return (y_pred_proba > self.threshold).astype(int)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

class CalibratedModelClassificationMulti:
    def __init__(self, model, calibrators, classes):
        self.model = model
        self.calibrators = calibrators
        self.classes = np.array(classes)

    def predict_proba(self, data):
        if isinstance(data, pd.DataFrame):
            data_h2o = h2o.H2OFrame(data)
            for col in data.columns:
                if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                    data_h2o[col] = data_h2o[col].asfactor()
        elif isinstance(data, h2o.H2OFrame):
            data_h2o = data
        else:
            raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

        preds = self.model.predict(data_h2o)
        preds_df = preds.as_data_frame()

        prob_cols = [f"p{cls}" for cls in self.classes]
        p_raw = preds_df[prob_cols].values

        calibrated_probs = np.zeros_like(p_raw)
        for idx, cls in enumerate(self.classes):
            calibrated_probs[:, idx] = self.calibrators[cls].transform(p_raw[:, idx])

        calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)

        return calibrated_probs
    
    def predict_dataframe(self, data, include_data=True, threshold=0.0):
        if isinstance(data, pd.DataFrame):
            data_h2o = h2o.H2OFrame(data)
            for col in data.columns:
                if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                    data_h2o[col] = data_h2o[col].asfactor()
        elif isinstance(data, h2o.H2OFrame):
            data_h2o = data
        else:
            raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

        preds = self.model.predict(data_h2o)
        preds_df = preds.as_data_frame()

        prob_cols = [f"p{cls}" for cls in self.classes]
        p_raw = preds_df[prob_cols].values

        calibrated_probs = np.zeros_like(p_raw)
        for idx, cls in enumerate(self.classes):
            calibrated_probs[:, idx] = self.calibrators[cls].transform(p_raw[:, idx])

        calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)

        for idx, cls in enumerate(self.classes):
            preds_df[f"Score_{cls}"] = calibrated_probs[:, idx]

        max_probs = np.max(calibrated_probs, axis=1)
        predicted_classes = self.classes[np.argmax(calibrated_probs, axis=1)]

        pred_labels = np.where(max_probs >= threshold, predicted_classes, 'Unknown')

        preds_df['Label'] = pred_labels

        preds_df['Max_Probability'] = max_probs

        score_cols = [f"Score_{cls}" for cls in self.classes]
        preds_df = preds_df[score_cols + ['Label', 'Max_Probability']]

        if include_data:
            if isinstance(data, h2o.H2OFrame):
                data = data.as_data_frame()
            return pd.concat([data.reset_index(drop=True), preds_df], axis=1)
        else:
            return preds_df

    def predict(self, data, threshold=0.5):
        calibrated_probs = self.predict_proba(data)
        pred_labels = self.classes[np.argmax(calibrated_probs, axis=1)]
        return pred_labels

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        model_path = h2o.save_model(model=self.model, path=directory, force=True)
        
        calibrator_path = os.path.join(directory, "calibrators.pkl")
        joblib.dump(self.calibrators, calibrator_path)
        
        metadata = {
            "model_path": model_path,
            "classes": self.classes.tolist()  # json compatibility
        }
        metadata_path = os.path.join(directory, "metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        print(f"CalibratedModel saved to: {directory}")

    @classmethod
    def load(cls, directory):
        metadata_path = os.path.join(directory, "metadata.pkl")
        metadata = joblib.load(metadata_path)
        model = h2o.load_model(metadata["model_path"])
        
        calibrator_path = os.path.join(directory, "calibrators.pkl")
        calibrators = joblib.load(calibrator_path)
        
        classes = metadata["classes"]
        
        return cls(model, calibrators, classes)

class CalibratedModelRegression:
    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator

    def predict(self, data):
        if isinstance(data, pd.DataFrame):
            data_h2o = h2o.H2OFrame(data)
        elif isinstance(data, h2o.H2OFrame):
            data_h2o = data
        else:
            raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

        preds = self.model.predict(data_h2o)
        preds_df = preds.as_data_frame()
        
        p_raw = preds_df['predict'].values
        
        calibrated_preds = self.calibrator.predict(p_raw)
        
        return calibrated_preds

    def predict_dataframe(self, data, include_data=True):
        calibrated_preds = self.predict(data)
        
        preds_df = pd.DataFrame({'Calibrated_Prediction': calibrated_preds})
        
        if include_data:
            return pd.concat([data.reset_index(drop=True), preds_df], axis=1)
        else:
            return preds_df

    def save(self, directory):
        self.model_path = h2o.save_model(model=self.model, path=directory, force=True)
        save_path = os.path.join(directory, "calibrated_model.pkl")
        joblib.dump({'calibrator': self.calibrator, 'model_path': self.model_path}, save_path)
        print(f"CalibratedModel saved to: {save_path}")

    @classmethod
    def load(cls, directory):
        load_path = os.path.join(directory, "calibrated_model.pkl")
        data = joblib.load(load_path)
        calibrator = data['calibrator']
        model_path = data['model_path']
        model = h2o.load_model(model_path)
        return cls(model, calibrator)

def ML_Util(
    train_size = 0.8,
    fold = 10,
    y_col = "",
    y_col_type = "",
    apply_weights = True,
    force_balanced_metric = False,
    calibration_ratio = 0.15,
    seed = 123,
    log_experiment_name = "",
    pycaret_models = True,
    pycaret_data = None,
    pycaret_metric_prioritize = "",
    pycaret_n_iter = 10,
    pycaret_n_select = 3,
    pycaret_categorical_features = None,
    pycaret_ordinal_features = None,
    pycaret_numeric_features = None,
    pycaret_fold_strategy = "stratifiedkfold",
    pycaret_threshold_grid_interval = 0.02,
    pycaret_use_gpu_boost = True,
    pycaret_use_CUDA = False,
    pycaret_reg_calibration_IncreasingRelationship = True,
    pycaret_CatBoost_parameters = {
        "iterations": [1000],
        "learning_rate": [0.01],
        "l2_leaf_reg": [10],
        "colsample_bylevel": [0.8],
        "subsample": [0.8],
        "depth": [6],
        "random_strength": [4]
    },
    pycaret_LGBM_parameters = {
        'n_estimators': [1000],
        'learning_rate': [0.01],
        'max_depth': [8],
        'reg_alpha': [100],
        'reg_lambda': [10],
        'colsample_bytree': [0.8],
        'subsample': [0.8]
    },
    pycaret_LogisticRegression_parameters = {
        'C': [0.01],
        "intercept_scaling": [1],
        "l1_ratio": [0.2],
        "max_iter": [1000],
        "penalty": ['elasticnet'],
        "solver": ['saga'],
        "tol": [0.001]
    },
    pycaret_RandomForest_parameters = {
        'n_estimators': [1000],
        'min_samples_leaf': [4],
        'min_samples_split': [4],
        'max_depth': [8],
        'ccp_alpha': [0.01],
        'oob_score': [True]
    },

    pycaret_ExtraTrees_parameters = {
        'n_estimators': [1000],
        'min_samples_leaf': [4],
        'min_samples_split': [4],
        'max_depth': [8],
        'ccp_alpha': [0.01],
        'oob_score': [True],
        'bootstrap': [True]
    },
    pycaret_DecisionTree_parameters = {
        'min_samples_leaf': [4],
        'min_samples_split': [4],
        'max_depth': [8],
        'ccp_alpha': [0.01]
    },
    pycaret_SGD_parameters = {
        'alpha': [0.001],
        'early_stopping': [True],
        'tol': [0.001],
        'validation_fraction': [0.2],
        'l1_ratio': [0.1]
    },
    pycaret_Ridge_parameters = {
        'alpha': [10.0],
        'max_iter': [1000],
        'tol': [0.001]
    },
    pycaret_XGB_parameters = {
        'n_estimators': [1000], 
        "learning_rate": [0.01], 
        "reg_alpha": [180], 
        "reg_lambda": [10],
        "subsample": [0.8], 
        "colsample_bytree": [0.8]
    },
    pycaret_GBC_parameters = {
        'n_estimators': [1000],
        'learning_rate': [0.01],
        'ccp_alpha': [0],
        'tol': [0.0001],
        'min_samples_split': [4],
        'min_samples_leaf': [4],
        'max_depth': [8],
        'subsample': [0.8],
        'validation_fraction': [0]
    },
    pycaret_ADA_parameters = {
        'n_estimators': [70],
        'learning_rate': [0.01],
        'base_estimator': [None]
    },
    pycaret_LinearDiscriminant_parameters = {
        'shrinkage': ['auto'],
        'solver': ['lsqr'],
        'tol': [0.0001]
    },
    pycaret_KNeighbors_parameters = {
        'n_neighbors': [3, 5, 7],
        'leaf_size': [10, 30],
        'p': [1, 2]
    },
    pycaret_QuadraticDiscriminant_parameters = {
        'reg_param': [0.55, 0.6],
        'tol': [0.0001]
    },
    pycaret_Dummy_parameters = {
        'constant': [None], 
        'strategy': ['prior', 'most_frequent']
    },
    pycaret_GaussianNB_parameters = {
        'priors': [None], 
        'var_smoothing': [1e-09, 5e-09, 1e-08]
    },
    pycaret_reg_LassoLars_parameters = {
        "alpha": [1.0],          # Regularization strength; higher values reduce overfitting
        "max_iter": [500],       # Maximum number of iterations
        "normalize": [True]     # Normalizes the regressors X before regression
    },
    pycaret_reg_ElasticNet_parameters = {
        "alpha": [1.0],          # Overall regularization strength
        "l1_ratio": [0.5],       # Balance between Lasso (L1) and Ridge (L2) regularization
        "max_iter": [1000]       # Maximum number of iterations
    },
    pycaret_reg_BayesianRidge_parameters = {
        "alpha_1": [1e-06],      # Hyperparameter for the Gamma distribution prior over alpha
        "alpha_2": [1e-06],      # Hyperparameter for the Gamma distribution prior over alpha
        "lambda_1": [1e-06],     # Hyperparameter for the Gamma distribution prior over lambda
        "lambda_2": [1e-06],     # Hyperparameter for the Gamma distribution prior over lambda
        "n_iter": [1000]          # Number of iterations for the optimization
    },
    pycaret_reg_Lasso_parameters = {
        "alpha": [1.0],          # Regularization strength; higher values reduce overfitting
        "max_iter": [1000]       # Maximum number of iterations
    },
    pycaret_reg_Lars_parameters = {
        "n_nonzero_coefs": [500] # Maximum number of non-zero coefficients; controls model complexity
    },
    pycaret_reg_Ridge_parameters = {
        "alpha": [1.0]           # Regularization strength; higher values reduce overfitting
    },
    pycaret_reg_LinearRegression_parameters = {
        "n_jobs": [1]            # Number of jobs to run in parallel; not directly related to overfitting
    },
    pycaret_reg_HuberRegressor_parameters = {
        "alpha": [0.0001],       # Regularization strength; higher values reduce overfitting
        "epsilon": [1.35],       # The epsilon parameter in the Huber loss function
        "max_iter": [1000]        # Maximum number of iterations
    },
    pycaret_reg_OMP_parameters = {
        "n_nonzero_coefs": [None], # Number of non-zero coefficients; can limit model complexity
        "tol": [None]              # Tolerance for the optimization; lower values can lead to simpler models
    },
    pycaret_reg_AdaBoost_parameters = {
        "n_estimators": [70],       # Number of base estimators; more estimators can improve performance but may increase overfitting
        "learning_rate": [0.01],     # Learning rate shrinks the contribution of each regressor; lower values can reduce overfitting
        "loss": ['linear']          # Loss function to use when updating the weights
    },
    pycaret_reg_GBR_parameters = {
        "n_estimators": [1000],      # Number of boosting stages to perform
        "learning_rate": [0.01],     # Learning rate shrinks the contribution of each tree
        "max_depth": [3],           # Maximum depth of the individual regression estimators
        "subsample": [1.0]          # Fraction of samples to be used for fitting the individual base learners
    },
    pycaret_reg_RandomForest_parameters = {
        "n_estimators": [100],      # Number of trees in the forest
        "max_depth": [None],        # Maximum depth of the tree; limiting can reduce overfitting
        "min_samples_split": [2],   # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1],    # Minimum number of samples required to be at a leaf node
        "max_features": ['auto']    # Number of features to consider when looking for the best split
    },
    pycaret_reg_ExtraTrees_parameters = {
        "n_estimators": [100],      # Number of trees in the forest
        "max_depth": [None],        # Maximum depth of the tree; limiting can reduce overfitting
        "min_samples_split": [2],   # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1],    # Minimum number of samples required to be at a leaf node
        "max_features": ['auto']    # Number of features to consider when looking for the best split
    },
    pycaret_reg_DecisionTree_parameters = {
        "max_depth": [None],        # Maximum depth of the tree; limiting can reduce overfitting
        "min_samples_split": [2],   # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1]     # Minimum number of samples required to be at a leaf node
    },
    pycaret_reg_LGBM_parameters = {
        "learning_rate": [0.01],     # Learning rate shrinks the contribution of each tree
        "num_leaves": [31],         # Maximum tree leaves for base learners; lower values prevent overfitting
        "max_depth": [8],          # Limits the depth of the tree; -1 means no limit
        "min_child_samples": [20],  # Minimum number of data needed in a child; larger values prevent overfitting
        "subsample": [1.0],         # Fraction of data to be used for fitting the individual base learners
        "colsample_bytree": [1.0],  # Fraction of features to be used for fitting the individual base learners
        "reg_alpha": [0.0],         # L1 regularization term on weights
        "reg_lambda": [0.0]         # L2 regularization term on weights
    },
    pycaret_reg_CatBoost_parameters = {
        "iterations": [1000],        # Number of boosting iterations
        "depth": [6],               # Depth of the trees; lower values prevent overfitting
        "learning_rate": [0.01],     # Learning rate shrinks the contribution of each tree
        "l2_leaf_reg": [3],         # L2 regularization coefficient
        "verbose": [False]              # Verbosity level; 0 suppresses output
    },
    pycaret_reg_KNeighbors_parameters = {
        "n_neighbors": [5],        # Number of neighbors to use
        "weights": ['uniform']     # Weight function used in prediction; 'uniform' assigns equal weight
    },
    pycaret_reg_PassiveAggressive_parameters = {
        "C": [1.0],                # Regularization parameter; smaller values specify stronger regularization
        "epsilon": [0.1],          # Epsilon in the epsilon-insensitive loss functions
        "max_iter": [1000],        # Maximum number of passes over the training data
        "tol": [0.001]             # Tolerance for the stopping criteria
    },
    pycaret_reg_Dummy_parameters = {
        "strategy": ['mean']       # Strategy to use when predicting; 'mean' predicts the mean of the training targets
    },
    pycaret_reg_XGB_parameters = {
       'learning_rate': [0.01],
        'max_depth': [8],
        'n_estimators': [1000],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [10],
        'reg_lambda': [100]
    },
    h2o_models = False,
    h2o_data = None,
    h2o_metric_prioritize = "AUTO",
    h2o_optimize_threshold_metric = "auc",
    h2o_threshold_grid_interval = 0.02,
    h2o_max_models = None,
    h2o_x_cols = None
):
    """
    Machine Learning Utility Function for Automated Model Training, Tuning, and Calibration using PyCaret and H2O.

    This function automates the process of setting up, training, tuning, and calibrating machine learning models for both classification and regression tasks using PyCaret and H2O AutoML. It supports both binary and multiclass classification, as well as regression tasks. The function allows for extensive customization through various parameters, including model hyperparameters, data preprocessing options, and calibration methods.

    Depending on the task type (`y_col_type`), the function uses PyCaret and/or H2O to train models, tune hyperparameters, evaluate performance, and apply calibration techniques to improve probability estimates or predictions. The function also handles class imbalance by applying class weights if `apply_weights` is set to True.


    Notes+
    ------
    
    * (Pycaret) You may want to encode binary categorical features as numeric (even y), this increases accuracy. BUT BE AWARE; putting y inside 'pycaret_numeric_features' will crash the code. YOU HAVE TO ENCODE y AS NUMERIC INSIDE THE DATAFRAME YOU ARE GOING TO USE. 

    * (Pycaret) Categorical features will get one-hot encoded by Pycaret, which creates a column for each category. This can create a high-complexity model (lots of columns) so you want to be careful with categorical columns.
    
    * (H2O) MODELS MADE WITH H2O ARE NOT CROSS-COMPATIBLE BETWEEN ITS VERSIONS. SO CARE FOR H2O VERSION.


    Parameters
    ----------
    train_size : float, default=0.8
        Proportion of the dataset to include in the training split. Should be between 0.0 and 1.0.

    fold : int, default=10
        Number of folds to use for cross-validation during model training and evaluation.

    y_col : str
        Name of the target variable (dependent variable) column in the dataset.

    y_col_type : str
        Type of the target variable. Should be `"categorical"` for classification tasks or `"numeric"` for regression tasks.

    apply_weights : bool, default=True
        Whether to apply class weights to handle class imbalance during model training. Only applicable for classification tasks.

    seed : int, default=123
        Random seed for reproducibility.

    log_experiment_name : str, default=""
        Name of the experiment for logging purposes.

    PyCaret Parameters
    ------------------
    pycaret_models : bool, default=True
        Whether to use PyCaret for model training, tuning, and evaluation.

    pycaret_data : pandas.DataFrame or None, default=None
        The dataset to be used for PyCaret modeling. Should be a pandas DataFrame. Required if `pycaret_models` is True.

    pycaret_metric_prioritize : str, default=""
        The metric to prioritize during model selection and tuning in PyCaret.
        - For classification tasks: "Accuracy", "AUC", "Recall", "Precision", "F1", , "Balanced_Accuracy", etc.
        - For regression tasks: "R2", "RMSE", "MAE", etc.

    pycaret_n_iter : int, default=10
        Number of iterations for hyperparameter tuning in PyCaret.

    pycaret_n_select : int, default=3
        Number of top models to select after comparing models in PyCaret.

    pycaret_categorical_features : list of str or None, default=None
        List of column names in the dataset to be treated as categorical features in PyCaret.

    pycaret_ordinal_features : dict or None, default=None
        Dictionary of ordinal features and their categories in order, to be used in PyCaret.
        Example: `{'feature_name': ['low', 'medium', 'high']}`.

    pycaret_numeric_features : list of str or None, default=None
        List of column names in the dataset to be treated as numeric features in PyCaret.

    pycaret_fold_strategy : str, default="stratifiedkfold"
        Cross-validation fold strategy to use in PyCaret. Possible values include * 'kfold'* 'stratifiedkfold'* 'groupkfold'* 'timeseries'* a custom CV generator object compatible with scikit-learn.

    pycaret_use_gpu_boost : bool, default=True
        Whether to use GPU acceleration for models that support GPU in PyCaret (e.g., CatBoost, LightGBM).

    pycaret_use_CUDA : bool, default=False
        Whether to use CUDA for GPU acceleration in PyCaret (e.g., for XGBoost). Requires compatible hardware and software setup.

    pycaret_reg_calibration_IncreasingRelationship : bool, default=True
        Whether to assume an increasing relationship during isotonic regression calibration in regression tasks. If True, the calibration function is constrained to be non-decreasing.

    Model Hyperparameters (PyCaret)
    -------------------------------
    pycaret_CatBoost_parameters : dict, default={...}
        Hyperparameter grid for tuning CatBoostClassifier or CatBoostRegressor in PyCaret. Keys are parameter names, and values are lists of possible values to try during tuning.

    pycaret_LGBM_parameters : dict, default={...}
        Hyperparameter grid for tuning LightGBM models in PyCaret.

    pycaret_LogisticRegression_parameters : dict, default={...}
        Hyperparameter grid for tuning LogisticRegression in PyCaret.

    # ... similarly for other models ...

    H2O Parameters
    --------------
    h2o_models : bool, default=False
        Whether to use H2O AutoML for model training and tuning.

    h2o_data : pandas.DataFrame or None, default=None
        The dataset to be used for H2O modeling. Should be a pandas DataFrame. Required if `h2o_models` is True.

    h2o_metric_prioritize : str, default="AUTO"
        The metric to prioritize during model selection in H2O AutoML. Defaults to "AUTO" (This translates to "auc" for binomial classification, "mean_per_class_error" for multinomial classification, "deviance" for regression).
        
        For binomial classification, select from the following options:

        "auc"
        "aucpr"
        "logloss"
        "mean_per_class_error"
        "rmse"
        "mse"
        

        For multinomial classification, select from the following options:

        "mean_per_class_error"
        "logloss"
        "rmse"
        "mse"
        

        For regression, select from the following options:

        "deviance"
        "rmse"
        "mse"
        "mae"
        "rmlse"
    
    h2o_optimize_threshold_metric: str. default = "auc"
        Metric to maximize during threshold optimization for binary classification. Possible values are:
        
        'accuracy'
        'precision'
        'recall'
        'f1'
        'auc'
        'balanced_accuracy'

    h2o_max_models : int or None, default=None
        Maximum number of models to train during H2O AutoML.

    h2o_x_cols : list of str or None, default=None
        List of column names to be used as features (independent variables) in H2O modeling. If None, all columns except the target variable will be used.

    Returns
    -------
    None

    Side Effects
    ------------
    - Saves trained models, plots, and metrics to the current working directory in subfolders `pycaret_models` and `h2o_models`.
    - Generates and saves various plots such as learning curves, confusion matrices, feature importances, calibration curves, etc.
    - Prints model evaluation metrics to the console.

    Notes
    -----
    - The function requires the following libraries to be installed: PyCaret, H2O, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, venn_abers (for Venn-Abers calibration), catboost, lightgbm, xgboost, etc.
    - The function handles both classification and regression tasks based on the `y_col_type` parameter.
    - For classification tasks, the function applies class weighting to handle class imbalance if `apply_weights` is True.
    - For binary classification tasks, the function uses Venn-Abers calibration for probability estimates.
    - For multiclass classification tasks, the function uses isotonic regression for calibration.
    - For regression tasks, the function uses isotonic regression for calibration if `pycaret_reg_calibration_IncreasingRelationship` is True.
    - The function outputs are saved in directories named `pycaret_models` and `h2o_models` in the current working directory.
    - The function assumes that the input data is preprocessed appropriately, and that categorical features are specified if necessary.

    Examples
    --------

    >>> ML_Util(
    train_size = 0.8,
    fold = 10,
    y_col = "made_purchase",
    y_col_type = "categorical",
    apply_weights = True,
    seed = 1618033988,
    log_experiment_name = "func_test3",
    pycaret_models = True,
    pycaret_data = df_pycaret,
    pycaret_metric_prioritize = "precision",
    pycaret_n_iter = 10,
    pycaret_categorical_features = categorical_features,
    pycaret_ordinal_features = ordinal_features,
    pycaret_numeric_features = numeric_features,
    pycaret_fold_strategy = "kfold",
    pycaret_use_gpu_boost = True,
    pycaret_use_CUDA = False,
    pycaret_reg_calibration_IncreasingRelationship = True,
    pycaret_CatBoost_parameters = {
        "iterations": [1000],
        "learning_rate": [0.01],
        "l2_leaf_reg": [10],
        "colsample_bylevel": [0.8],
        "subsample": [0.8],
        "depth": [6],
        "random_strength": [4]
    },
    pycaret_LGBM_parameters = {
        'n_estimators': [1000],
        'learning_rate': [0.01],
        'max_depth': [8],
        'reg_alpha': [100],
        'reg_lambda': [10],
        'colsample_bytree': [0.8],
        'subsample': [0.8]
    },
    pycaret_LogisticRegression_parameters = {
        'C': [0.01],
        "intercept_scaling": [1],
        "l1_ratio": [0.2],
        "max_iter": [1000],
        "penalty": ['elasticnet'],
        "solver": ['saga'],
        "tol": [0.001]
    },
    pycaret_RandomForest_parameters = {
        'n_estimators': [1000],
        'min_samples_leaf': [4],
        'min_samples_split': [4],
        'max_depth': [8],
        'ccp_alpha': [0.01],
        'oob_score': [True]
    },

    pycaret_ExtraTrees_parameters = {
        'n_estimators': [1000],
        'min_samples_leaf': [4],
        'min_samples_split': [4],
        'max_depth': [8],
        'ccp_alpha': [0.01],
        'oob_score': [True],
        'bootstrap': [True]
    },
    pycaret_DecisionTree_parameters = {
        'min_samples_leaf': [4],
        'min_samples_split': [4],
        'max_depth': [8],
        'ccp_alpha': [0.01]
    },
    pycaret_SGD_parameters = {
        'alpha': [0.001],
        'early_stopping': [True],
        'tol': [0.001],
        'validation_fraction': [0.2],
        'l1_ratio': [0.1]
    },
    pycaret_Ridge_parameters = {
        'alpha': [10.0],
        'max_iter': [1000],
        'tol': [0.001]
    },
    pycaret_XGB_parameters = {
        'n_estimators': [1000], 
        "learning_rate": [0.01], 
        "reg_alpha": [180], 
        "reg_lambda": [10],
        "subsample": [0.8], 
        "colsample_bytree": [0.8]
    },
    pycaret_GBC_parameters = {
        'n_estimators': [1000],
        'learning_rate': [0.01],
        'ccp_alpha': [0],
        'tol': [0.0001],
        'min_samples_split': [4],
        'min_samples_leaf': [4],
        'max_depth': [8],
        'subsample': [0.8],
        'validation_fraction': [0]
    },
    pycaret_ADA_parameters = {
        'n_estimators': [70],
        'learning_rate': [0.01],
        'base_estimator': [None]
    },
    pycaret_LinearDiscriminant_parameters = {
        'shrinkage': ['auto'],
        'solver': ['lsqr'],
        'tol': [0.0001]
    },
    pycaret_KNeighbors_parameters = {
        'n_neighbors': [3, 5, 7],
        'leaf_size': [10, 30],
        'p': [1, 2]
    },
    pycaret_QuadraticDiscriminant_parameters = {
        'reg_param': [0.55, 0.6],
        'tol': [0.0001]
    },
    pycaret_Dummy_parameters = {
        'constant': [None], 
        'strategy': ['prior', 'most_frequent']
    },
    pycaret_GaussianNB_parameters = {
        'priors': [None], 
        'var_smoothing': [1e-09, 5e-09, 1e-08]
    },
    pycaret_reg_LassoLars_parameters = {
        "alpha": [1.0],          # Regularization strength; higher values reduce overfitting
        "max_iter": [500],       # Maximum number of iterations
        "normalize": [True]     # Normalizes the regressors X before regression
    },
    pycaret_reg_ElasticNet_parameters = {
        "alpha": [1.0],          # Overall regularization strength
        "l1_ratio": [0.5],       # Balance between Lasso (L1) and Ridge (L2) regularization
        "max_iter": [1000]       # Maximum number of iterations
    },
    pycaret_reg_BayesianRidge_parameters = {
        "alpha_1": [1e-06],      # Hyperparameter for the Gamma distribution prior over alpha
        "alpha_2": [1e-06],      # Hyperparameter for the Gamma distribution prior over alpha
        "lambda_1": [1e-06],     # Hyperparameter for the Gamma distribution prior over lambda
        "lambda_2": [1e-06],     # Hyperparameter for the Gamma distribution prior over lambda
        "n_iter": [1000]          # Number of iterations for the optimization
    },
    pycaret_reg_Lasso_parameters = {
        "alpha": [1.0],          # Regularization strength; higher values reduce overfitting
        "max_iter": [1000]       # Maximum number of iterations
    },
    pycaret_reg_Lars_parameters = {
        "n_nonzero_coefs": [500] # Maximum number of non-zero coefficients; controls model complexity
    },
    pycaret_reg_Ridge_parameters = {
        "alpha": [1.0]           # Regularization strength; higher values reduce overfitting
    },
    pycaret_reg_LinearRegression_parameters = {
        "n_jobs": [1]            # Number of jobs to run in parallel; not directly related to overfitting
    },
    pycaret_reg_HuberRegressor_parameters = {
        "alpha": [0.0001],       # Regularization strength; higher values reduce overfitting
        "epsilon": [1.35],       # The epsilon parameter in the Huber loss function
        "max_iter": [1000]        # Maximum number of iterations
    },
    pycaret_reg_OMP_parameters = {
        "n_nonzero_coefs": [None], # Number of non-zero coefficients; can limit model complexity
        "tol": [None]              # Tolerance for the optimization; lower values can lead to simpler models
    },
    pycaret_reg_AdaBoost_parameters = {
        "n_estimators": [70],       # Number of base estimators; more estimators can improve performance but may increase overfitting
        "learning_rate": [0.01],     # Learning rate shrinks the contribution of each regressor; lower values can reduce overfitting
        "loss": ['linear']          # Loss function to use when updating the weights
    },
    pycaret_reg_GBR_parameters = {
        "n_estimators": [1000],      # Number of boosting stages to perform
        "learning_rate": [0.01],     # Learning rate shrinks the contribution of each tree
        "max_depth": [3],           # Maximum depth of the individual regression estimators
        "subsample": [1.0]          # Fraction of samples to be used for fitting the individual base learners
    },
    pycaret_reg_RandomForest_parameters = {
        "n_estimators": [100],      # Number of trees in the forest
        "max_depth": [None],        # Maximum depth of the tree; limiting can reduce overfitting
        "min_samples_split": [2],   # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1],    # Minimum number of samples required to be at a leaf node
        "max_features": ['auto']    # Number of features to consider when looking for the best split
    },
    pycaret_reg_ExtraTrees_parameters = {
        "n_estimators": [100],      # Number of trees in the forest
        "max_depth": [None],        # Maximum depth of the tree; limiting can reduce overfitting
        "min_samples_split": [2],   # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1],    # Minimum number of samples required to be at a leaf node
        "max_features": ['auto']    # Number of features to consider when looking for the best split
    },
    pycaret_reg_DecisionTree_parameters = {
        "max_depth": [None],        # Maximum depth of the tree; limiting can reduce overfitting
        "min_samples_split": [2],   # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1]     # Minimum number of samples required to be at a leaf node
    },
    pycaret_reg_LGBM_parameters = {
        "learning_rate": [0.01],     # Learning rate shrinks the contribution of each tree
        "num_leaves": [31],         # Maximum tree leaves for base learners; lower values prevent overfitting
        "max_depth": [8],          # Limits the depth of the tree; -1 means no limit
        "min_child_samples": [20],  # Minimum number of data needed in a child; larger values prevent overfitting
        "subsample": [1.0],         # Fraction of data to be used for fitting the individual base learners
        "colsample_bytree": [1.0],  # Fraction of features to be used for fitting the individual base learners
        "reg_alpha": [0.0],         # L1 regularization term on weights
        "reg_lambda": [0.0]         # L2 regularization term on weights
    },
    pycaret_reg_CatBoost_parameters = {
        "iterations": [1000],        # Number of boosting iterations
        "depth": [6],               # Depth of the trees; lower values prevent overfitting
        "learning_rate": [0.01],     # Learning rate shrinks the contribution of each tree
        "l2_leaf_reg": [3],         # L2 regularization coefficient
        "verbose": [False]              # Verbosity level; 0 suppresses output
    },
    pycaret_reg_KNeighbors_parameters = {
        "n_neighbors": [5],        # Number of neighbors to use
        "weights": ['uniform']     # Weight function used in prediction; 'uniform' assigns equal weight
    },
    pycaret_reg_PassiveAggressive_parameters = {
        "C": [1.0],                # Regularization parameter; smaller values specify stronger regularization
        "epsilon": [0.1],          # Epsilon in the epsilon-insensitive loss functions
        "max_iter": [1000],        # Maximum number of passes over the training data
        "tol": [0.001]             # Tolerance for the stopping criteria
    },
    pycaret_reg_Dummy_parameters = {
        "strategy": ['mean']       # Strategy to use when predicting; 'mean' predicts the mean of the training targets
    },
    pycaret_reg_XGB_parameters = {
       'learning_rate': [0.01],
        'max_depth': [8],
        'n_estimators': [1000],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [10],
        'reg_lambda': [100]
    },
    h2o_models = True,
    h2o_data = data_h2o,
    h2o_metric_prioritize = "AUTO",
    h2o_x_cols = ['member_rating', 'country_code', 'tag_count', 'optin_days', 'email_provider', 'tag_count_by_optin_day', 'tag_aws_webinar', 'tag_learning_lab', 'tag_learning_lab_05', 'tag_learning_lab_09', 'tag_learning_lab_11', 'tag_learning_lab_12', 'tag_learning_lab_13', 'tag_learning_lab_14', 'tag_learning_lab_15', 'tag_learning_lab_16', 'tag_learning_lab_17', 'tag_learning_lab_18', 'tag_learning_lab_19', 'tag_learning_lab_20', 'tag_learning_lab_21', 'tag_learning_lab_22', 'tag_learning_lab_23', 'tag_learning_lab_24', 'tag_learning_lab_25', 'tag_learning_lab_26', 'tag_learning_lab_27', 'tag_learning_lab_28', 'tag_learning_lab_29', 'tag_learning_lab_30', 'tag_learning_lab_31', 'tag_learning_lab_32', 'tag_learning_lab_33', 'tag_learning_lab_34', 'tag_learning_lab_35', 'tag_learning_lab_36', 'tag_learning_lab_37', 'tag_learning_lab_38', 'tag_learning_lab_39', 'tag_learning_lab_40', 'tag_learning_lab_41', 'tag_learning_lab_42', 'tag_learning_lab_43', 'tag_learning_lab_44', 'tag_learning_lab_45', 'tag_learning_lab_46', 'tag_learning_lab_47', 'tag_time_series_webinar', 'tag_webinar', 'tag_webinar_01', 'tag_webinar_no_degree', 'tag_webinar_no_degree_02']
    
    ---------------------
    
    To load models:
    
        For classification:
            For Pycaret models:
                import pycaret.classification as clf
                pycaret_model = clf.load_model("model_directory/model_name")
                
            For H2O models:
                Binary:
                    import h2o
                    import joblib
                    import os
                    from my_pandas_extensions.modeling import ML_Util

                    h2o.init()
                    loaded_model = h2o.load_model("h2o_models\StackedEnsemble_BestOfFamily_4_AutoML_1_20241009_191953")

                    output_dir_h2o = os.path.join(os.getcwd(), "h2o_models")

                    additional_model_info = joblib.load(os.path.join(output_dir_h2o, 'additional_model_info.pkl'))
                    best_threshold = additional_model_info['threshold']
                    venn_abers = additional_model_info['venn_abers']

                    class CalibratedModelClassification:
                        def __init__(self, model, calibrator):
                            self.model = model
                            self.calibrator = calibrator

                        def predict_proba(self, data):
                            if isinstance(data, pd.DataFrame):
                                data_h2o = h2o.H2OFrame(data)
                                for col in data.columns:
                                    if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                                        data_h2o[col] = data_h2o[col].asfactor()
                            elif isinstance(data, h2o.H2OFrame):
                                data_h2o = data
                            else:
                                raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

                            preds = self.model.predict(data_h2o)
                            preds_df = preds.as_data_frame()

                            if 'p0' in preds_df.columns and 'p1' in preds_df.columns:
                                p_raw = preds_df[['p0', 'p1']].values
                            else:
                                raise KeyError("Class probabilities not found in model output.")

                            calibrated_probs = self.calibrator.predict_proba(p_raw)

                            prob_lower = calibrated_probs[0]
                            prob_upper = calibrated_probs[1]
                            prob_mean = (prob_lower + prob_upper) / 2

                            return prob_mean[:, 1]

                        def predict_dataframe(self, data, include_data=True, threshold=0.5):
                            if isinstance(data, pd.DataFrame):
                                data_h2o = h2o.H2OFrame(data)
                                for col in data.columns:
                                    if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                                        data_h2o[col] = data_h2o[col].asfactor()
                            elif isinstance(data, h2o.H2OFrame):
                                data_h2o = data
                            else:
                                raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

                            preds = self.model.predict(data_h2o)
                            preds_df = preds.as_data_frame()

                            p0 = preds_df['p0'].values
                            p1 = preds_df['p1'].values
                            p_raw = np.column_stack((p0, p1))

                            calibrated_probs = self.calibrator.predict_proba(p_raw)

                            prob_lower = calibrated_probs[0]
                            prob_upper = calibrated_probs[1]
                            prob_mean = (prob_lower + prob_upper) / 2

                            prob_sum = prob_mean.sum(axis=1).reshape(-1, 1)
                            normalized_probs = prob_mean / prob_sum

                            preds_df['Score_0'] = normalized_probs[:, 0]
                            preds_df['Score_1'] = normalized_probs[:, 1]
                            preds_df['Label'] = (preds_df['Score_1'] > threshold).astype(int)

                            if include_data:
                                return pd.concat([data.reset_index(drop=True), preds_df[['Score_0', 'Score_1', 'Label']]], axis=1)
                            else:
                                return preds_df[['Score_0', 'Score_1', 'Label']]

                        def save(self, directory):
                            self.model_path = h2o.save_model(model=self.model, path=directory, force=True)
                            save_path = os.path.join(directory, "calibrated_model.pkl")
                            joblib.dump({'calibrator': self.calibrator, 'model_path': self.model_path}, save_path)
                            print(f"CalibratedModel saved to: {save_path}")

                        @classmethod
                        def load(cls, directory):
                            load_path = os.path.join(directory, "calibrated_model.pkl")
                            data = joblib.load(load_path)
                            model = h2o.load_model(data['model_path'])
                            calibrator = data['calibrator']
                            return cls(model, calibrator)

                    class CalibratedModelWithOptimizedThreshold:
                        def __init__(self, calibrated_model, threshold=0.5):
                            self.calibrated_model = calibrated_model  # calibrated model (Venn-Abers or similar)
                            self.threshold = threshold                # optimized threshold

                        def predict_proba(self, h2o_test):
                            return self.calibrated_model.predict_proba(h2o_test)
                        
                        def predict_dataframe(self, data, include_data=True):
                            preds_df = self.calibrated_model.predict_dataframe(data, include_data=include_data, threshold=self.threshold)
                            return preds_df

                        def predict(self, h2o_test):
                            y_pred_proba = self.predict_proba(h2o_test)
                            return (y_pred_proba > self.threshold).astype(int)

                        def save(self, path):
                            joblib.dump(self, path)

                        @staticmethod
                        def load(path):
                            return joblib.load(path)
                        
                    best_model_calibrated_h2o = CalibratedModelClassification(loaded_model, venn_abers)
                    final_model = CalibratedModelWithOptimizedThreshold(best_model_calibrated_h2o, threshold=best_threshold)
                        # output_dir_h2o: This should be the same directory path you used when saving the model. Ensure that it contains the following files:
                        # -calibrators.pkl
                        # -metadata.pkl
                        # -H2O model files (saved by h2o.save_model)
                
                
                Multiclass:
                    h2o.init()
                    from my_pandas_extensions.modeling import CalibratedModelClassificationMulti
                    best_model_calibrated_h2o = CalibratedModelClassificationMulti.load(output_dir_h2o)
                    # output_dir_h2o: This should be the same directory path you used when saving the model.
        
        For regression:
            For Pycaret models:
                import pycaret.regression as reg
                pycaret_model = reg.load_model("model_directory/model_name")
        
            For H2O models:
                h2o.init()
                from my_pandas_extensions.modeling import CalibratedModelRegression
                best_model_calibrated_h2o = CalibratedModelRegression.load(output_dir_h2o)
                # Ensure that the directory (output_dir_h2o) contains:
                # - calibrated_model.pkl
                # - H2O model files (saved by h2o.save_model)


    
    To create dataframe with predictions:
        
        For binary y:
            For Pycaret models:
                clf.predict_model(
                    estimator = best_model_calibrated_pycaret,
                    data = leads_df,
                    raw_score = True
                )
                
            For H2O models:
                df_predictions = best_model_calibrated_h2o.predict_dataframe(leads_df, include_data=True)
        
        For multiclass y:
            For Pycaret models:
                clf.predict_model(
                    estimator = best_model_calibrated_pycaret,
                    data = leads_df,
                    raw_score = True
                )
            
            For H2O models:
                df_predictions = best_model_calibrated_h2o.predict_dataframe(leads_df, include_data=True, threshold=0.0):
                # threshold: Minimum probability threshold for assigning a class label. If no class meets the threshold, 'Unknown' is assigned.
        
        For numeric y:
            For Pycaret models:
                reg.predict_model(
                    estimator = best_model_calibrated_pycaret,
                    data = leads_df,
                    raw_score = True
                )
            
            For H2O models:
                df_predictions = best_model_calibrated_h2o.predict_dataframe(leads_df, include_data=True)
    
    """
    
    import h2o
    from h2o.automl import H2OAutoML
    from venn_abers import VennAbers, VennAbersCalibrator
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, confusion_matrix, mean_squared_error, 
        mean_absolute_error, r2_score, balanced_accuracy_score
    )
    import os
    import pycaret.classification as clf
    import pycaret.regression as reg
    from pycaret.classification import add_metric
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.linear_model import (
        LogisticRegression, SGDClassifier, RidgeClassifier, 
        LassoLars, ElasticNet, BayesianRidge, Lasso, 
        OrthogonalMatchingPursuit, Ridge, LinearRegression, 
        HuberRegressor, PassiveAggressiveRegressor
    )
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, 
        GradientBoostingClassifier, AdaBoostClassifier
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.isotonic import IsotonicRegression
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if y_col_type == "categorical":
        

        if pycaret_models:
               
            counts = pycaret_data[y_col].value_counts().to_dict()
            counts_non_dict = pycaret_data[y_col].value_counts()
            max_count = max(counts.values())
            min_count = min(counts.values())
            total = sum(counts.values())
            minority_class = counts_non_dict.idxmin()
            num_classes = pycaret_data[y_col].nunique()
            
            class_weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
            class_weights_list = [class_weights[cls] for cls in sorted(class_weights.keys())]
            
            clf.setup(
                data = pycaret_data,
                target = y_col,
                train_size = train_size,
                preprocess = True,
                categorical_features = pycaret_categorical_features,
                handle_unknown_categorical = True,
                combine_rare_levels = False,
                ordinal_features = pycaret_ordinal_features,
                numeric_features = pycaret_numeric_features,
                fold_strategy = pycaret_fold_strategy,
                fold = fold,
                session_id = seed,
                log_experiment = False,
                experiment_name = log_experiment_name,
                use_gpu = pycaret_use_gpu_boost,
                silent = True, 
                n_jobs=1
            )
            
            def balanced_accuracy_metric(y_true, y_pred):
                return balanced_accuracy_score(y_true, y_pred)
            add_metric('Balanced_Accuracy', 'Balanced_Accuracy', balanced_accuracy_metric, greater_is_better=True)
            
            if pycaret_data[y_col].nunique() == 2:
                
                if apply_weights:  
                    list_models_pycaret = clf.compare_models(
                    sort=pycaret_metric_prioritize,
                    n_select=pycaret_n_select,
                    exclude=["gbc", "ada", "lda", "qda", "nb", "knn", "dummy", "svm", "ridge"]
                    )
                else:
                    list_models_pycaret = clf.compare_models(
                    sort=pycaret_metric_prioritize,
                    n_select=pycaret_n_select,
                    exclude = ["svm", "ridge"]
                    )

                tuned_models = []

                for model in list_models_pycaret:
                    model_name = model.__class__.__name__
                    
                    tuned_model = None

                    if model_name == 'CatBoostClassifier':
                        params_1 = pycaret_CatBoost_parameters
                        if apply_weights:
                            params_2 = {'class_weights': [class_weights_list]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'LGBMClassifier':
                        params_1 = pycaret_LGBM_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'LogisticRegression':
                        params_1 = pycaret_LogisticRegression_parameters
                        if apply_weights:
                            params_2 = {"multi_class": ['auto'], 'class_weight': [class_weights]}
                        else:
                            params_2 = {"multi_class": ['auto']}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'RandomForestClassifier':
                        params_1 = pycaret_RandomForest_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'ExtraTreesClassifier':
                        params_1 = pycaret_ExtraTrees_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'DecisionTreeClassifier':
                        params_1 = pycaret_DecisionTree_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'SGDClassifier':
                        params_1 = pycaret_SGD_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid= params_1
                        )
                    
                    elif model_name == 'RidgeClassifier':
                        params_1 = pycaret_Ridge_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize = pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'GradientBoostingClassifier':
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = pycaret_GBC_parameters
                        )
                    
                    elif model_name == 'AdaBoostClassifier':
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = pycaret_ADA_parameters
                        )
                    
                    elif model_name == 'LinearDiscriminantAnalysis':
                        tuned_model = clf.tune_model(
                            model,
                            optimize= pycaret_metric_prioritize,
                            custom_grid= pycaret_LinearDiscriminant_parameters
                        )
                    
                    elif model_name == 'KNeighborsClassifier':
                        tuned_model = clf.tune_model(
                            model,
                            optimize= pycaret_metric_prioritize,
                            custom_grid = pycaret_KNeighbors_parameters
                        )
                    
                    elif model_name == 'QuadraticDiscriminantAnalysis':
                        tuned_model = clf.tune_model(
                            model,
                            optimize = pycaret_metric_prioritize,
                            custom_grid = pycaret_QuadraticDiscriminant_parameters
                        )
                    
                    elif model_name == 'DummyClassifier':
                        params_1 = pycaret_SGD_parameters
                        params_2 = {'random_state': [seed]}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize= pycaret_metric_prioritize,
                            custom_grid = pycaret_Dummy_parameters
                        )
                        
                    elif model_name == 'GaussianNB':
                        tuned_model = clf.tune_model(
                            model,
                            optimize= pycaret_metric_prioritize,
                            custom_grid = pycaret_GaussianNB_parameters
                        )
                            
                    tuned_models.append(tuned_model)

                
                output_dir_pycaret = os.path.join(os.getcwd(), "pycaret_models")
                if not os.path.exists(output_dir_pycaret):
                    os.makedirs(output_dir_pycaret)
                
                for i, model in enumerate(tuned_models):
                    learning_curve_path = os.path.join(output_dir_pycaret, f"model_{model.__class__.__name__}_learning_curve.png")
                    clf.plot_model(model, plot="learning", save=True)
                    os.rename("Learning Curve.png", learning_curve_path)

                    predictions = clf.predict_model(model)
                    predictions = clf.pull()
                    if 'AUC' in predictions.columns:
                        auc_score = predictions['AUC'][0]
                    elif 'Label' in predictions.columns and 'Score' in predictions.columns:
                        auc_score = roc_auc_score(predictions['Label'], predictions['Score'])
                    else:
                        auc_score = 0
                        
                    if auc_score > 0:
                        auc_curve_path = os.path.join(output_dir_pycaret, f"model_{model.__class__.__name__}_auc.png")
                        clf.plot_model(model, plot="auc", save=True)
                        os.rename("AUC.png", auc_curve_path)
                    
                    confusion_matrix_path = os.path.join(output_dir_pycaret, f"model_{model.__class__.__name__}_confusion_matrix.png")
                    clf.plot_model(model, plot="confusion_matrix", plot_kwargs={"percent": True}, save=True)
                    os.rename("Confusion Matrix.png", confusion_matrix_path)
                
                metrics_list = []
                
                for model in tuned_models:
                    model_name = model.__class__.__name__
                    clf.predict_model(model, verbose=False)
                    metrics_df = clf.pull()
                    metrics_df['Model'] = model_name
                    metrics_list.append(metrics_df)

                tuned_models_grid = pd.concat(metrics_list, ignore_index=True)

                cols = ['Model'] + [col for col in tuned_models_grid.columns if col != 'Model']
                tuned_models_grid = tuned_models_grid[cols]
                
                if pycaret_metric_prioritize == "Precision" or pycaret_metric_prioritize == "precision":
                    tuned_models_grid = tuned_models_grid \
                        .sort_values(by = "Prec.", ascending = False)
                else:
                    tuned_models_grid = tuned_models_grid \
                        .sort_values(by = pycaret_metric_prioritize, ascending = False)
                
                first_model_name = tuned_models_grid.iloc[0]['Model']
                
                # LIST - Best
                best_of_list_tuned_pycaret = None
                for model in tuned_models:
                    if model.__class__.__name__ == first_model_name:
                        best_of_list_tuned_pycaret = model
                        break
                
                best_of_list_tuned_pycaret_grid = tuned_models_grid.iloc[[0]]
                
                if apply_weights:
                    best_of_list_tuned_pycaret_grid["Model"] = best_of_list_tuned_pycaret_grid["Model"].str.replace(f"{first_model_name}", f"{first_model_name} (Weighted)")
                else:
                    best_of_list_tuned_pycaret_grid["Model"] = best_of_list_tuned_pycaret_grid["Model"].str.replace(f"{first_model_name}", f"{first_model_name}")
                
                # Individual Model - XGB
                if pycaret_use_CUDA is False:
                    xgb_model_pycaret = clf.create_model(
                        estimator = "xgboost",
                        tree_method = "hist",
                        n_jobs=-1
                    )
                else:
                    xgb_model_pycaret = clf.create_model(
                        estimator = "xgboost",
                        tree_method = "gpu_hist",
                        n_jobs=-1
                    )
                
                y_train_xgb = clf.get_config('y_train')
                unique_classes, class_counts = np.unique(y_train_xgb, return_counts=True)
                class_weights_xgb = len(y_train_xgb) / (len(unique_classes) * class_counts)
                class_weight_dict = dict(zip(unique_classes, class_weights_xgb))
                
                if len(unique_classes) == 2:
                    scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]
                else:
                    scale_pos_weight = None
                
                params_1 = pycaret_XGB_parameters
                if apply_weights:
                    params_2 = {'scale_pos_weight': [scale_pos_weight] if scale_pos_weight is not None else None}
                else:
                    params_2 = {}
                params_1.update(params_2)
                
                xgb_model_tuned_pycaret = clf.tune_model(
                    estimator = xgb_model_pycaret,
                    n_iter = pycaret_n_iter,
                    optimize = pycaret_metric_prioritize,
                    custom_grid = params_1
                )

                
                xgb_model_name = xgb_model_tuned_pycaret.__class__.__name__
                
                clf.predict_model(xgb_model_tuned_pycaret, verbose=False)
                XGB_tuned_pycaret_grid = clf.pull()
                
                if apply_weights:
                    XGB_tuned_pycaret_grid['Model'] = XGB_tuned_pycaret_grid['Model'].str.replace("Extreme Gradient Boosting", f"{xgb_model_name} (Weighted)")
                else:
                    XGB_tuned_pycaret_grid['Model'] = XGB_tuned_pycaret_grid['Model'].str.replace("Extreme Gradient Boosting", f"{xgb_model_name}")

                learning_curve_path = os.path.join(output_dir_pycaret, "xgb_model_learning_curve.png")
                clf.plot_model(xgb_model_tuned_pycaret, plot="learning", save=True)
                os.rename("Learning Curve.png", learning_curve_path)

                auc_curve_path = os.path.join(output_dir_pycaret, "xgb_model_auc.png")
                clf.plot_model(xgb_model_tuned_pycaret, plot="auc", save=True)
                os.rename("AUC.png", auc_curve_path)

                confusion_matrix_path = os.path.join(output_dir_pycaret, "xgb_model_confusion_matrix.png")
                clf.plot_model(xgb_model_tuned_pycaret, plot="confusion_matrix", plot_kwargs={"percent": True}, save=True)
                os.rename("Confusion Matrix.png", confusion_matrix_path)
                
                # LIST - Blended models
                list_models_final_pycaret = list(tuned_models) + [xgb_model_tuned_pycaret]
                
                blended_models_tuned_pycaret = clf.blend_models(
                    list_models_final_pycaret,
                    optimize = pycaret_metric_prioritize    
                )
                                
                clf.predict_model(blended_models_tuned_pycaret, verbose=False)
                blended_models_tuned_pycaret_grid = clf.pull()
                
                if apply_weights:
                    blended_models_tuned_pycaret_grid['Model'] = blended_models_tuned_pycaret_grid['Model'].str.replace("Voting Classifier", f"Pycaret Models Blended ({' + '.join([str(model.__class__.__name__) for model in tuned_models])}) (Weighted)")
                else:
                    blended_models_tuned_pycaret_grid['Model'] = blended_models_tuned_pycaret_grid['Model'].str.replace("Voting Classifier", f"Pycaret Models Blended ({' + '.join([str(model.__class__.__name__) for model in tuned_models])})")
                
                blended_learning_curve_path = os.path.join(output_dir_pycaret, "blended_model_learning_curve.png")
                clf.plot_model(blended_models_tuned_pycaret, plot="learning", save=True)
                os.rename("Learning Curve.png", blended_learning_curve_path)

                predictions = clf.predict_model(blended_models_tuned_pycaret)
                predictions = clf.pull()
                if 'AUC' in predictions.columns:
                    auc_score = predictions['AUC'][0]
                elif 'Label' in predictions.columns and 'Score' in predictions.columns:
                    auc_score = roc_auc_score(predictions['Label'], predictions['Score'])
                else:
                    auc_score = 0
                    
                if auc_score > 0:
                    blended_auc_curve_path = os.path.join(output_dir_pycaret, "blended_model_auc.png")
                    clf.plot_model(blended_models_tuned_pycaret, plot="auc", save=True)
                    os.rename("AUC.png", blended_auc_curve_path)


                blended_confusion_matrix_path = os.path.join(output_dir_pycaret, "blended_model_confusion_matrix.png")
                clf.plot_model(blended_models_tuned_pycaret, plot="confusion_matrix", plot_kwargs={"percent": True}, save=True)
                os.rename("Confusion Matrix.png", blended_confusion_matrix_path)

                # FINAL
                
                pycaret_models_ = [best_of_list_tuned_pycaret] + [xgb_model_tuned_pycaret] + [blended_models_tuned_pycaret]

                if pycaret_metric_prioritize == "Precision" or pycaret_metric_prioritize == "precision":
                    pycaret_grid = pd.concat([blended_models_tuned_pycaret_grid, XGB_tuned_pycaret_grid, best_of_list_tuned_pycaret_grid]) \
                        .sort_values(by = "Prec.", ascending = False) \
                        .reset_index() \
                        .drop("index", axis = 1)
                else:
                    pycaret_grid = pd.concat([blended_models_tuned_pycaret_grid, XGB_tuned_pycaret_grid, best_of_list_tuned_pycaret_grid]) \
                        .sort_values(by = pycaret_metric_prioritize, ascending = False) \
                        .reset_index() \
                        .drop("index", axis = 1)
                
                pycaret_csv_file_path = os.path.join(output_dir_pycaret, "pycaret_ml_grid.csv")
                pycaret_grid.to_csv(pycaret_csv_file_path)
                                
                pycaret_models_scores = {}

                for model in pycaret_models_:
                    predictions_ = clf.predict_model(model)
                    metrics_df = clf.pull()
                    
                    if pycaret_metric_prioritize == "Precision" or pycaret_metric_prioritize == "precision":
                        score = metrics_df["Prec."].iloc[0]
                    else:
                        score = metrics_df[pycaret_metric_prioritize].iloc[0]
                    
                    pycaret_models_scores[model.__class__.__name__] = score

                best_model_name = max(pycaret_models_scores, key=pycaret_models_scores.get)
                
                best_model_pycaret = None
                for model in pycaret_models_:
                    if model.__class__.__name__ == best_model_name:
                        best_model_pycaret = model
                    else:
                        pass
                
                X_train = clf.get_config('X_train')
                y_train = clf.get_config('y_train')

                X_test = clf.get_config('X_test')
                y_test = clf.get_config('y_test')
                
                    ### Venn-ABERS calibration

                best_model_calibrated_pycaret = OriginalAndCalibratedWrapper(
                    base_estimator=best_model_pycaret, 
                    inductive=False, 
                    n_splits=fold,
                    random_state=seed
                )
                
                best_model_calibrated_pycaret.fit(X_train, y_train)
                
                if force_balanced_metric:
                    best_model_calibrated_pycaret = clf.optimize_threshold(best_model_calibrated_pycaret, optimize = 'Balanced_Accuracy', grid_interval = pycaret_threshold_grid_interval)
                else:
                    best_model_calibrated_pycaret = clf.optimize_threshold(best_model_calibrated_pycaret, optimize = pycaret_metric_prioritize, grid_interval = pycaret_threshold_grid_interval)

                
                best_model_calibrated_pycaret = clf.finalize_model(best_model_calibrated_pycaret)
                
                
                clf.predict_model(best_model_calibrated_pycaret)
                best_model_calibrated_metrics_pycaret = clf.pull()
                
                predictions = clf.predict_model(best_model_calibrated_pycaret)
                predictions = clf.pull()
                if 'AUC' in predictions.columns:
                    auc_score = predictions['AUC'][0]
                elif 'Label' in predictions.columns and 'Score' in predictions.columns:
                    auc_score = roc_auc_score(predictions['Label'], predictions['Score'])
                else:
                    auc_score = 0

                if auc_score > 0:
                    calibration_auc_curve_path = os.path.join(output_dir_pycaret, f"model_{best_model_name}(Calibrated_VennABERS)_AUC.png")
                    clf.plot_model(best_model_calibrated_pycaret, plot="auc", save=True)
                    os.rename("AUC.png", calibration_auc_curve_path)
                
                calibration_curve_path = os.path.join(output_dir_pycaret, f"model_{best_model_name}(Calibrated_VennABERS)_calibration_curve.png")
                clf.plot_model(best_model_calibrated_pycaret, plot="calibration", save=True)
                os.rename("Calibration Curve.png", calibration_curve_path)

                calibration_confusion_matrix_path = os.path.join(output_dir_pycaret, f"model_{best_model_name}(Calibrated_VennABERS)_confusion_matrix.png")
                clf.plot_model(best_model_calibrated_pycaret, plot="confusion_matrix", plot_kwargs={"percent": True}, save=True)
                os.rename("Confusion Matrix.png", calibration_confusion_matrix_path)
                
                clf.save_model(
                    model = best_model_calibrated_pycaret,
                    model_name = os.path.join(output_dir_pycaret, f"{best_model_name}_Calibrated_VennABERS_pycaret")
                )
                
                try:
                    best_model_calibrated_metrics_pycaret["Model"] = best_model_calibrated_pycaret.classifier.__class__.__name__
                except:
                    best_model_calibrated_metrics_pycaret["Model"] = best_model_calibrated_pycaret.base_estimator.__class__.__name__
                
                print("Best Model (Pycaret):")
                print(best_model_calibrated_metrics_pycaret)
                print(f"Best threshold for Pycaret Model: {best_model_calibrated_pycaret.probability_threshold}")
                
                best_model_calibrated_metrics_path = os.path.join(output_dir_pycaret, "pycaret_best_model_grid.csv")
                best_model_calibrated_metrics_pycaret.to_csv(best_model_calibrated_metrics_path, index=False)
                
                
            if pycaret_data[y_col].nunique() > 2:
                
                if apply_weights:  
                    list_models_pycaret = clf.compare_models(
                    sort=pycaret_metric_prioritize,
                    n_select=pycaret_n_select,
                    exclude=["gbc", "ada", "lda", "qda", "nb", "knn", "dummy", "svm", "ridge"]
                    )
                else:
                    list_models_pycaret = clf.compare_models(
                    sort=pycaret_metric_prioritize,
                    n_select=pycaret_n_select,
                    exclude = ["svm", "ridge"]
                    )

                tuned_models = []

                for model in list_models_pycaret:
                    model_name = model.__class__.__name__
                    
                    if model_name == 'CatBoostClassifier':
                        params_1 = pycaret_CatBoost_parameters
                        if apply_weights:
                            params_2 = {"bootstrap_type": ['Bernoulli'], 'class_weights': [class_weights_list]}
                        else:
                            params_2 = {"bootstrap_type": ['Bernoulli']}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'LGBMClassifier':
                        params_1 = pycaret_LGBM_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                        
                    elif model_name == 'GradientBoostingClassifier':
                        params_1 = pycaret_GBC_parameters
                        if apply_weights:
                            params_2 = {}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = pycaret_GBC_parameters
                        )
                    
                    elif model_name == 'LogisticRegression':
                        params_1 = pycaret_LogisticRegression_parameters
                        if apply_weights:
                            params_2 = {"multi_class": ['multinomial'], 'class_weight': [class_weights]}
                        else:
                            params_2 = {"multi_class": ['multinomial']}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'RandomForestClassifier':
                        params_1 = pycaret_RandomForest_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'ExtraTreesClassifier':
                        params_1 = pycaret_ExtraTrees_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'DecisionTreeClassifier':
                        params_1 = pycaret_DecisionTree_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'SGDClassifier':
                        params_1 = pycaret_SGD_parameters
                        if apply_weights:
                            params_2 = {'loss': ['log'], 'penalty': ['elasticnet'], 'class_weight': [class_weights]}
                        else:
                            params_2 = {'loss': ['log'], 'penalty': ['elasticnet']}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'RidgeClassifier':
                        params_1 = pycaret_Ridge_parameters
                        if apply_weights:
                            params_2 = {'class_weight': [class_weights]}
                        else:
                            params_2 = {}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = params_1
                        )
                    
                    elif model_name == 'GradientBoostingClassifier':
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = pycaret_GBC_parameters
                        )
                    
                    elif model_name == 'AdaBoostClassifier':
                        tuned_model = clf.tune_model(
                            model,
                            optimize=pycaret_metric_prioritize,
                            custom_grid = pycaret_ADA_parameters
                        )
                    
                    elif model_name == 'LinearDiscriminantAnalysis':
                        tuned_model = clf.tune_model(
                            model,
                            optimize= pycaret_metric_prioritize,
                            custom_grid= pycaret_LinearDiscriminant_parameters
                        )
                    
                    elif model_name == 'KNeighborsClassifier':
                        tuned_model = clf.tune_model(
                            model,
                            optimize= pycaret_metric_prioritize,
                            custom_grid = pycaret_KNeighbors_parameters
                        )
                    
                    elif model_name == 'QuadraticDiscriminantAnalysis':
                        tuned_model = clf.tune_model(
                            model,
                            optimize = pycaret_metric_prioritize,
                            custom_grid = pycaret_QuadraticDiscriminant_parameters
                        )
                    
                    elif model_name == 'DummyClassifier':
                        params_1 = pycaret_SGD_parameters
                        params_2 = {'random_state': [seed]}
                        params_1.update(params_2)
                        tuned_model = clf.tune_model(
                            model,
                            optimize= pycaret_metric_prioritize,
                            custom_grid = pycaret_Dummy_parameters
                        )
                        
                    elif model_name == 'GaussianNB':
                        tuned_model = clf.tune_model(
                            model,
                            optimize= pycaret_metric_prioritize,
                            custom_grid = pycaret_GaussianNB_parameters
                        )
                    
                    tuned_models.append(tuned_model)

                
                output_dir_pycaret = os.path.join(os.getcwd(), "pycaret_models")
                if not os.path.exists(output_dir_pycaret):
                    os.makedirs(output_dir_pycaret)
                
                for i, model in enumerate(tuned_models):
                    learning_curve_path = os.path.join(output_dir_pycaret, f"model_{model.__class__.__name__}_learning_curve.png")
                    clf.plot_model(model, plot="learning", save=True)
                    os.rename("Learning Curve.png", learning_curve_path)
                    
                    confusion_matrix_path = os.path.join(output_dir_pycaret, f"model_{model.__class__.__name__}_confusion_matrix.png")
                    clf.plot_model(model, plot="confusion_matrix", plot_kwargs={"percent": True}, save=True)
                    os.rename("Confusion Matrix.png", confusion_matrix_path)
                
                metrics_list = []
                
                for model in tuned_models:
                    model_name = model.__class__.__name__
                    clf.predict_model(model, verbose=False)
                    metrics_df = clf.pull()
                    metrics_df['Model'] = model_name
                    metrics_list.append(metrics_df)

                tuned_models_grid = pd.concat(metrics_list, ignore_index=True)

                cols = ['Model'] + [col for col in tuned_models_grid.columns if col != 'Model']
                tuned_models_grid = tuned_models_grid[cols]
                
                if pycaret_metric_prioritize == "Precision" or pycaret_metric_prioritize == "precision":
                    tuned_models_grid = tuned_models_grid \
                        .sort_values(by = "Prec.", ascending = False)
                else:
                    tuned_models_grid = tuned_models_grid \
                        .sort_values(by = pycaret_metric_prioritize, ascending = False)
                
                first_model_name = tuned_models_grid.iloc[0]['Model']
                
                # LIST - Best
                best_of_list_tuned_pycaret = None
                for model in tuned_models:
                    if model.__class__.__name__ == first_model_name:
                        best_of_list_tuned_pycaret = model
                        break
                
                best_of_list_tuned_pycaret_grid = tuned_models_grid.iloc[[0]].reset_index().drop("index", axis = 1)
                best_of_list_tuned_pycaret_grid["Model"] = best_of_list_tuned_pycaret_grid["Model"].str.replace(f"{first_model_name}", f"{first_model_name} (Weighted)")
                
                
                # Individual Model - XGB
                
                y_train = clf.get_config('y_train')

                def create_balanced_sample_weights(y_train):
                    classes = np.unique(y_train)
                    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
                    sample_weights = np.array([class_weight_dict[y] for y in y_train])
                    return sample_weights

                weight_ = create_balanced_sample_weights(y_train)
                
                tree_method = "gpu_hist" if pycaret_use_CUDA else "hist"
                xgb_model_pycaret = clf.create_model(
                    estimator="xgboost",
                    tree_method=tree_method,
                    n_jobs=-1
                )

                params_1 = pycaret_XGB_parameters
                if apply_weights:
                    params_2 = {'objective': ['multi:softprob'], 'num_class': [num_classes], 'eval_metric': ['mlogloss'], "weights": [weight_]}
                else:
                    params_2 = {'objective': ['multi:softprob'], 'num_class': [num_classes], 'eval_metric': ['mlogloss']}
                params_1.update(params_2)

                if apply_weights:
                    xgb_model_tuned_pycaret = clf.tune_model(
                        estimator=xgb_model_pycaret,
                        n_iter=pycaret_n_iter,
                        optimize=pycaret_metric_prioritize,
                        custom_grid=params_1,
                        fit_kwargs={'sample_weight': weight_}
                    )
                else:
                    xgb_model_tuned_pycaret = clf.tune_model(
                        estimator=xgb_model_pycaret,
                        n_iter=pycaret_n_iter,
                        optimize=pycaret_metric_prioritize,
                        custom_grid=params_1
                    )
                
                xgb_model_name = xgb_model_tuned_pycaret.__class__.__name__
                
                clf.predict_model(xgb_model_tuned_pycaret, verbose=False)
                XGB_tuned_pycaret_grid = clf.pull()
                
                if apply_weights:
                    XGB_tuned_pycaret_grid['Model'] = XGB_tuned_pycaret_grid['Model'].str.replace("Extreme Gradient Boosting", f"{xgb_model_name} (Weighted)")
                else:
                    XGB_tuned_pycaret_grid['Model'] = XGB_tuned_pycaret_grid['Model'].str.replace("Extreme Gradient Boosting", f"{xgb_model_name}")
                    
                learning_curve_path = os.path.join(output_dir_pycaret, "xgb_model_learning_curve.png")
                clf.plot_model(xgb_model_tuned_pycaret, plot="learning", save=True)
                os.rename("Learning Curve.png", learning_curve_path)

                confusion_matrix_path = os.path.join(output_dir_pycaret, "xgb_model_confusion_matrix.png")
                clf.plot_model(xgb_model_tuned_pycaret, plot="confusion_matrix", plot_kwargs={"percent": True}, save=True)
                os.rename("Confusion Matrix.png", confusion_matrix_path)
                
                # LIST - Blended models
                list_models_final_pycaret = list(tuned_models) + [xgb_model_tuned_pycaret]
                
                blended_models_tuned_pycaret = clf.blend_models(
                    list_models_final_pycaret,
                    optimize = pycaret_metric_prioritize    
                )
                                
                clf.predict_model(blended_models_tuned_pycaret, verbose=False)
                blended_models_tuned_pycaret_grid = clf.pull()
                
                blended_models_tuned_pycaret_grid['Model'] = blended_models_tuned_pycaret_grid['Model'].str.replace("Voting Classifier", f"Pycaret Models Blended ({' + '.join([str(model.__class__.__name__) for model in tuned_models])}) (Weighted)")
                
                
                blended_learning_curve_path = os.path.join(output_dir_pycaret, "blended_model_learning_curve.png")
                clf.plot_model(blended_models_tuned_pycaret, plot="learning", save=True)
                os.rename("Learning Curve.png", blended_learning_curve_path)

                blended_confusion_matrix_path = os.path.join(output_dir_pycaret, "blended_model_confusion_matrix.png")
                clf.plot_model(blended_models_tuned_pycaret, plot="confusion_matrix", plot_kwargs={"percent": True}, save=True)
                os.rename("Confusion Matrix.png", blended_confusion_matrix_path)

                # FINAL
                
                pycaret_models_ = [best_of_list_tuned_pycaret] + [xgb_model_tuned_pycaret] + [blended_models_tuned_pycaret]

                if pycaret_metric_prioritize == "Precision" or pycaret_metric_prioritize == "precision":
                    pycaret_grid = pd.concat([blended_models_tuned_pycaret_grid, XGB_tuned_pycaret_grid, best_of_list_tuned_pycaret_grid]) \
                        .sort_values(by = "Prec.", ascending = False) \
                        .reset_index() \
                        .drop("index", axis = 1)
                else:
                    pycaret_grid = pd.concat([blended_models_tuned_pycaret_grid, XGB_tuned_pycaret_grid, best_of_list_tuned_pycaret_grid]) \
                        .sort_values(by = pycaret_metric_prioritize, ascending = False) \
                        .reset_index() \
                        .drop("index", axis = 1)
                
                pycaret_csv_file_path = os.path.join(output_dir_pycaret, "pycaret_ml_grid.csv")
                pycaret_grid.to_csv(pycaret_csv_file_path)
                                
                pycaret_models_scores = {}

                for model in pycaret_models_:
                    predictions_ = clf.predict_model(model)
                    metrics_df = clf.pull()
                    
                    if pycaret_metric_prioritize == "Precision" or pycaret_metric_prioritize == "precision":
                        score = metrics_df["Prec."].iloc[0]
                    else:
                        score = metrics_df[pycaret_metric_prioritize].iloc[0]
                    
                    pycaret_models_scores[model.__class__.__name__] = score

                best_model_name = max(pycaret_models_scores, key=pycaret_models_scores.get)
                
                best_model_pycaret = None
                for model in pycaret_models_:
                    if model.__class__.__name__ == best_model_name:
                        best_model_pycaret = model
                    else:
                        pass
                
                best_model_calibrated_pycaret = clf.calibrate_model(method = "isotonic")

                best_model_calibrated_pycaret = clf.finalize_model(best_model_calibrated_pycaret)


                clf.predict_model(best_model_calibrated_pycaret)
                best_model_calibrated_metrics_pycaret = clf.pull()

                calibration_confusion_matrix_path = os.path.join(output_dir_pycaret, f"model_{best_model_name}(Isotonic)_confusion_matrix.png")
                clf.plot_model(best_model_calibrated_pycaret, plot="confusion_matrix", plot_kwargs={"percent": True}, save=True)
                os.rename("Confusion Matrix.png", calibration_confusion_matrix_path)
                
                clf.save_model(
                    model = best_model_calibrated_pycaret,
                    model_name = os.path.join(output_dir_pycaret, f"{best_model_name}_Isotonic_pycaret")
                )
                
                print("Best Model (Pycaret):")
                print(best_model_calibrated_metrics_pycaret)
                
                best_model_calibrated_metrics_path = os.path.join(output_dir_pycaret, "pycaret_best_model_grid.csv")
                best_model_calibrated_metrics_pycaret.to_csv(best_model_calibrated_metrics_path, index=False)
                

            
        if h2o_models:
            
            h2o.init()

            if h2o_data[y_col].nunique() == 2:
                
                def get_metric_function(metric_name):
                    metric_dict = {
                        'accuracy': accuracy_score,
                        'precision': precision_score,
                        'recall': recall_score,
                        'f1': f1_score,
                        'auc': roc_auc_score,
                        'balanced_accuracy': balanced_accuracy_score
                    }
                    return metric_dict.get(metric_name, roc_auc_score)

                def optimize_threshold(y_true, y_pred_proba, metric_function):
                    best_threshold = 0.5
                    best_metric_value = -1

                    thresholds = np.arange(0.0, 1.01, h2o_threshold_grid_interval)
                    
                    for threshold in thresholds:
                        y_pred = (y_pred_proba > threshold).astype(int)
                        
                        if metric_function == roc_auc_score:
                            metric_value = metric_function(y_true, y_pred_proba)
                        else:
                            metric_value = metric_function(y_true, y_pred)
                        
                        if metric_value > best_metric_value:
                            best_metric_value = metric_value
                            best_threshold = threshold
                    
                    return best_threshold, best_metric_value
                            
                df_h2o = h2o.H2OFrame(h2o_data)
                df_h2o[y_col] = df_h2o[y_col].asfactor()

                if apply_weights:
                    counts = h2o_data[y_col].value_counts().to_dict()
                    total = sum(counts.values())
                    class_weights = {str(cls): total / (len(counts) * count) for cls, count in counts.items()}
                    levels = df_h2o[y_col].levels()[0]
                    class_sampling_factors = [class_weights[cls] for cls in levels]
                else:
                    class_sampling_factors = None


                h2o_train, h2o_test = df_h2o.split_frame(ratios=[train_size], seed=seed)
                
                train_ratio = 1 - calibration_ratio
                h2o_train_actual, h2o_calibration = h2o_train.split_frame(ratios=[train_ratio], seed=seed)

                if apply_weights:
                    aml = H2OAutoML(
                        nfolds=fold,
                        exclude_algos=["DeepLearning"],
                        seed=seed,
                        balance_classes=True,
                        class_sampling_factors=class_sampling_factors,
                        stopping_metric=h2o_metric_prioritize,
                        stopping_tolerance=0.007946274100756871, 
                        stopping_rounds=3,
                        max_models=h2o_max_models
                    )
                else:
                    aml = H2OAutoML(
                        nfolds=fold,
                        exclude_algos=["DeepLearning"],
                        seed=seed,
                        stopping_metric=h2o_metric_prioritize,
                        stopping_tolerance=0.007946274100756871, 
                        stopping_rounds=3,
                        max_models=h2o_max_models
                    )

                aml.train(
                    x=h2o_x_cols,
                    y=y_col,
                    training_frame=h2o_train_actual
                )

                best_model_h2o = aml.leader
                
                # Calibrate model using Venn-Abers
                calibration_preds = best_model_h2o.predict(h2o_calibration)
                calibration_df = h2o_calibration.as_data_frame()
                calibration_preds_df = calibration_preds.as_data_frame()

                y_true_calib = calibration_df[y_col].astype(int).values
                y_pred_calib = calibration_preds_df['p1'].values
                
                va = VennAbers()
                
                p0_calib = calibration_preds_df['p0'].values
                p1_calib = calibration_preds_df['p1'].values

                p_cal = np.column_stack((p0_calib, p1_calib))

                va.fit(p_cal, y_true_calib)
                
                output_dir_h2o = os.path.join(os.getcwd(), "h2o_models")
                if not os.path.exists(output_dir_h2o):
                    os.makedirs(output_dir_h2o)
                    
                h2o_csv_file_path = os.path.join(output_dir_h2o, "h2o_ml_grid.csv")
                aml.get_leaderboard().as_data_frame().to_csv(h2o_csv_file_path)

                best_model_calibrated_h2o = CalibratedModelClassification(best_model_h2o, va)
                
                test_frame = h2o_test.as_data_frame()
                y_true = test_frame[y_col].astype(int).values

                y_pred_proba = best_model_calibrated_h2o.predict_proba(h2o_test)

                if force_balanced_metric:
                    metric_function = get_metric_function('balanced_accuracy')
                else:
                    metric_function = get_metric_function(h2o_optimize_threshold_metric)

                best_threshold, best_metric_value = optimize_threshold(y_true, y_pred_proba, metric_function)

                if force_balanced_metric:
                    print(f"Best threshold: {best_threshold}, Best balanced_accuracy: {best_metric_value}")
                else:
                    print(f"Best threshold: {best_threshold}, Best {h2o_optimize_threshold_metric}: {best_metric_value}")



                final_model = CalibratedModelWithOptimizedThreshold(best_model_calibrated_h2o, threshold=best_threshold)


                y_pred = final_model.predict(h2o_test)

                cm = confusion_matrix(y_true, y_pred)
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')
                auc = roc_auc_score(y_true, y_pred_proba)
                balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

                class_precision = precision_score(y_true, y_pred, average=None)
                class_recall = recall_score(y_true, y_pred, average=None)
                class_f1 = f1_score(y_true, y_pred, average=None)

                metrics = {
                    "Accuracy": accuracy,
                    "Weighted Precision": precision,
                    "Weighted Recall": recall,
                    "Weighted F1 Score": f1,
                    "AUC": auc,
                    "Balanced Accuracy": balanced_accuracy
                }

                for i, (p, r, f) in enumerate(zip(class_precision, class_recall, class_f1)):
                    metrics[f"Class {i} Precision"] = p
                    metrics[f"Class {i} Recall"] = r
                    metrics[f"Class {i} F1 Score"] = f

                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                metrics_df.index.name = 'Metric'
                metrics_df.to_csv('h2o_models/calibrated_h2o_model_metrics.csv')

                print(metrics)
                print(cm)
            
            # Save H2O model using H2O's save_model function
            model_path = h2o.save_model(model=best_model_calibrated_h2o.model, path=output_dir_h2o, force=True)

            # Save additional Python objects using joblib (for example, the threshold and the Venn-Abers calibration)
            additional_model_info = {
                'threshold': best_threshold,
                'venn_abers': va,
                'model_path': model_path
            }

            joblib.dump(additional_model_info, os.path.join(output_dir_h2o, 'additional_model_info.pkl'))

            # Loading the saved model for later use
            # Initialize H2O
            # h2o.init()

            # # Load the H2O model
            # loaded_model = h2o.load_model(model_path)

            # # Load additional Python objects
            # additional_model_info = joblib.load(os.path.join(output_dir_h2o, 'additional_model_info.pkl'))
            # best_threshold = additional_model_info['threshold']
            # venn_abers = additional_model_info['venn_abers']

            # # Recreate your CalibratedModelWithOptimizedThreshold if needed
            # best_model_calibrated_h2o = CalibratedModelClassification(loaded_model, venn_abers)
            # final_model = CalibratedModelWithOptimizedThreshold(best_model_calibrated_h2o, threshold=best_threshold)
                
            if h2o_data[y_col].nunique() > 2:
                
                df_h2o = h2o.H2OFrame(h2o_data)
                df_h2o[y_col] = df_h2o[y_col].asfactor()

                counts = h2o_data[y_col].value_counts().to_dict()
                counts_non_dict = h2o_data[y_col].value_counts()
                max_count = max(counts.values())
                min_count = min(counts.values())
                imbalance_ratio = max_count / min_count
                total = sum(counts.values())
                minority_class = counts_non_dict.idxmin()
                minority_class_percentage = (counts_non_dict[minority_class] / counts_non_dict.sum()) * 100

                num_classes = h2o_data[y_col].nunique()
                class_weights = {str(cls): total / (num_classes * count) for cls, count in counts.items()}
                levels = df_h2o[y_col].levels()[0]
                class_sampling_factors = [class_weights[cls] for cls in levels]

                h2o_train, h2o_test = df_h2o.split_frame(ratios=[train_size], seed=seed)
                train_ratio = 1 - calibration_ratio
                h2o_train_actual, h2o_calibration = h2o_train.split_frame(ratios=[train_ratio], seed=seed)

                if apply_weights:
                    aml = H2OAutoML(
                        nfolds=fold,
                        exclude_algos=["DeepLearning"],
                        seed=seed,
                        balance_classes=True,
                        class_sampling_factors=class_sampling_factors,
                        stopping_metric=h2o_metric_prioritize,
                        stopping_tolerance=0.007946274100756871,
                        stopping_rounds=3,
                        max_models=h2o_max_models
                    )
                else:
                    aml = H2OAutoML(
                        nfolds=fold,
                        exclude_algos=["DeepLearning"],
                        seed=seed,
                        stopping_metric=h2o_metric_prioritize,
                        stopping_tolerance=0.007946274100756871,
                        stopping_rounds=3,
                        max_models=h2o_max_models
                    )

                aml.train(
                    x=h2o_x_cols,
                    y=y_col,
                    training_frame=h2o_train_actual
                )

                best_model_h2o = aml.leader
                
                if force_balanced_metric:
                    leaderboard = aml.leaderboard.head(rows=5)

                    leaderboard_df = leaderboard.as_data_frame()

                    balanced_accuracies = []

                    for model_id in leaderboard_df['model_id']:
                        model = h2o.get_model(model_id)
                        
                        predictions = model.predict(h2o_test)

                        predictions_df = predictions['predict'].as_data_frame()
                        y_pred = predictions_df['predict'].values
                        
                        test_frame = h2o_test.as_data_frame()
                        y_true = test_frame[y_col].astype(int).values
                        
                        balanced_acc = balanced_accuracy_score(y_true, y_pred)
                        
                        balanced_accuracies.append((model_id, balanced_acc))
                    
                    balanced_acc_df = pd.DataFrame(balanced_accuracies, columns=['model_id', 'balanced_accuracy'])
                    best_model_id = balanced_acc_df.loc[balanced_acc_df['balanced_accuracy'].idxmax(), 'model_id']
                    best_model = h2o.get_model(best_model_id)
                    aml.leader = best_model
                
                pred_calib = best_model_h2o.predict(h2o_calibration)
                calib_df = h2o_calibration.as_data_frame()
                pred_calib_df = pred_calib.as_data_frame()

                y_calib = calib_df[y_col].astype(int).values

                classes = h2o_train[y_col].levels()[0]
                prob_cols = [f"p{cls}" for cls in classes]

                X_calib = pred_calib_df[prob_cols].values
                
                isotonic_regressors = {}

                for idx, cls in enumerate(classes):
                    p_raw = X_calib[:, idx]
                    
                    y_binary = (y_calib == int(cls)).astype(int)
                    
                    iso_reg = IsotonicRegression(out_of_bounds='clip')
                    iso_reg.fit(p_raw, y_binary)
                    
                    isotonic_regressors[cls] = iso_reg
                
                best_model_calibrated_h2o = CalibratedModelClassificationMulti(best_model_h2o, isotonic_regressors, classes)

                output_dir_h2o = os.path.join(os.getcwd(), "h2o_models")
                if not os.path.exists(output_dir_h2o):
                    os.makedirs(output_dir_h2o)

                h2o_csv_file_path = os.path.join(output_dir_h2o, "h2o_ml_grid.csv")
                aml.get_leaderboard().as_data_frame().to_csv(h2o_csv_file_path)
                
                test_frame = h2o_test.as_data_frame()
                y_true = test_frame[y_col].astype(int).values

                y_pred_proba = best_model_calibrated_h2o.predict_proba(h2o_test)

                y_pred = best_model_calibrated_h2o.predict(h2o_test)

                y_pred = y_pred.astype(int)

                cm = confusion_matrix(y_true, y_pred)
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')
                balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

                try:
                    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                except ValueError:
                    auc = None
                    print("AUC cannot be computed due to insufficient class representation.")

                class_precision = precision_score(y_true, y_pred, average=None)
                class_recall = recall_score(y_true, y_pred, average=None)
                class_f1 = f1_score(y_true, y_pred, average=None)

                metrics = {
                    "Accuracy": accuracy,
                    "Weighted Precision": precision,
                    "Weighted Recall": recall,
                    "Weighted F1 Score": f1,
                    "Balanced Accuracy": balanced_accuracy
                }

                if auc is not None:
                    metrics["AUC"] = auc

                for i, cls in enumerate(classes):
                    metrics[f"Class {cls} Precision"] = class_precision[i]
                    metrics[f"Class {cls} Recall"] = class_recall[i]
                    metrics[f"Class {cls} F1 Score"] = class_f1[i]

                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                metrics_df.index.name = 'Metric'

                metrics_csv_path = os.path.join(output_dir_h2o, "calibrated_h2o_model_metrics.csv")
                metrics_df.to_csv(metrics_csv_path)

                print("Model evaluation metrics saved to:", metrics_csv_path)
                print(metrics)
                print(cm)

                ## TO SAVE THE MODEL
                best_model_calibrated_h2o.save(output_dir_h2o)
                
                ## TO LOAD THE MODEL
                # h2o.init()
                # from my_pandas_extensions.modeling import CalibratedModelClassificationMulti
                # # Define paths for model files
                # output_dir_h2o = os.path.join(os.getcwd(), "h2o_models")
                # model_path = os.path.join(output_dir_h2o, "H2O model files")  # Replace with your H2O model filename
                # calibrators_path = os.path.join(output_dir_h2o, 'calibrators.pkl')
                # metadata_path = os.path.join(output_dir_h2o, 'metadata.pkl')

                # # Load the H2O model
                # loaded_h2o_model = h2o.load_model(model_path)

                # # Load calibration information
                # with open(calibrators_path, 'rb') as f:
                #     isotonic_regressors = joblib.load(f)

                # with open(metadata_path, 'rb') as f:
                #     metadata = joblib.load(f)

                # # Extract the class labels from metadata
                # classes = metadata['classes']

                # # Recreate the calibrated model using the loaded components
                # best_model_calibrated_h2o = CalibratedModelClassificationMulti(loaded_h2o_model, isotonic_regressors, classes)
                
    
    elif y_col_type == "numeric":
        if pycaret_models:
  
            reg.setup(
                data = pycaret_data,
                target = y_col,
                train_size = train_size,
                preprocess = True,
                categorical_features = pycaret_categorical_features,
                handle_unknown_categorical = True,
                combine_rare_levels = False,
                ordinal_features = pycaret_ordinal_features,
                numeric_features = pycaret_numeric_features,
                fold_strategy = pycaret_fold_strategy,
                fold = fold,
                session_id = seed,
                log_experiment = False,
                experiment_name = log_experiment_name,
                use_gpu = pycaret_use_gpu_boost,
                silent = True, 
                n_jobs=1
            )
            
            list_models_pycaret = reg.compare_models(
                sort=pycaret_metric_prioritize,
                n_select=pycaret_n_select
            )

            tuned_models = []

            for model in list_models_pycaret:
                model_name = model.__class__.__name__
                
                tuned_model = None

                if model_name == 'LassoLars':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_LassoLars_parameters
                    )
                
                elif model_name == 'DummyRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize=pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_Dummy_parameters
                    )
                
                elif model_name == 'ElasticNet':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_ElasticNet_parameters
                    )
                
                elif model_name == 'BayesianRidge':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_BayesianRidge_parameters
                    )
                
                elif model_name == 'Lasso':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_Lasso_parameters
                    )
                
                elif model_name == 'AdaBoostRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_AdaBoost_parameters
                    )
                
                elif model_name == 'OrthogonalMatchingPursuit':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_OMP_parameters
                    )
                
                elif model_name == 'GradientBoostingRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_GBR_parameters
                    )
                
                elif model_name == 'Ridge':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_Ridge_parameters
                    )
                
                elif model_name == 'LinearRegression':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_LinearRegression_parameters
                    )
                
                elif model_name == 'Lars':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_Lars_parameters
                    )
                
                elif model_name == 'HuberRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize= pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_HuberRegressor_parameters
                    )
                
                elif model_name == 'LGBMRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize = pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_LGBM_parameters
                    )
                
                elif model_name == 'CatBoostRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize= pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_CatBoost_parameters
                    )
                    
                elif model_name == 'RandomForestRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize= pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_RandomForest_parameters
                    )
                
                elif model_name == 'KNeighborsRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize= pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_KNeighbors_parameters
                    )
                
                elif model_name == 'ExtraTreesRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize= pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_ExtraTrees_parameters
                    )
                
                elif model_name == 'DecisionTreeRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize= pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_DecisionTree_parameters
                    )
                
                elif model_name == 'PassiveAggressiveRegressor':
                    tuned_model = reg.tune_model(
                        model,
                        optimize= pycaret_metric_prioritize,
                        custom_grid = pycaret_reg_PassiveAggressive_parameters
                    )
                        
                tuned_models.append(tuned_model)

            
            output_dir_pycaret = os.path.join(os.getcwd(), "pycaret_models")
            if not os.path.exists(output_dir_pycaret):
                os.makedirs(output_dir_pycaret)
                    
            for i, model in enumerate(tuned_models):
                learning_curve_path = os.path.join(output_dir_pycaret, f"model_{model.__class__.__name__}_learning_curve.png")
                reg.plot_model(model, plot="learning", save=True)
                os.rename("Learning Curve.png", learning_curve_path)

                prediction_error_path = os.path.join(output_dir_pycaret, f"model_{model.__class__.__name__}_prediction_error.png")
                reg.plot_model(model, plot="error", save=True)
                os.rename("Prediction Error.png", prediction_error_path)

                residuals_path = os.path.join(output_dir_pycaret, f"model_{model.__class__.__name__}_residuals.png")
                reg.plot_model(model, plot="residuals", save=True)
                os.rename("Residuals.png", residuals_path)
                
                if model.__class__.__name__ == ("AdaBoostRegressor" or "GradientBoostingRegressor" or "LGBMRegressor" or "CatBoostRegressor" or "RandomForestRegressor" or "ExtraTreesRegressor" or "DecisionTreeRegressor"):
                    feature_importance_path = os.path.join(output_dir_pycaret, f"model_{model.__class__.__name__}_importance.png")
                    reg.plot_model(model, plot="feature", save=True)
                    os.rename("Feature Importance.png", feature_importance_path)
            
            # Collect metrics
            metrics_list = []
            
            for model in tuned_models:
                model_name = model.__class__.__name__
                reg.predict_model(model, verbose=False)
                metrics_df = reg.pull()
                metrics_df['Model'] = model_name
                metrics_list.append(metrics_df)

            tuned_models_grid = pd.concat(metrics_list, ignore_index=True)
    
            cols = ['Model'] + [col for col in tuned_models_grid.columns if col != 'Model']
            tuned_models_grid = tuned_models_grid[cols]
            
            tuned_models_grid = tuned_models_grid \
                .sort_values(by = pycaret_metric_prioritize, ascending = False)
            
            first_model_name = tuned_models_grid.iloc[0]['Model']
            
            # LIST - Best
            best_of_list_tuned_pycaret = None
            for model in tuned_models:
                if model.__class__.__name__ == first_model_name:
                    best_of_list_tuned_pycaret = model
                    break
            
            best_of_list_tuned_pycaret_grid = tuned_models_grid.iloc[[0]].reset_index().drop("index", axis = 1)
            
            # Individual Model - XGB
            if pycaret_use_CUDA is False:
                xgb_model_pycaret = reg.create_model(
                    estimator = "xgboost",
                    tree_method = "hist",
                    n_jobs=-1
                )
            else:
                xgb_model_pycaret = reg.create_model(
                    estimator = "xgboost",
                    tree_method = "gpu_hist",
                    n_jobs=-1
                )
            
            xgb_model_tuned_pycaret = reg.tune_model(
                estimator = xgb_model_pycaret,
                n_iter = pycaret_n_iter,
                optimize = pycaret_metric_prioritize,
                custom_grid = pycaret_reg_XGB_parameters
            )
            
            xgb_model_name = xgb_model_tuned_pycaret.__class__.__name__
            
            reg.predict_model(xgb_model_tuned_pycaret, verbose=False)
            XGB_tuned_pycaret_grid = reg.pull()
            
            XGB_tuned_pycaret_grid['Model'] = XGB_tuned_pycaret_grid['Model'].str.replace("Extreme Gradient Boosting", f"{xgb_model_name}")
            
                
            learning_curve_path = os.path.join(output_dir_pycaret, f"model_{xgb_model_name}_learning_curve.png")
            reg.plot_model(xgb_model_tuned_pycaret, plot="learning", save=True)
            os.rename("Learning Curve.png", learning_curve_path)

            prediction_error_path = os.path.join(output_dir_pycaret, f"model_{xgb_model_name}_prediction_error.png")
            reg.plot_model(xgb_model_tuned_pycaret, plot="error", save=True)
            os.rename("Prediction Error.png", prediction_error_path)

            residuals_path = os.path.join(output_dir_pycaret, f"model_{xgb_model_name}_residuals.png")
            reg.plot_model(xgb_model_tuned_pycaret, plot="residuals", save=True)
            os.rename("Residuals.png", residuals_path)
            
            feature_importance_path = os.path.join(output_dir_pycaret, f"model_{xgb_model_name}_importance.png")
            reg.plot_model(xgb_model_tuned_pycaret, plot="feature", save=True)
            os.rename("Feature Importance.png", feature_importance_path)
            
            # LIST - Blended models
            list_models_final_pycaret = list(tuned_models) + [xgb_model_tuned_pycaret]
            
            blended_models_tuned_pycaret = reg.blend_models(
                list_models_final_pycaret,
                optimize = pycaret_metric_prioritize    
            )
            
            
            reg.predict_model(blended_models_tuned_pycaret, verbose=False)
            blended_models_tuned_pycaret_grid = reg.pull()
            
            blended_models_tuned_pycaret_grid['Model'] = blended_models_tuned_pycaret_grid['Model'].str.replace("Voting Regressor", f"Pycaret Models Blended ({' + '.join([str(model.__class__.__name__) for model in tuned_models])})")
            
            
            blended_learning_curve_path = os.path.join(output_dir_pycaret, "blended_model_learning_curve.png")
            reg.plot_model(blended_models_tuned_pycaret, plot="learning", save=True)
            os.rename("Learning Curve.png", blended_learning_curve_path)

            prediction_error_path = os.path.join(output_dir_pycaret, "blended_model_prediction_error.png")
            reg.plot_model(blended_models_tuned_pycaret, plot="error", save=True)
            os.rename("Prediction Error.png", prediction_error_path)

            residuals_path = os.path.join(output_dir_pycaret, "blended_model_residuals.png")
            reg.plot_model(blended_models_tuned_pycaret, plot="residuals", save=True)
            os.rename("Residuals.png", residuals_path)


            # FINAL
            
            pycaret_models_ = [best_of_list_tuned_pycaret] + [xgb_model_tuned_pycaret] + [blended_models_tuned_pycaret]

            pycaret_grid = pd.concat([blended_models_tuned_pycaret_grid, XGB_tuned_pycaret_grid, best_of_list_tuned_pycaret_grid]) \
                .sort_values(by = pycaret_metric_prioritize, ascending = False) \
                .reset_index() \
                .drop("index", axis = 1)
            
            pycaret_csv_file_path = os.path.join(output_dir_pycaret, "pycaret_ml_grid.csv")
            pycaret_grid.to_csv(pycaret_csv_file_path)
                            
            pycaret_models_scores = {}

            for model in pycaret_models_:
                predictions_ = reg.predict_model(model)
                metrics_df = reg.pull()
                
                score = metrics_df[pycaret_metric_prioritize].iloc[0]
                
                pycaret_models_scores[model.__class__.__name__] = score

            best_model_name = max(pycaret_models_scores, key=pycaret_models_scores.get)
            
            best_model_pycaret = None
            for model in pycaret_models_:
                if model.__class__.__name__ == best_model_name:
                    best_model_pycaret = model
                else:
                    pass
            
            X_train = reg.get_config('X_train')
            y_train = reg.get_config('y_train')

            X_test = reg.get_config('X_test')
            y_test = reg.get_config('y_test')
            
                ### Calibration (isotonic regression)
                
            y_pred_raw = best_model_pycaret.predict(X_test)

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_pred_raw, y=y_test, alpha=0.5)
            plt.xlabel('Raw Predictions')
            plt.ylabel('Actual Target Values')
            plt.title('Raw Predictions vs. Actual Target Values')

            plt.savefig(os.path.join(output_dir_pycaret, "RELATIONSHIP_FOR_CALIBRATION.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            best_model_calibrated_pycaret = CalibrationWrapper(
                base_estimator = best_model_pycaret, 
                increasing = pycaret_reg_calibration_IncreasingRelationship
            )
            
            best_model_calibrated_pycaret.fit(X_train, y_train)
            
            best_model_calibrated_pycaret = reg.finalize_model(best_model_calibrated_pycaret)
            
            predictions_calibrated = best_model_calibrated_pycaret.predict(X_test)
            predictions_raw = best_model_calibrated_pycaret.predict_raw(X_test)

            rmse_calibrated = mean_squared_error(y_test, predictions_calibrated, squared=False)
            r2_calibrated = r2_score(y_test, predictions_calibrated)

            print(f"Calibrated Model RMSE: {rmse_calibrated}")
            print(f"Calibrated Model R2: {r2_calibrated}")
            
            reg.predict_model(best_model_calibrated_pycaret)
            best_model_calibrated_metrics_pycaret = reg.pull()
            
            prediction_error_path = os.path.join(output_dir_pycaret, f"{best_model_name}_prediction_error_Calibrated_Isotonic.png")
            reg.plot_model(best_model_calibrated_pycaret, plot="error", save=True)
            os.rename("Prediction Error.png", prediction_error_path)

            residuals_path = os.path.join(output_dir_pycaret, f"{best_model_name}_residuals_Calibrated_Isotonic.png")
            reg.plot_model(best_model_calibrated_pycaret, plot="residuals", save=True)
            os.rename("Residuals.png", residuals_path)
            
            reg.save_model(
                model = best_model_calibrated_pycaret,
                model_name = os.path.join(output_dir_pycaret, f"{best_model_name}_Calibrated_Isotonic_pycaret")
            )
            
            print("Best Model (Pycaret):")
            print(best_model_calibrated_metrics_pycaret)
                
        if h2o_models:

            h2o.init()

            df_h2o = h2o.H2OFrame(h2o_data)
            
            h2o_train, h2o_test = df_h2o.split_frame(ratios=[train_size], seed=seed)
            
            train_ratio = 1 - calibration_ratio
            h2o_train_actual, h2o_calibration = h2o_train.split_frame(ratios=[train_ratio], seed=seed)
            
            aml = H2OAutoML(
                nfolds = fold,
                seed = seed,
                stopping_metric = h2o_metric_prioritize,
                stopping_tolerance = 0.00858630100946073,
                stopping_rounds = 3,
                max_models=h2o_max_models
            )
            
            y = y_col
            x = h2o_x_cols
            
            aml.train(
                x=x,
                y=y,
                training_frame=h2o_train_actual
            )
            
            best_model_h2o = aml.leader
            
            calibration_preds = best_model_h2o.predict(h2o_calibration)
            
            calibration_df = h2o_calibration.as_data_frame()
            calibration_preds_df = calibration_preds.as_data_frame()
            
            y_true_calib = calibration_df[y_col].values
            y_pred_calib = calibration_preds_df['predict'].values
            
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(y_pred_calib, y_true_calib)

            best_model_calibrated_h2o = CalibratedModelRegression(best_model_h2o, iso_reg)
            
            output_dir_h2o = os.path.join(os.getcwd(), "h2o_models")
            if not os.path.exists(output_dir_h2o):
                os.makedirs(output_dir_h2o)
                
            h2o_csv_file_path = os.path.join(output_dir_h2o, "h2o_ml_leaderboard.csv")
            aml.leaderboard.as_data_frame().to_csv(h2o_csv_file_path, index=False)
            
            best_model_calibrated_h2o.save(output_dir_h2o)
            
            ## PREDICT DATAFRAME (Example)
            # leads_df = pd.read_csv('your_leads_data.csv')  # Replace with your actual data
            # df_predictions = best_model_calibrated_h2o.predict_dataframe(leads_df, include_data=True)
            
            test_frame = h2o_test.as_data_frame()
            y_true = test_frame[y_col].values
            
            y_pred = best_model_calibrated_h2o.predict(test_frame)
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                "Mean Absolute Error (MAE)": mae,
                "Root Mean Squared Error (RMSE)": rmse,
                "R-squared (R2)": r2
            }
            
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            metrics_df.index.name = 'Metric'
            
            metrics_csv_path = os.path.join(output_dir_h2o, 'calibrated_h2o_model_metrics.csv')
            metrics_df.to_csv(metrics_csv_path)
            
            print(metrics)
            
            ## TO LOAD THE MODEL
            # h2o.init()
            # from my_pandas_extensions.modeling import CalibratedModelRegression
            # best_model_calibrated_h2o = CalibratedModelRegression.load(output_dir_h2o)
            # Ensure that the directory contains:
            # - calibrated_model.pkl
            # - H2O model files (saved by h2o.save_model)
            
            # Example: Loading the Calibrated Model and Making Predictions
            # Ensure that the output_dir_h2o contains the saved model files
            # Load the calibrated model
            # calibrated_model = CalibratedModelRegression.load(output_dir_h2o)
            # Load new data for prediction
            # new_data = pd.read_csv('new_data.csv')  # Replace with your actual new data
            # Make calibrated predictions
            # calibrated_predictions = calibrated_model.predict(new_data)
            # Or get predictions along with the original data
            # predictions_with_data = calibrated_model.predict_dataframe(new_data, include_data=True)
            # print(predictions_with_data.head())


################################# END OF FUNCTION #################################

# clf_convert_to_probabilities()
# required packages: 
#   import pandas as pd
#   import numpy as np

def clf_convert_to_probabilities(
    df, 
    probability_threshold,
    score_0_column,
    score_1_column
):
    """
    Convert classifier scores to calibrated probabilities.

    This function takes classifier scores and a decision threshold, 
    and returns calibrated probabilities. It's useful for classifiers 
    that use a non-standard decision threshold (not 0.5).

    Args:
        df (pandas.DataFrame): DataFrame containing the classifier scores.
        probability_threshold (float): The decision threshold used by the classifier.
            This is the value of score_1 at which the classifier changes 
            its prediction from 0 to 1.
        score_0_column (str): Name of the column in df containing scores for class 0.
        score_1_column (str): Name of the column in df containing scores for class 1.

    Returns:
        pandas.Series: Calibrated probabilities, rounded to 4 decimal places.

    Example:
        >>> import pandas as pd
        >>> data = {'score_0': [0.8, 0.6], 'score_1': [0.2, 0.4]}
        >>> df = pd.DataFrame(data)
        >>> probs = clf_convert_to_probabilities(df, 0.3, 'score_0', 'score_1')
        >>> print(probs)
        0    0.2941
        1    0.7059
        dtype: float64
    """
    probability_threshold = probability_threshold
    odds_ratio = probability_threshold / (1 - probability_threshold)
    
    adjusted_odds = (df[score_1_column] / df[score_0_column]) / odds_ratio
    probabilities = adjusted_odds / (1 + adjusted_odds)
    probabilities = round(probabilities, 4)
    
    return probabilities

# clf_apply_model()
# required packages: 
#   import pandas as pd
#   import pycaret.classification as clf

def clf_apply_model(
    df,
    model_path
):
    """
    Apply a pre-trained classification model to a DataFrame and generate predictions.

    This function loads a saved PyCaret classification model and uses it to make predictions
    on the provided DataFrame. It returns a new DataFrame with the original data and
    additional columns for the predictions and raw scores.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the features for prediction.
    model_path : str
        The file path to the saved PyCaret classification model.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the original input data along with additional columns
        for the model's predictions and raw scores.

    Example:
    --------
    >>> import pandas as pd
    >>> leads_df = pd.read_csv('leads_data.csv')
    >>> result = apply_model(leads_df, "models/blended_models_final")

    Notes:
    ------
    - This function requires the PyCaret classification module to be imported as 'clf'.
    - The function uses PyCaret's 'load_model' and 'predict_model' functions.
    - The 'raw_score' parameter is set to True, so the output will include probability scores 
      for each class (if applicable).
    """
    model = clf.load_model(model_path)
    
    predictions_df = clf.predict_model(
        estimator = model,
        data = df,
        raw_score = True
    )
    
    return predictions_df

################################# END OF FUNCTION #################################