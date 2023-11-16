from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer,MinMaxScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
from sklearn_pandas import DataFrameMapper
import sklearn

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import optuna
import xgboost as xgb

SEED = 123
from optuna.integration import XGBoostPruningCallback
import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:\\Users\\magda\\OneDrive\\Escritorio\\SCRIPTS_VARIOS\\LabMDS-1\\Lab_9\\sales.csv')








df['date'] = pd.to_datetime(df['date'])

X = df.drop("quantity",axis=1)
y = df["quantity"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33333, random_state=SEED)


def date2dmy(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day.astype('category')
    df['month'] = df['date'].dt.month.astype('category')
    df['year'] = df['date'].dt.year.astype('category')
    return df[['day', 'month', 'year']]

date_transformer = FunctionTransformer(date2dmy, validate=False)
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(transformers=[('date', date_transformer, ['date']),
                                               ('num', StandardScaler(), numerical_features),
                                               ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_features)])
preprocessor.set_output(transform='pandas')

pipe_dummy = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', DummyRegressor())])
#display(pipe_dummy)

 
pipe_dummy.fit(X_train, y_train)

y_val_pred = pipe_dummy.predict(X_val)

mae = mean_absolute_error(y_val, y_val_pred)
print(f'MAE del conjunto val para dummy: {mae}')

pipe_xgboost_const = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', XGBRegressor(random_state=SEED, 
                                                                monotone_constraints={"num__price": -1},
                                                                enable_categorical = True,))])
X_transformed = preprocessor.transform(X_train)
 
 
pipe_xgboost_const.fit(X_train, y_train)
y_val_pred_xgboost_const = pipe_xgboost_const.predict(X_val)

mae_xgboost_const = mean_absolute_error(y_val, y_val_pred_xgboost_const)
print(f'MAE del conjunto val para XGBoost con relaciones: {mae_xgboost_const}')

from sklearn.base import BaseEstimator, TransformerMixin
def objective(trial):
     

    # Create the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    # Assuming you have a DataFrame X
    X_transformed = pipeline.fit_transform(X_train)
    X_val_trans = pipeline.transform(X_val)
    
    dtrain = xgb.DMatrix(X_transformed, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_val_trans, label=y_val, enable_categorical=True)

    xgb_params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'max_leaves': trial.suggest_int('max_leaves', 0, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }

    # Set the XGBoost parameters
    reg = XGBRegressor(random_state=SEED, **xgb_params, enable_categorical=True)

    # Set the min_frequency parameter in OneHotEncoder
    min_frequency = trial.suggest_float('min_frequency', 0.0, 0.1)
    pipeline.named_steps['preprocessor'].named_transformers_['cat'].set_params(min_frequency=min_frequency)

    # Add a callback for pruning.
    pruning_callback = XGBoostPruningCallback(trial, 'validation_1-mae')

    reg.fit(X_transformed, y_train, 
            eval_metric=['mae'], 
            eval_set=[(X_transformed, y_train), (X_val_trans, y_val)], 
            callbacks=[pruning_callback])
    y_val_pred_xgboost_optuna = reg.predict(X_val_trans)
    # Calculate the mean absolute error
    mae_xgboost_optuna = mean_absolute_error(y_val, y_val_pred_xgboost_optuna)

    return mae_xgboost_optuna 


study = optuna.create_study(direction="minimize"
    )
study.optimize(objective, n_trials=100)
print(study.best_trial)