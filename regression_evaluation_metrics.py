
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def rmsle(y_test, y_preds):
    '''
    Computes Root Mean Squared Log Error of a regression model
    y_test: Ground truth
    y_preds: model predictions
    return: RMSLE
    '''
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

def regression_scores_from_model(model, X_train, y_train, X_val, y_val):
    '''
    Computes evaluation matrics for a regression model
    model: The trained model
    X_train: Training features
    y_train: Training targets
    X_val: Validation features
    y_val: Validation targets
    return: dictionary of scores
    '''
    y_preds_train = model.predict(X_train)
    y_preds_val = model.predict(X_val)
    scores = {'Training MAE': mean_absolute_error(y_train, y_preds_train),
              'Validation MAE': mean_absolute_error(y_val, y_preds_val),
              'Training RMSLE': rmsle(y_train, y_preds_train),
              'Validation RMSLE': rmsle(y_val, y_preds_val),
              'Training R^2': r2_score(y_train, y_preds_train),
              'Validation R^2': r2_score(y_val, y_preds_val),
             }
    return scores

def plot_feature_importance(model, X_train, n = 2):
    '''
    Computes and plots feature importance from model.
    model: Trained model
    X_train: Training features
    n: top n features for plotting
    return fig
    '''
    fig, ax = plt.subplots()
    features = pd.DataFrame({'Importance': model.feature_importances_*100}, index = list(X_train))
    features.sort_values('Importance', axis = 0).tail(n).plot(kind = 'barh', color = 'b', ax = ax)
    ax.set_xlabel('Feature Importance (x100)')
    ax.set_ylabel('Features')
    ax.get_legend().remove()
    plt.show()
    return fig

def preprocess_data(data_df):
    df = data_df.copy()
 
    # Handling numerical columns
    for col in list(df.select_dtypes(exclude = ['object'])):
        df[col + '_is_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(df[col].median())
        
    # Handling categorical columns
    for col in list(df.select_dtypes(exclude = ['number', 'datetime64'])):
        df[col + '_is_missing'] = df[col].isnull().astype(int)
        df[col] = pd.Categorical(df[col]).codes + 1

    df['saleYear'] = df.saledate.dt.year
    df['saleMonth'] = df.saledate.dt.month
    df['saleDay'] = df.saledate.dt.day
    df['saleDayOfWeek'] = df.saledate.dt.dayofweek
    df['saleDayOfYear'] = df.saledate.dt.dayofyear
    df.drop('saledate', axis = 1, inplace = True)

    return df
