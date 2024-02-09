
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
    
    missing_cols = set(feature_names_sorted) - set(list(df))
    if len(missing_cols) > 0:
        for col in missing_cols:
            df[col] =  [0] * len(df)
        new_col_names = list(df)
        new_col_names.sort()
        df = df.loc[:, new_col_names] 
    else:
        new_col_names = list(df)
        new_col_names.sort()
        df = df.loc[:, new_col_names]   
    return df

feature_names_sorted = ['Backhoe_Mounting', 'Backhoe_Mounting_is_missing', 'Blade_Extension',
                        'Blade_Extension_is_missing', 'Blade_Type', 'Blade_Type_is_missing',
                        'Blade_Width', 'Blade_Width_is_missing', 'Coupler', 'Coupler_System',
                        'Coupler_System_is_missing', 'Coupler_is_missing','Differential_Type',
                        'Differential_Type_is_missing', 'Drive_System', 'Drive_System_is_missing',
                        'Enclosure', 'Enclosure_Type', 'Enclosure_Type_is_missing', 'Enclosure_is_missing',
                        'Engine_Horsepower', 'Engine_Horsepower_is_missing', 'Forks', 'Forks_is_missing',
                        'Grouser_Tracks', 'Grouser_Tracks_is_missing', 'Grouser_Type', 'Grouser_Type_is_missing',
                        'Hydraulics', 'Hydraulics_Flow', 'Hydraulics_Flow_is_missing', 'Hydraulics_is_missing',
                        'MachineHoursCurrentMeter', 'MachineHoursCurrentMeter_is_missing', 'MachineID',
                        'MachineID_is_missing', 'ModelID', 'ModelID_is_missing', 'Pad_Type',
                        'Pad_Type_is_missing', 'Pattern_Changer', 'Pattern_Changer_is_missing', 'ProductGroup',
                        'ProductGroupDesc', 'ProductGroupDesc_is_missing', 'ProductGroup_is_missing', 
                        'ProductSize', 'ProductSize_is_missing', 'Pushblock', 'Pushblock_is_missing',
                        'Ride_Control', 'Ride_Control_is_missing','Ripper', 'Ripper_is_missing',
                        'SalePrice_is_missing', 'SalesID', 'SalesID_is_missing', 'Scarifier', 
                        'Scarifier_is_missing', 'Steering_Controls', 'Steering_Controls_is_missing', 'Stick',
                        'Stick_Length', 'Stick_Length_is_missing', 'Stick_is_missing', 'Thumb',
                        'Thumb_is_missing', 'Tip_Control', 'Tip_Control_is_missing', 'Tire_Size',
                        'Tire_Size_is_missing', 'Track_Type', 'Track_Type_is_missing', 'Transmission',
                        'Transmission_is_missing', 'Travel_Controls', 'Travel_Controls_is_missing',
                        'Turbocharged', 'Turbocharged_is_missing', 'Undercarriage_Pad_Width',
                        'Undercarriage_Pad_Width_is_missing', 'UsageBand', 'UsageBand_is_missing',
                        'YearMade', 'YearMade_is_missing', 'auctioneerID', 'auctioneerID_is_missing',
                        'datasource', 'datasource_is_missing', 'fiBaseModel', 'fiBaseModel_is_missing',
                        'fiModelDesc', 'fiModelDesc_is_missing', 'fiModelDescriptor',
                        'fiModelDescriptor_is_missing', 'fiModelSeries', 'fiModelSeries_is_missing',
                        'fiProductClassDesc', 'fiProductClassDesc_is_missing', 'fiSecondaryDesc',
                        'fiSecondaryDesc_is_missing', 'saleDay', 'saleDayOfWeek', 'saleDayOfYear',
                        'saleMonth', 'saleYear', 'saledate_is_missing','state', 'state_is_missing']
