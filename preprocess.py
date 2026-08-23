import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans raw training dataset by dropping duplicates and imputing missing values.
    """
    if df is None or df.empty:
        return df
        
    # Drop duplicates
    cleaned_df = df.drop_duplicates()
    
    # Fill numerical missing values with mean
    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            
    # Fill categorical missing values with mode
    for col in cleaned_df.select_dtypes(exclude=[np.number]).columns:
        if cleaned_df[col].isnull().any():
            mode_val = cleaned_df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Normal"
            cleaned_df[col] = cleaned_df[col].fillna(fill_val)
            
    return cleaned_df
