import pandas as pd
import ast
import streamlit as st

@st.cache_data
def load_data(csv_path="TSLA_data.csv"):
    # st.write(f"[Data Loader] Attempting to load: {csv_path}") # Debug
    try:
        df = pd.read_csv(csv_path)
        # st.write(f"[Data Loader] CSV loaded. Original columns: {df.columns.tolist()}") # Debug

        original_date_column = 'timestamp' 
        expected_date_column = 'Date'

        if original_date_column not in df.columns:
            st.error(f"Data Loader: Critical date column '{original_date_column}' not found in CSV. Columns present: {df.columns.tolist()}")
            return pd.DataFrame()
        
        try:
            # *** CRITICAL: ADJUST THIS LINE FOR YOUR CSV'S DATE FORMAT ***
            # Example for Unix seconds: df[original_date_column] = pd.to_datetime(df[original_date_column], unit='s', errors='coerce')
            # Example for Unix milliseconds: df[original_date_column] = pd.to_datetime(df[original_date_column], unit='ms', errors='coerce')
            # Example for specific string format: df[original_date_column] = pd.to_datetime(df[original_date_column], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df[original_date_column] = pd.to_datetime(df[original_date_column], errors='coerce') 
        except Exception as e: 
            st.error(f"Data Loader: Error converting '{original_date_column}' to datetime: {e}. Check format.")
            if original_date_column in df and df[original_date_column].isnull().all():
                 st.error(f"Data Loader: All '{original_date_column}' values are NaT. Check format in CSV.")
                 return pd.DataFrame()
        
        if df[original_date_column].isnull().all():
            st.error(f"Data Loader: All '{original_date_column}' values are NaT after conversion. Please check the date format/values in your CSV and the pd.to_datetime() call in data_loader.py (e.g., add unit='s' if dates are Unix seconds).")
            return pd.DataFrame()
        df.dropna(subset=[original_date_column], inplace=True)

        if original_date_column != expected_date_column:
            df.rename(columns={original_date_column: expected_date_column}, inplace=True)

        ohlcv_rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        cols_to_rename_actually_present = {k:v for k,v in ohlcv_rename_map.items() if k in df.columns}
        if cols_to_rename_actually_present:
            df.rename(columns=cols_to_rename_actually_present, inplace=True)

        expected_ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_ohlcv_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') 
            else: 
                df[col] = pd.NA 

        def parse_price_list(price_str): 
            if pd.notnull(price_str) and isinstance(price_str, str) and price_str.strip().startswith('['):
                try: return ast.literal_eval(price_str)
                except (ValueError, SyntaxError): return []
            return []

        for band_col_prefix in ['Support', 'Resistance']: 
            list_col_csv = band_col_prefix 
            lower_col_df = f'{band_col_prefix}_Lower'
            upper_col_df = f'{band_col_prefix}_Upper'
            if list_col_csv in df.columns:
                df[f'{band_col_prefix}_List_Parsed_Temp'] = df[list_col_csv].apply(parse_price_list)
                df[lower_col_df] = df[f'{band_col_prefix}_List_Parsed_Temp'].apply(lambda x: min(x) if x else pd.NA)
                df[upper_col_df] = df[f'{band_col_prefix}_List_Parsed_Temp'].apply(lambda x: max(x) if x else pd.NA)
                df[lower_col_df] = pd.to_numeric(df[lower_col_df], errors='coerce')
                df[upper_col_df] = pd.to_numeric(df[upper_col_df], errors='coerce')
                if f'{band_col_prefix}_List_Parsed_Temp' in df.columns: 
                    df.drop(columns=[f'{band_col_prefix}_List_Parsed_Temp'], inplace=True, errors='ignore')
            else: 
                df[lower_col_df], df[upper_col_df] = pd.NA, pd.NA
        
        if 'direction' not in df.columns: 
            df['direction'] = None 

        if expected_date_column not in df.columns: 
            st.error(f"Data Loader: Logic error - Expected date column '{expected_date_column}' not found before sorting.")
            return pd.DataFrame()

        df.sort_values(expected_date_column, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e: 
        st.error(f"Data Loader Error: An unexpected error occurred: {e}")
        import traceback; st.text(traceback.format_exc()) 
        return pd.DataFrame()