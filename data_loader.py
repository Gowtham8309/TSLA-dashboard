import pandas as pd
import ast
import streamlit as st

@st.cache_data
def load_data(csv_path="TSLA_data.csv"):
    """Loads data from CSV, performs cleaning and preprocessing."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            st.error("Data Loader: CSV file is empty.")
            return pd.DataFrame()

        original_date_column = 'timestamp'
        expected_date_column = 'Date'

        if original_date_column not in df.columns:
            st.error(f"Data Loader: Date column '{original_date_column}' not found in CSV.")
            return pd.DataFrame()

        # Convert to datetime, coercing errors to NaT
        # Add unit='s' if your timestamp is in seconds since epoch, otherwise pandas might guess incorrectly
        # df[original_date_column] = pd.to_datetime(df[original_date_column], errors='coerce', unit='s') # Example for seconds epoch
        df[original_date_column] = pd.to_datetime(df[original_date_column], errors='coerce') # Standard datetime parsing

        if df[original_date_column].isnull().all():
            st.error(f"Data Loader: All values in '{original_date_column}' became NaT. Check CSV date format and parsing.")
            return pd.DataFrame()

        # Drop rows where date is essential and failed conversion
        initial_row_count = len(df)
        df.dropna(subset=[original_date_column], inplace=True)
        if len(df) < initial_row_count:
             st.warning(f"Dropped {initial_row_count - len(df)} rows due to invalid date values.")


        if original_date_column != expected_date_column:
            df.rename(columns={original_date_column: expected_date_column}, inplace=True)

        # Rename and convert OHLCV columns
        ohlcv_rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        cols_to_rename_present = {k:v for k,v in ohlcv_rename_map.items() if k in df.columns}
        if cols_to_rename_present:
            df.rename(columns=cols_to_rename_present, inplace=True)

        expected_ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_ohlcv_cols:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                 # Add missing OHLCV columns as NaN if they don't exist
                 df[col] = pd.NA # Use pandas nullable integer/float type

        # --- Add derived columns here ---
        # Calculate Intraday Range
        if 'High' in df.columns and 'Low' in df.columns:
             # Ensure they are numeric before subtraction
             df['Intraday_Range'] = pd.to_numeric(df['High'], errors='coerce') - pd.to_numeric(df['Low'], errors='coerce')
        else:
             df['Intraday_Range'] = pd.NA # Add as NaN if High/Low missing


        # --- Handle Support/Resistance Lists ---
        def parse_price_list(price_str):
            if pd.notnull(price_str) and isinstance(price_str, str) and price_str.strip().startswith('['):
                try:
                    # Safely evaluate the string as a literal list
                    parsed_list = ast.literal_eval(price_str)
                    # Filter out non-numeric values after parsing
                    return [float(x) for x in parsed_list if isinstance(x, (int, float))]
                except (ValueError, SyntaxError, TypeError):
                    return [] # Return empty list if parsing or conversion fails
            return [] # Return empty list for NaN, non-string, or non-list-like strings

        for band_col_prefix in ['Support', 'Resistance']:
            list_col_csv = band_col_prefix
            lower_col_df = f'{band_col_prefix}_Lower'
            upper_col_df = f'{band_col_prefix}_Upper'

            if list_col_csv in df.columns:
                # Apply parsing function
                parsed_list_col_name = f'{band_col_prefix}_List_Parsed_Temp'
                df[parsed_list_col_name] = df[list_col_csv].apply(parse_price_list)

                # Extract min and max from the parsed list
                df[lower_col_df] = df[parsed_list_col_name].apply(lambda x: min(x) if x else pd.NA)
                df[upper_col_df] = df[parsed_list_col_name].apply(lambda x: max(x) if x else pd.NA)

                # Ensure the results are numeric
                df[lower_col_df] = pd.to_numeric(df[lower_col_df], errors='coerce')
                df[upper_col_df] = pd.to_numeric(df[upper_col_df], errors='coerce')

                # Drop the temporary parsed list column
                if parsed_list_col_name in df.columns:
                    df.drop(columns=[parsed_list_col_name], inplace=True, errors='ignore')
            else:
                # Add columns as NaN if the original list column is missing
                df[lower_col_df], df[upper_col_df] = pd.NA, pd.NA # Ensure columns exist


        # --- Handle Direction Column ---
        if 'direction' not in df.columns:
            df['direction'] = None # Add if missing
        else:
            # Standardize direction: convert to string and handle common missing values
            df['direction'] = df['direction'].apply(lambda x: None if pd.isna(x) or str(x).strip().lower() in ['', 'none', 'nan'] else str(x).strip().upper()) # Standardize to uppercase, map empty/NaN to None


        if expected_date_column not in df.columns:
             # This is a critical error, should be caught earlier, but defensive check
            st.error(f"Data Loader: Critical logic error - '{expected_date_column}' column not found before sorting.")
            return pd.DataFrame()

        # Sort by date
        df.sort_values(expected_date_column, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Optional: Add a final check for critical columns if desired
        # critical_plot_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        # nan_check = df[critical_plot_cols].isnull().sum()
        # if nan_check.sum() > 0:
        #     st.warning(f"[Data Loader] NaN values found in critical columns after processing: \n{nan_check[nan_check > 0]}")

        return df
    except Exception as e:
        st.error(f"Data Loader Error: An unexpected error occurred: {e}")
        import traceback; st.text(traceback.format_exc())
        return pd.DataFrame()