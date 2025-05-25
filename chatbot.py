import streamlit as st
import google.generativeai as genai
import pandas as pd
import yaml # Import yaml for better representation in prompt

def configure_gemini_model():
    """Configures the Gemini API model from Streamlit secrets."""
    try:
        # Access the API key from Streamlit secrets
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        # Use a suitable model, gemini-1.5-flash-latest is generally good for this
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        # Test the model to ensure it's configured correctly (optional)
        # try:
        #     model.generate_content("Ping", safety_settings=[{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}])
        #     # If successful, continue
        # except Exception as e:
        #     st.sidebar.warning(f"Gemini model configured but initial 'Ping' failed: {e}. May indicate connectivity or safety settings issue.")
        return model
    except KeyError:
        st.sidebar.error("GEMINI_API_KEY not found in Streamlit secrets. Chatbot will not function.")
        return None
    except FileNotFoundError:
        st.sidebar.error("Secrets file (.streamlit/secrets.toml) not found. Chatbot will not function.")
        return None
    except Exception as e:
        st.sidebar.error(f"Error configuring Gemini API: {e}")
        return None

def get_gemini_response(model: genai.GenerativeModel, dataframe: pd.DataFrame, user_prompt: str):
    """
    Gets a response from the Gemini model based on user prompt and dataframe schema.
    Guides the model to output a structured plan.
    """
    if not model:
        return "Gemini model not available. Check API key configuration."
    if dataframe.empty:
        return "Stock data not loaded. Cannot answer questions based on data."

    # Create a copy to avoid modifying the original cached DataFrame
    df_copy = dataframe.copy()

    # Ensure Date column is datetime for proper description if not already
    if 'Date' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['Date']):
        try:
            df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
        except Exception:
            pass # Ignore if conversion fails, describe it as is

    # --- Generate a description of the DataFrame schema and data range ---
    schema_description_parts = ["DataFrame Schema (available columns and their types, with examples):"]
    for col in df_copy.columns:
        col_type = str(df_copy[col].dtype)
        # Get up to 2 unique non-null examples
        examples = df_copy[col].dropna().unique()[:2]
        example_str = ", ".join(map(str, examples)) if len(examples) > 0 else ("Empty/all NaN" if df_copy[col].isnull().all() else "No distinct examples in sample")

        col_detail = f"- `{col}` (type: {col_type}, examples: {example_str})"
        if col == 'Date' and pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            col_detail += ". This column should be used for any date-based filtering (e.g., by year, month, quarter, specific date range, day of week) and aggregations. Dates are in YYYY-MM-DD format after processing."
        elif col == 'direction':
            col_detail += ". For 'bullish'/'bearish' queries, 'LONG' means bullish, 'SHORT' means bearish. NaN or empty values in this column mean 'NEUTRAL' or undefined. Possible values include: LONG, SHORT."
        elif col_type.startswith('float') or col_type.startswith('int'):
            col_detail += ". This is a numeric column, likely representing price, count, or volume."
        schema_description_parts.append(col_detail)
    columns_description = "\n".join(schema_description_parts)

    # Get the date range of the actual data
    min_date_str, max_date_str = "N/A", "N/A"
    if 'Date' in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy['Date']) and not df_copy['Date'].dropna().empty:
        min_date = df_copy['Date'].dropna().min()
        max_date = df_copy['Date'].dropna().max()
        # Format dates if they are valid Timestamps
        if isinstance(min_date, pd.Timestamp): min_date_str = min_date.strftime('%Y-%m-%d')
        if isinstance(max_date, pd.Timestamp): max_date_str = max_date.strftime('%Y-%m-%d')


    # Define few-shot examples to guide the model's output format
    # IMPORTANT: These examples must match the structure expected by execute_llm_plan in app.py
    few_shot_examples = f"""
Example Questions & Your Structured Plan Output:

Q: What was the highest 'High' price recorded in the dataset?
PLAN:
  operation: find_max
  column: High

Q: How many days were marked 'LONG' in the 'direction' column for the year 2023?
PLAN:
  operation: count_rows
  filters:
    - column: Date
      condition: year_equals
      value: 2023
    - column: direction
      condition: equals
      value: LONG
  notes: This counts rows filtered by year 2023 and direction equals LONG.

Q: What was the highest closing price of TSLA in 2023, and on which date did it occur?
PLAN:
  operation: find_max_with_date
  column: Close
  filters:
    - column: Date
      condition: year_equals
      value: 2023
  output_columns: [Date, Close, Open, Low, High] # Request related OHLC values for context

Q: What was the lowest opening price in Q2 2023?
PLAN:
  operation: find_min_with_date
  column: Open
  filters:
    - column: Date
      condition: quarter_equals
      value: 2
    - column: Date
      condition: year_equals
      value: 2023
  output_columns: [Date, Open, Close, High, Low] # Request related OHLC values for context

Q: What was the average trading volume in December 2023?
PLAN:
  operation: calculate_average
  column: Volume
  filters:
    - column: Date
      condition: month_equals
      value: 12
    - column: Date
      condition: year_equals
      value: 2023

Q: Which month in 2023 had the most bullish days?
PLAN:
  operation: find_group_with_max_count
  group_by_column: Date # The column to group by (should be datetime)
  group_by_period: month # The period to group by ('month' or 'dayofweek')
  count_condition_column: direction # The column to check the condition on for counting
  count_condition_value: LONG # The value in count_condition_column to count occurrences of
  filters:
    - column: Date
      condition: year_equals
      value: 2023 # Filter the overall data first
  notes: This finds the month in 2023 with the highest count of days where 'direction' is 'LONG'.

Q: What was the average high price on Mondays in 2023?
PLAN:
  operation: calculate_average
  column: High
  filters:
    - column: Date
      condition: year_equals
      value: 2023
    - column: Date
      condition: dayofweek_equals # Use the new condition
      value: Monday # Use string name for clarity in the plan (execute_llm_plan handles conversion)

Q: What was the largest intraday range in 2023 and on which date?
PLAN:
  operation: find_max_with_date
  column: Intraday_Range # Refer to the calculated column
  filters:
    - column: Date
      condition: year_equals
      value: 2023
  output_columns: [Date, Intraday_Range, High, Low] # Show the range and its components

Q: How many days in 2023 did the close price exceed 250?
PLAN:
  operation: count_rows
  filters:
    - column: Date
      condition: year_equals
      value: 2023
    - column: Close
      condition: greater_than # Use the new comparison condition
      value: 250

Q: Which weekday had the most days with a 'SHORT' direction?
PLAN:
  operation: find_group_with_max_count
  group_by_column: Date
  group_by_period: dayofweek # Use the new grouping period
  count_condition_column: direction
  count_condition_value: SHORT
  filters: [] # No specific year filter requested

# Add other examples here as you expand execute_llm_plan operations
"""

    # Construct the full prompt for the LLM
    data_description_prompt = f"""
You are an AI assistant specializing in analyzing TSLA stock data provided in a pandas DataFrame.
Your task is to understand the user's question and generate a structured plan in YAML format for a Python script to execute the required data operation using Pandas.
Adhere strictly to the specified structure and available operations.

If the user's question is not about extracting specific data points or calculations from the provided DataFrame that directly map to the 'Available Operations' (e.g., a general question about market trends, predictions, news, or complex analysis like "volatility difference", "consecutive streaks", or "testing support/resistance levels" which require custom calculations or sequence analysis not included below), respond conversationally by stating that you can only analyze the provided data using the defined operations and may offer to provide a general answer based on your training data if applicable.

**Available Operations you can plan for:**
- `count_rows`: Count rows after applying filters.
- `find_max`: Find the maximum value in a specified numeric column after filtering.
- `find_max_with_date`: Find the maximum value in a specified numeric column and the date it occurred, after filtering.
- `find_min_with_date`: Find the minimum value in a specified numeric column and the date it occurred, after filtering.
- `calculate_average`: Calculate the mean of a specified numeric column after filtering.
- `find_group_with_max_count`: Find the group (e.g., month, weekday) with the maximum count of rows that meet a specific condition, after applying filters.

DataFrame Schema (derived columns like Intraday_Range are also available):
{columns_description}

Overall Date range of the full dataset: {min_date_str} to {max_date_str}.

Definitions for your plan:
- When a user asks about "bullish" days, assume they mean days where the 'direction' column is 'LONG'. "Bearish" days are where 'direction' is 'SHORT'. Days with no direction are 'NEUTRAL'.
- When counting occurrences based on a condition (like 'direction' is 'LONG' or 'SHORT'), assume you should ignore rows where the `count_condition_column` is null or empty, unless the query specifically asks about neutral/undetermined days.
- For operations requiring a specific column (`column` key), it must be a column from the DataFrame schema (including derived ones like `Intraday_Range`).
- For filtering by date (`filters` with `column: Date`), use conditions:
    - `year_equals`, `month_equals`, `quarter_equals`: `value` is the numeric year, month (1-12), or quarter (1-4).
    - `dayofweek_equals`: `value` is the full name of the day (e.g., 'Monday', 'Tuesday', ..., 'Sunday').
- For filtering by other numeric columns, use conditions:
    - `equals`: `value` is the exact value.
    - `greater_than` (or `gt`, `>`): `value` is the threshold.
    - `less_than` (or `lt`, `<`): `value` is the threshold.
    - `greater_than_or_equals` (or `gte`, `>=`): `value` is the threshold.
    - `less_than_or_equals` (or `lte`, `<=`): `value` is the threshold.
- For filtering by string/object columns (like 'direction'), only use the `equals` condition.
- For `find_max_with_date` and `find_min_with_date`, if the user asks for other details, list those column names in the `output_columns` list.
- For `find_group_with_max_count`, you must specify `group_by_column` (must be 'Date'), `group_by_period` (must be 'month' or 'dayofweek'), `count_condition_column`, and `count_condition_value` ('LONG' or 'SHORT'). Do NOT attempt to group by periods not listed here or count 'NEUTRAL' days with this operation.

{few_shot_examples}

If a question clearly asks for a calculation or specific data retrieval that fits one of the 'Available Operations' using the specified filters and grouping methods, generate a structured PLAN.
If the question is ambiguous, too broad, requires complex logic (like sequence analysis, custom calculations not based on existing columns, or comparisons between columns), or asks about operations not listed, respond textually explaining your limitations.

User's question: "{user_prompt}"

Your Structured PLAN (or a textual response if a plan is not applicable):
"""
    try:
        # Generate content using the prompt
        # Added safety settings to potentially reduce blocked responses
        response = model.generate_content(data_description_prompt, safety_settings=[{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}])

        # Check if the response is empty or blocked
        if not response.parts:
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 # Provide specific reason if blocked
                 return f"Response blocked by AI due to: {response.prompt_feedback.block_reason}."
             if hasattr(response, 'candidates') and response.candidates and response.candidates[0].finish_reason != 1:
                  # Provide finish reason if not completed successfully (finish_reason 1 means success)
                  return f"AI could not complete the response. Reason code: {response.candidates[0].finish_reason}."
             # Generic message for empty response if no specific reason
             return "AI returned an empty response."

        # Return the text content of the response
        return response.text

    except Exception as e:
        # Catch any exceptions during API communication
        return f"Error communicating with AI: {e}"