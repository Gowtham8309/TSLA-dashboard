import streamlit as st
import google.generativeai as genai
import pandas as pd 

def configure_gemini_model():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model
    except Exception as e:
        st.sidebar.error(f"Error configuring Gemini API: {e}")
        return None

def get_gemini_response(model: genai.GenerativeModel, dataframe: pd.DataFrame, user_prompt: str):
    if not model:
        return "Gemini model not available. Check API key configuration."
    if dataframe.empty:
        return "Data not loaded. Cannot answer questions."

    df_copy = dataframe.copy()
    if 'Date' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['Date']):
        try:
            df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
        except Exception:
            pass

    schema_description_parts = ["DataFrame Schema:"]
    for col in df_copy.columns:
        col_type = str(df_copy[col].dtype)
        examples = df_copy[col].dropna().unique()[:2]
        example_str = ", ".join(map(str, examples)) if len(examples) > 0 else ("Empty/all NaN" if df_copy[col].isnull().all() else "N/A")
        col_detail = f"- {col} ({col_type}, examples: {example_str})"
        if col == 'Date' and pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            col_detail += ". Use for date operations (filtering by year, month, quarter, weekday, etc.). Dates are YYYY-MM-DD."
        elif col == 'direction':
            col_detail += ". Contains 'LONG', 'SHORT', or NaN/None. 'LONG' indicates bullish. If missing, fallback to Close > Open."
        elif col_type.startswith('float') or col_type.startswith('int'):
            col_detail += ". Numeric data."
        schema_description_parts.append(col_detail)
    schema_description = "\n".join(schema_description_parts)

    min_date_str, max_date_str = "N/A", "N/A"
    if 'Date' in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy['Date']) and not df_copy['Date'].dropna().empty:
        min_date_str = df_copy['Date'].dropna().min().strftime('%Y-%m-%d')
        max_date_str = df_copy['Date'].dropna().max().strftime('%Y-%m-%d')

    few_shot_examples = f"""
Example Questions & Your Structured Plan Output:

Q: How many days in 2023 was TSLA bullish?
PLAN:
  operation: count_rows
  filters:
    - column: Date
      condition: year_equals
      value: 2023
  compute:
    logic: count rows where (direction == 'LONG') or (direction is NaN and Close > Open)

Q: What was the highest 'High' price recorded in the dataset?
PLAN:
  operation: find_max
  column: High
  filters: None

Q: What was the average high price on Mondays in 2023?
PLAN:
  operation: calculate_average
  column: High
  filters:
    - column: Date
      condition: year_equals
      value: 2023
    - column: Date
      condition: weekday_equals
      value: Monday
"""

    data_description_prompt = f"""
You are an AI assistant that helps analyze TSLA stock data by generating a structured plan for Python/Pandas execution.
Based on the user's question and the provided DataFrame schema, output a structured PLAN in YAML-like format that outlines the operations needed.
If the question is qualitative or cannot be answered by a simple data operation, provide a concise textual answer or explanation. Your primary goal is to create an actionable plan if possible.

DataFrame Schema:
{schema_description}

Overall Date range of the full dataset: {min_date_str} to {max_date_str}.

Definitions for your plan:
- "Bullish day": 'direction' == 'LONG' OR ('direction' is missing AND Close > Open).
- For counts based on 'direction', assume NaN/missing values should fallback to price-based logic.
- For weekday filters, use `weekday_equals` with values like Monday, Tuesday, etc.

{few_shot_examples}

User's question: "{user_prompt}"

Your Structured PLAN (or textual answer if a plan is not applicable):
"""
    try:
        response = model.generate_content(data_description_prompt)
        if not response.parts:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                return f"Response blocked by API due to: {response.prompt_feedback.block_reason}."
            if hasattr(response, 'candidates') and response.candidates and response.candidates[0].finish_reason != 1:
                return f"AI could not complete the response. Reason code: {response.candidates[0].finish_reason}."
            return "AI returned an empty or blocked response."
        return response.text
    except Exception as e:
        return f"Error communicating with AI: {e}"
