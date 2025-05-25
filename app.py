import streamlit as st
import pandas as pd
import time
import re
import yaml
import ast # Make sure ast is imported for literal_eval in parse_llm_plan_robustly

from data_loader import load_data
from chart_generator import create_tv_style_plotly_chart
from chatbot import configure_gemini_model, get_gemini_response # Ensure this import is correct

st.set_page_config(layout="wide", page_title="Pro TSLA Dashboard", initial_sidebar_state="expanded")

DATA_FILE_PATH = "TSLA_data.csv"
# Load data once and add derived columns like Intraday_Range - this is now done in data_loader.py
data_df = load_data(DATA_FILE_PATH)

if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = configure_gemini_model()

st.title("🚀 TSLA ProView Dashboard")

if data_df.empty:
    st.error("Failed to load TSLA data. Dashboard cannot be displayed. Please check `data_loader.py` and your CSV, especially the date column format.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Data Overview")
    st.markdown(f"**Total Records:** {len(data_df):,}")
    if 'Date' in data_df.columns and not data_df['Date'].empty:
        min_d, max_d = data_df['Date'].min(), data_df['Date'].max()
        if pd.notna(min_d): st.markdown(f"**From:** {min_d.strftime('%b %d, %Y')}")
        if pd.notna(max_d): st.markdown(f"**To:** {max_d.strftime('%b %d, %Y')}")
    else: st.markdown("**Date Range:** Not available")
    st.markdown("---")
    st.subheader("Recent Data:")
    st.dataframe(data_df.tail(5), height=210, use_container_width=True)

tab_titles = ["📊 Trading Chart", "🤖 AI Data Insights", "🎬 Chart Replay"]
tab1, tab2, tab_bonus = st.tabs(tab_titles)

with tab1:
    if data_df.empty:
        st.warning("No data available to display chart.")
    else:
        chart_render_height = 700
        plotly_fig = create_tv_style_plotly_chart(data_df, chart_title="Tesla, Inc. (TSLA)", chart_height=chart_render_height, is_animation_frame=False)
        if plotly_fig:
            st.plotly_chart(plotly_fig, use_container_width=True, theme=None)
        else:
            st.error("Plotly chart could not be generated.")

def parse_llm_plan_robustly(plan_str: str):
    """
    Attempts to parse the LLM's plan string into a dictionary,
    first using YAML, then using regex for basic operations if YAML fails.
    Extracts notes if present.
    Returns None if parsing fails and it doesn't look like a structured plan.
    """
    plan = {}
    cleaned_plan_str = plan_str.strip() # Strip leading/trailing whitespace

    # Attempt YAML parsing first (most reliable if format is consistent)
    # Add "PLAN:" back if it was stripped but the content looks like YAML
    # Check if it starts with a common key indicating YAML structure
    looks_like_yaml_start = cleaned_plan_str.lower().startswith(('operation:', 'filters:', 'column:', '- '))
    if looks_like_yaml_start and not cleaned_plan_str.lower().startswith('plan:'):
         cleaned_plan_str_to_parse = "PLAN:\n" + cleaned_plan_str
    else:
         cleaned_plan_str_to_parse = cleaned_plan_str # Assume it's either pure text or starts with PLAN:


    # Extract notes section if present (case-insensitive, start of line or after newline)
    notes_section = "";
    # Look for notes section specifically structured after potential YAML keys
    # Adjusted regex to look for notes anywhere, not just start of line, but as a distinct section
    notes_match = re.search(r"(?:\n|^)\s*notes:\s*(.*)$", cleaned_plan_str, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if notes_match:
         notes_section = notes_match.group(1).strip()
         # Remove the notes section from the string used for parsing
         cleaned_plan_str_to_parse = re.sub(r"(?:\n|^)\s*notes:\s*.*$", "", cleaned_plan_str_to_parse, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL).strip()


    parsed_yaml = None
    # Try parsing the potentially prefixed/cleaned string
    try:
        parsed_yaml = yaml.safe_load(cleaned_plan_str_to_parse)
        if isinstance(parsed_yaml, dict) and 'operation' in parsed_yaml:
             if notes_section: parsed_yaml['notes_from_llm'] = notes_section
             return parsed_yaml
    except (yaml.YAMLError, Exception) as e:
        print(f"YAML parsing failed: {e}")
        # If YAML fails, proceed to fallback check

    # Fallback: If YAML parsing failed, check if the *original* raw string looks like an attempted plan structure
    # This helps distinguish malformed plans from pure textual responses
    if re.search(r"operation:\s*([\w_]+)", plan_str, re.IGNORECASE) or \
       re.search(r"filters:\s*(\[|(-|\w+:))", plan_str, re.IGNORECASE) or \
       re.search(r"column:\s*([\w\.'\"\[\]]+)", plan_str, re.IGNORECASE):
        # It contains keywords suggesting a plan structure, but YAML failed.
        # Attempt basic regex parsing to get at least the operation and maybe few keys.
        # Note: This regex fallback is less robust for complex structures like filters lists.
        plan_fallback = {}
        operation_match_fb = re.search(r"operation:\s*([\w_]+)", cleaned_plan_str, re.IGNORECASE)
        if operation_match_fb:
            plan_fallback['operation'] = operation_match_fb.group(1).lower()
            # Add basic keys found via regex (simple key: value lines)
            for key_in_plan in ['column', 'group_by_column', 'group_by_period', 'count_condition_column', 'count_condition_value']:
                 # Use specific regex for each key assuming it's on its own line
                 key_match_regex = rf"^\s*{re.escape(key_in_plan)}:\s*([\w\.'\"\[\]]+(?:,\s*[\w\.'\"\[\]]+)*)" # Allow for lists like output_columns
                 key_val_match = re.search(key_match_regex, cleaned_plan_str, re.IGNORECASE | re.MULTILINE)
                 if key_val_match:
                     raw_val = key_val_match.group(1).strip().strip("'\"")
                     # Special handling for potential list values (like output_columns)
                     if ',' in raw_val or '[' in raw_val or ']' in raw_val:
                          try:
                              # Attempt to parse as a literal list
                              list_val = ast.literal_eval(f"[{raw_val}]") # Wrap in brackets to parse comma-separated
                              plan_fallback[key_in_plan] = [item.strip().strip("'\"") for item in list_val]
                          except (ValueError, SyntaxError):
                              # If it fails, keep as the raw string value
                              plan_fallback[key_in_plan] = raw_val
                     else:
                         plan_fallback[key_in_plan] = raw_val # Keep as single string/value

            # This fallback parsing might be incomplete but indicates a plan attempt
            plan_fallback['parse_error'] = True # Mark as having a parsing issue
            plan_fallback['raw_plan_str_attempt'] = plan_str # Store the original raw string
            if notes_section: plan_fallback['notes_from_llm'] = notes_section
            return plan_fallback
        else:
            # Didn't even find an operation key in the fallback regex search
            return None # Doesn't look like a plan attempt

    # If neither YAML nor the fallback regex found keywords indicating a plan, return None
    return None


def execute_llm_plan(plan: dict, df: pd.DataFrame):
    """Executes a parsed plan dictionary against the DataFrame."""
    # Mapping for day names to pandas dayofweek (Monday=0, Sunday=6)
    day_name_to_dayofweek = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    # Reverse map for displaying messages
    dayofweek_to_day_name = {v: k.capitalize() for k, v in day_name_to_dayofweek.items()}


    try:
        if not plan or 'operation' not in plan:
             # Check if it's a marked parse error from the robust parser
             if plan and plan.get('parse_error'):
                  # This indicates the LLM tried to send a plan, but it was malformed
                  raw_attempt = plan.get('raw_plan_str_attempt', 'N/A') # Use the stored attempt
                  notes = plan.get('notes_from_llm', 'N/A')
                  return f"AI provided a plan, but I had trouble parsing it. AI's attempted plan:\n```\n{raw_attempt}\n```\nAI's Notes: {notes}"
             # Otherwise, it's just an invalid structure passed
             return f"Execution failed: Invalid or incomplete plan structure received. Plan: {plan}"


        operation = plan.get('operation','').lower()
        df_exec = df.copy()

        # Ensure Date column is datetime for filtering
        if 'Date' in df_exec.columns:
            # Coerce errors means invalid parsing results in NaT, which dropna handles
            df_exec['Date'] = pd.to_datetime(df_exec['Date'], errors='coerce')
            # Drop rows where Date became NaT after conversion
            df_exec.dropna(subset=['Date'], inplace=True)
            if df_exec.empty:
                 return "No valid dates found in data after processing."

        filtered_df = df_exec.copy()

        # --- Apply Filters ---
        if 'filters' in plan and isinstance(plan['filters'], list):
            for i, f_item in enumerate(plan['filters']):
                if not (isinstance(f_item, dict) and all(k in f_item for k in ['column','condition','value'])):
                     # Skip malformed filters in the list
                     print(f"Skipping malformed filter item {i}: {f_item}")
                     continue # Move to the next filter item

                col_name, cond, val = f_item['column'], f_item['condition'].lower(), f_item['value']

                # Find the actual column name (case-insensitive)
                actual_col = next((c for c in filtered_df.columns if c.lower()==col_name.lower()), None)
                if not actual_col:
                    return f"Filter column '{col_name}' not found in data."

                # Handle date filters (requires datetime column)
                if actual_col.lower() == 'date':
                    if not pd.api.types.is_datetime64_any_dtype(filtered_df[actual_col]):
                         return f"Cannot apply date filter on '{actual_col}': Column is not in datetime format."
                    try:
                        if cond == 'year_equals':
                            filtered_df = filtered_df[filtered_df[actual_col].dt.year == int(val)].copy() # Use .copy()
                        elif cond == 'month_equals':
                            filtered_df = filtered_df[filtered_df[actual_col].dt.month == int(val)].copy() # Use .copy()
                        elif cond == 'quarter_equals':
                            filtered_df = filtered_df[filtered_df[actual_col].dt.quarter == int(val)].copy() # Use .copy()
                        elif cond == 'dayofweek_equals': # Condition for day of the week
                            # Convert the value (which could be a string name or a number) to the expected integer
                            if isinstance(val, str) and val.lower() in day_name_to_dayofweek:
                                day_int = day_name_to_dayofweek[val.lower()]
                            elif isinstance(val, int) and 0 <= val <= 6:
                                day_int = val
                            else:
                                # Provide more specific error if day name/int is invalid
                                expected_days = ", ".join(day_name_to_dayofweek.keys()) + " or integers 0-6"
                                return f"Invalid value '{val}' for dayofweek_equals condition. Expected day name (e.g., Monday) or integer (0-6)."
                            filtered_df = filtered_df[filtered_df[actual_col].dt.dayofweek == day_int].copy() # Use .copy()
                        # Add other date conditions like day, specific date range, etc. if needed by LLM examples
                        else:
                            return f"Unsupported date filter condition: '{cond}'. Supported: year_equals, month_equals, quarter_equals, dayofweek_equals."
                    except ValueError:
                        # Catch errors if int(val) fails for year/month/quarter
                        return f"Invalid value '{val}' for date filter condition '{cond}'. Expected an integer."
                    except Exception as e:
                        # Catch any other unexpected errors during date filtering
                        return f"Error applying date filter on '{actual_col}' with condition '{cond}' and value '{val}': {e}"

                # Handle other column filters (comparison operators)
                # Attempt numeric comparison first if column is numeric, fallback to string if needed/applicable
                is_numeric_col = pd.api.types.is_numeric_dtype(filtered_df[actual_col])

                if cond == 'equals':
                    # Case-insensitive string comparison for object/string columns and 'direction'
                    if not is_numeric_col or actual_col.lower() == 'direction': # Prioritize string for direction
                        filtered_df = filtered_df[filtered_df[actual_col].astype(str).str.lower() == str(val).lower()].copy() # Use .copy()
                    else: # Numeric or potentially mixed types, try numeric first
                        try:
                            # Attempt numeric comparison
                            numeric_val = pd.to_numeric(val, errors='raise') # Raise error if conversion fails
                            filtered_df = filtered_df[pd.to_numeric(filtered_df[actual_col], errors='coerce') == numeric_val].copy() # Use .copy(), coerce col to numeric
                        except (ValueError, TypeError):
                             # Fallback to string comparison if numeric conversion of value fails or column isn't purely numeric after coerce
                             filtered_df = filtered_df[filtered_df[actual_col].astype(str).str.lower() == str(val).lower()].copy() # Use .copy()

                elif cond in ['greater_than', 'gt', '>']:
                     if not is_numeric_col: return f"Cannot apply '{cond}' filter on non-numeric column '{actual_col}'."
                     try: numeric_val = pd.to_numeric(val, errors='raise') # Raise error if conversion fails
                     except (ValueError, TypeError): return f"Invalid numeric value '{val}' for condition '{cond}' on column '{actual_col}'."
                     filtered_df = filtered_df[pd.to_numeric(filtered_df[actual_col], errors='coerce') > numeric_val].copy() # Use .copy()
                elif cond in ['less_than', 'lt', '<']:
                     if not is_numeric_col: return f"Cannot apply '{cond}' filter on non-numeric column '{actual_col}'."
                     try: numeric_val = pd.to_numeric(val, errors='raise')
                     except (ValueError, TypeError): return f"Invalid numeric value '{val}' for condition '{cond}' on column '{actual_col}'."
                     filtered_df = filtered_df[pd.to_numeric(filtered_df[actual_col], errors='coerce') < numeric_val].copy() # Use .copy()
                elif cond in ['greater_than_or_equals', 'gte', '>=']:
                     if not is_numeric_col: return f"Cannot apply '{cond}' filter on non-numeric column '{actual_col}'."
                     try: numeric_val = pd.to_numeric(val, errors='raise')
                     except (ValueError, TypeError): return f"Invalid numeric value '{val}' for condition '{cond}' on column '{actual_col}'."
                     filtered_df = filtered_df[pd.to_numeric(filtered_df[actual_col], errors='coerce') >= numeric_val].copy() # Use .copy()
                elif cond in ['less_than_or_equals', 'lte', '<=']:
                     if not is_numeric_col: return f"Cannot apply '{cond}' filter on non-numeric column '{actual_col}'."
                     try: numeric_val = pd.to_numeric(val, errors='raise')
                     except (ValueError, TypeError): return f"Invalid numeric value '{val}' for condition '{cond}' on column '{actual_col}'."
                     # Corrected the syntax error here: escaped the double quote at the end of the literal string
                     filtered_df = filtered_df[pd.to_numeric(filtered_df[actual_col], errors='coerce') <= numeric_val].copy() # Use .copy()

                # Add other conditions like 'between', 'isin', etc. if needed by LLM examples
                else:
                    # Corrected the syntax error here: escaped the double quote at the end of the literal string
                    return f"Unsupported filter condition: '{cond}' on column '{actual_col}'. Supported for non-date columns: equals, greater_than/gt/>, less_than/lt/<, greater_than_or_equals/gte/>=, less_than_or_equals/lte/<=\". Corrected Error Message."


        # Check if any data remains after filtering
        if filtered_df.empty:
            # Provide a more specific message if key filters were applied
            messages = ["No data found matching the specified filters."]
            # Check for specific filters applied
            year_val = next((f['value'] for f in plan.get('filters', []) if isinstance(f,dict) and f.get('column','').lower() == 'date' and f.get('condition') == 'year_equals'), None)
            if year_val is not None: messages.insert(0, f"For year {year_val}:")

            day_val_filter = next((f['value'] for f in plan.get('filters', []) if isinstance(f,dict) and f.get('column','').lower() == 'date' and f.get('condition') == 'dayofweek_equals'), None)
            if day_val_filter is not None:
                 day_name = day_val_filter.lower() if isinstance(day_val_filter, str) else dayofweek_to_day_name.get(day_val_filter, str(day_val_filter))
                 if messages[0].startswith("For year"): messages[0] += f" on {day_name.capitalize()}s:"
                 else: messages.insert(0, f"For {day_name.capitalize()}s:")
            # Can add checks for other specific filters too

            return " ".join(messages)


        # --- Execute Operation ---
        if operation == 'count_rows':
            return f"Matching days: **{len(filtered_df)}**."

        # Operations requiring a target column
        col_plan = plan.get('column')
        if operation in ['find_max', 'find_max_with_date', 'find_min_with_date', 'calculate_average']:
            if not col_plan:
                 return f"Cannot perform '{operation}': Missing 'column' parameter in plan."

            actual_col_op = next((c for c in filtered_df.columns if c.lower()==col_plan.lower()),None)
            if not actual_col_op:
                 return f"Cannot perform '{operation}': Target column '{col_plan}' not found."


            if not pd.api.types.is_numeric_dtype(filtered_df[actual_col_op]):
                 return f"Cannot perform '{operation}': Target column '{actual_col_op}' is not numeric."

            # Ensure there's valid numeric data after filtering and column selection
            valid_data_series = filtered_df[actual_col_op].dropna()
            if valid_data_series.empty:
                 return f"No valid numeric data for '{actual_col_op}' after applying filters for operation '{operation}'."


            if operation=='find_max':
                val_op = valid_data_series.max()
                return f"Highest '{actual_col_op}': **{val_op:.2f}**."

            if operation=='find_max_with_date':
                val_op = valid_data_series.max()
                # Find the row(s) where the max value occurs
                # Use .loc[valid_data_series.idxmax()] to get the specific row index if multiple max values exist
                try:
                    max_row_idx = valid_data_series.idxmax()
                    rows_op = filtered_df.loc[[max_row_idx]].copy()
                except ValueError: # Happens if valid_data_series is empty (should be caught above) or other issue
                    # This might still happen if the max value is NaN, though valid_data_series excludes NaNs
                    return f"Found max {val_op:.2f} for '{actual_col_op}', but could not find matching date row (idxmax failed)."


                if not rows_op.empty:
                    # Take the first date if multiple rows have the max value
                    date_op = rows_op['Date'].iloc[0].strftime('%Y-%m-%d')
                    res_str = f"Highest '{actual_col_op}': **{val_op:.2f}** on **{date_op}**."
                    # Optionally add other columns from the output_columns list
                    output_cols_plan = plan.get('output_columns', [])
                    details = []
                    for ocp in output_cols_plan:
                        actual_out_col = next((c for c in rows_op.columns if c.lower() == ocp.lower()), None)
                        # Avoid repeating the primary column or date unless specifically requested and different
                        if actual_out_col and actual_out_col.lower() != actual_col_op.lower() and actual_out_col.lower() != 'date':
                            # Add type-aware formatting for details
                             if actual_out_col in rows_op.columns and not rows_op[actual_out_col].empty:
                                first_val = rows_op[actual_out_col].iloc[0]
                                if pd.notna(first_val): # Check if the value is not NaN
                                    if pd.api.types.is_numeric_dtype(rows_op[actual_out_col]):
                                        details.append(f"{actual_out_col}: {first_val:.2f}")
                                    elif isinstance(first_val, pd.Timestamp):
                                        details.append(f"{actual_out_col}: {first_val.strftime('%Y-%m-%d')}")
                                    else:
                                        details.append(f"{actual_col}: {first_val}")


                    if details: res_str += " Other details: " + "; ".join(details)
                    return res_str
                # Should not happen if valid_data_series is not empty and idxmax works, but as a safeguard
                return f"Found max {val_op:.2f} for '{actual_col_op}', but could not retrieve matching date row data."


            if operation=='find_min_with_date':
                val_op = valid_data_series.min()
                 # Find the row(s) where the min value occurs
                # Use .copy()
                # Use .loc[valid_data_series.idxmin()] to get the specific row index
                try:
                     min_row_idx = valid_data_series.idxmin()
                     rows_op = filtered_df.loc[[min_row_idx]].copy()
                except ValueError: # Happens if valid_data_series is empty (should be caught above) or other issue
                     return f"Found min {val_op:.2f} for '{actual_col_op}', but could not find matching date row (idxmin failed)."


                if not rows_op.empty:
                    # Take the first date if multiple rows have the min value
                    date_op = rows_op['Date'].iloc[0].strftime('%Y-%m-%d')
                    res_str = f"Lowest '{actual_col_op}': **{val_op:.2f}** on **{date_op}**."
                    # Optionally add other columns from the output_columns list
                    output_cols_plan = plan.get('output_columns', [])
                    details = []
                    for ocp in output_cols_plan:
                        actual_out_col = next((c for c in rows_op.columns if c.lower() == ocp.lower()), None)
                         # Avoid repeating the primary column or date unless specifically requested and different
                        if actual_out_col and actual_out_col.lower() != actual_col_op.lower() and actual_out_col.lower() != 'date':
                            # Add type-aware formatting
                             if actual_out_col in rows_op.columns and not rows_op[actual_out_col].empty:
                                first_val = rows_op[actual_out_col].iloc[0]
                                if pd.notna(first_val): # Check if the value is not NaN
                                    if pd.api.types.is_numeric_dtype(rows_op[actual_out_col]):
                                        details.append(f"{actual_out_col}: {first_val:.2f}")
                                    elif isinstance(first_val, pd.Timestamp):
                                        details.append(f"{actual_out_col}: {first_val.strftime('%Y-%m-%d')}")
                                    else:
                                        details.append(f"{actual_col}: {first_val}")
                    if details: res_str += " Other details: " + "; ".join(details)
                    return res_str
                 # Should not happen if valid_data_series is not empty and idxmin works, but as a safeguard
                return f"Found min {val_op:.2f} for '{actual_col_op}', but could not retrieve matching date row data."


            if operation=='calculate_average':
                avg_val = valid_data_series.mean()
                return f"Average '{actual_col_op}': **{avg_val:.2f}**."

        # --- Grouping and Counting Operation ---
        if operation == 'find_group_with_max_count':
            group_col_plan = plan.get('group_by_column');
            actual_group_col = next((c for c in filtered_df.columns if c.lower()==group_col_plan.lower()),None) if group_col_plan else None
            group_period = plan.get('group_by_period','').lower()
            count_col_plan = plan.get('count_condition_column');
            actual_count_col = next((c for c in filtered_df.columns if c.lower()==count_col_plan.lower()),None) if count_col_plan else None
            count_val_plan = plan.get('count_condition_value')

            # Validate plan parameters for this operation
            supported_periods = ['month', 'dayofweek']
            if not (actual_group_col and pd.api.types.is_datetime64_any_dtype(filtered_df[actual_group_col]) and group_period in supported_periods and actual_count_col is not None and count_val_plan is not None):
                return f"Cannot execute 'find_group_with_max_count': Missing or invalid plan parameters. Required: 'group_by_column' (datetime), 'group_by_period' ({'/'.join(supported_periods)}), 'count_condition_column', 'count_condition_value'. Plan: {plan}"

            # Filter for rows matching the count condition
            try:
                # Check if the count_condition_column actually exists and is not empty after initial filters
                if actual_count_col not in filtered_df.columns or filtered_df[actual_count_col].empty:
                     return f"Cannot execute 'find_group_with_max_count': Count condition column '{count_col_plan}' not found or is empty after initial filters."

                # Ensure comparison is case-insensitive for strings, handle potential type mismatches gracefully
                # Use .astype(str) safely and compare lowercase
                condition_df = filtered_df[filtered_df[actual_count_col].astype(str).str.lower() == str(count_val_plan).lower()].copy()

            except KeyError as e:
                 # This case should be less likely with the check above, but keep it as safeguard
                 return f"Error: Column '{e.args[0]}' specified in count_condition_column not found in DataFrame. Plan: {plan}"
            except Exception as e:
                 return f"An unexpected error occurred while filtering for count condition: {e}. Plan: {plan}"


            if condition_df.empty:
                 # Include context about how many days matched the *main* filters
                 return f"No days found where '{actual_count_col}' was '{count_val_plan}' in the filtered period ({len(filtered_df)} total days matched main filters)."

            # Group by the specified date period
            try:
                if group_period == 'month':
                    condition_df['grouping_key'] = condition_df[actual_group_col].dt.strftime('%B') # Format as full month name
                    group_counts = condition_df.groupby('grouping_key').size()
                     # Define order for sorting month names chronologically
                    group_names_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                         'July', 'August', 'September', 'October', 'November', 'December']
                elif group_period == 'dayofweek':
                     condition_df['grouping_key'] = condition_df[actual_group_col].dt.day_name() # Format as full day name
                     group_counts = condition_df.groupby('grouping_key').size()
                     # Define order for sorting day names chronologically
                     group_names_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            except Exception as e:
                return f"An unexpected error occurred during grouping by {group_period}: {e}. Plan: {plan}"


            if group_counts.empty:
                 # This case should be covered by condition_df.empty check, but as safeguard
                 return f"No '{count_val_plan}' days found to group by {group_period} in the filtered period."

            # Find the group(s) with the maximum count
            max_count = group_counts.max()
            best_group_names = group_counts[group_counts == max_count].index.tolist()

            # Sort the results based on the defined order
            # Use a lambda that returns the index if found, otherwise a large number to put unfound items at the end
            best_group_names_sorted = sorted(best_group_names, key=lambda name: group_names_order.index(name) if name in group_names_order else len(group_names_order))


            if len(best_group_names_sorted) == 1:
                 group_type_str = "month" if group_period == 'month' else "weekday" if group_period == 'dayofweek' else group_period
                 return f"The {group_type_str} with the most '{count_val_plan}' days was **{best_group_names_sorted[0]}** with **{max_count}** day(s)."
            else:
                 group_type_str = "months" if group_period == 'month' else "weekdays" if group_period == 'dayofweek' else group_period
                 return f"The {group_type_str}s with the most '{count_val_plan}' days (tied at **{max_count}** days each) were: **{', '.join(best_group_names_sorted)}**."


        # --- Fallback/Default Response if operation is not recognized ---
        # If we reached here, it means the parsed plan had an operation, but it's not one we handle
        notes_from_llm = plan.get('notes_from_llm', '')
        # Use yaml.dump for potentially complex plan dictionaries in the response
        plan_yaml_dump = "N/A"
        try: plan_yaml_dump = yaml.dump(plan, indent=2, default_flow_style=False)
        except Exception: pass

        response_str = f"I received a plan for operation '{operation}' from the AI, but I don't have specific execution logic coded for this operation type or parameters. Here's the plan from the AI:\n```yaml\n{plan_yaml_dump}\n```"
        if notes_from_llm: response_str += f"\n\nAI's Notes: {notes_from_llm}"
        # Consider adding a more specific message if the operation name itself is unknown

        return response_str

    except Exception as e:
        # Catch any unexpected errors during execution
        # Use yaml.dump for potentially complex plan dictionaries in the error message
        plan_yaml_dump = "N/A"
        try: plan_yaml_dump = yaml.dump(plan, indent=2, default_flow_style=False)
        except Exception: pass
        raw_plan_str = plan.get('raw_plan_str_attempt', plan.get('raw_plan_str', 'N/A')) # Prefer the raw attempt if parse_error was set
        return f"Error executing plan: {e}. Plan: {plan_yaml_dump}. Raw: {raw_plan_str}"


with tab2:
    st.header("🤖 AI-Powered Data Insights")
    if not st.session_state.gemini_model: st.warning("Gemini AI model not available.")
    elif data_df.empty: st.warning("Stock data not loaded for chatbot.")
    else:
        # Added checks for required columns for specific queries
        missing_cols_warnings = []
        if 'Date' not in data_df.columns: missing_cols_warnings.append('Date (critical)')
        if not all(col in data_df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            missing_cols_warnings.append('OHLCV (for price/volume)')
        if 'direction' not in data_df.columns:
            missing_cols_warnings.append('direction (for bullish/bearish/neutral)') # Clarify for Neutral
        if 'Intraday_Range' not in data_df.columns:
             missing_cols_warnings.append('Intraday_Range (derived)')
        if 'Support_Lower' not in data_df.columns or 'Support_Upper' not in data_df.columns:
             missing_cols_warnings.append('Support levels')
        if 'Resistance_Lower' not in data_df.columns or 'Resistance_Upper' not in data_df.columns:
             missing_cols_warnings.append('Resistance levels')


        if missing_cols_warnings:
             st.warning(f"Some expected data points are missing: {', '.join(missing_cols_warnings)}. Certain AI queries may not work correctly.")

        # Reduced the number of example questions in the info box to 3 and removed the intro text
        st.info("Examples: 'How many bullish days in 2023?', 'Highest Close in 2022 and date?', 'What was the average high price on Mondays in 2023?'")

        if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
        for msg in st.session_state.chat_messages: st.chat_message(msg["role"]).markdown(msg["content"])

        if prompt := st.chat_input("Your question about TSLA data..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            final_response_to_display = "Could not process the request."
            llm_response_was_textual = False # Flag to indicate if the LLM response was pure text


            # --- Python Direct Calculation Attempt FIRST ---
            # This section handles simple, common queries quickly without involving the LLM
            # Keep these for performance on very simple queries, but LLM fallback is primary
            python_calculated_answer = None

            # Pattern 1: "How many bullish/LONG days in [YEAR]?"
            year_bullish_match = re.search(r"how many\s+(bullish|long)\s+days\s+in\s+(\d{4})", prompt.lower())
            # Ensure 'direction' column exists for this calculation
            if year_bullish_match and all(c in data_df.columns for c in ['Date', 'direction']) and pd.api.types.is_datetime64_any_dtype(data_df['Date']):
                target_year = int(year_bullish_match.group(2))
                try:
                    df_year = data_df[data_df['Date'].dt.year == target_year].copy()
                    # Do NOT dropna here if we want to count NULL/None as neutral
                    # df_year.dropna(subset=['direction'], inplace=True) # Count only non-None/NaN directions
                    if not df_year.empty:
                        # Filter for 'LONG' case-insensitively
                        count = df_year[df_year['direction'].astype(str).str.upper() == 'LONG'].shape[0] # Use upper for consistency
                        python_calculated_answer = f"In {target_year}, there were **{count}** bullish ('LONG') days."
                    else:
                         # Check if any data for the year exists at all before saying no data for bullish days
                         df_year_all = data_df[data_df['Date'].dt.year == target_year]
                         if not df_year_all.empty:
                              python_calculated_answer = f"In {target_year}, no days were marked as 'LONG' in the 'direction' column (or column missing/empty)."
                         else:
                              python_calculated_answer = f"No data found for {target_year}."
                except Exception as e:
                    # Don't set python_calculated_answer, let it fall through to LLM if error happens
                    print(f"Python calculation error for bullish days: {e}") # Print error for debugging
                    pass # Continue to LLM path


            # Pattern 2: "How many days in [YEAR] did TSLA close lower than it opened?" (Bearish days by price action)
            if not python_calculated_answer: # Only attempt this if the previous pattern didn't provide an answer
                year_closed_lower_match = re.search(r"how many days in (\d{4}).*close lower than.*opened", prompt.lower())
                # Ensure required columns exist and are numeric/datetime
                if year_closed_lower_match and all(c in data_df.columns for c in ['Date', 'Open', 'Close']) and \
                   pd.api.types.is_datetime64_any_dtype(data_df['Date']) and \
                   pd.api.types.is_numeric_dtype(data_df['Open']) and pd.api.types.is_numeric_dtype(data_df['Close']):
                    target_year_cl = int(year_closed_lower_match.group(1))
                    try:
                        df_year_cl = data_df[data_df['Date'].dt.year == target_year_cl].copy()
                        df_year_cl.dropna(subset=['Open', 'Close'], inplace=True) # Need both for comparison
                        if not df_year_cl.empty:
                            closed_lower_count = df_year_cl[df_year_cl['Close'] < df_year_cl['Open']].shape[0]
                            python_calculated_answer = f"In {target_year_cl}, TSLA closed lower than it opened on **{closed_lower_count}** days."
                        else:
                            df_year_all = data_df[data_df['Date'].dt.year == target_year_cl]
                            if not df_year_all.empty:
                                python_calculated_answer = f"No valid 'Open' and 'Close' data found for {target_year_cl} to compare."
                            else:
                                python_calculated_answer = f"No data found for {target_year_cl}."

                    except Exception as e:
                         # Don't set python_calculated_answer
                         print(f"Python calculation error for closed lower days: {e}") # Print error for debugging
                         pass # Continue to LLM path

            # --- End Python Direct Calculation Attempt ---

            if python_calculated_answer:
                final_response_to_display = python_calculated_answer
            else: # Fallback to LLM for planning if Python tools didn't handle it
                with st.spinner("🤖 AI is planning..."):
                    # Pass the full dataframe, let the LLM know the schema and range
                    llm_plan_str = get_gemini_response(st.session_state.gemini_model, data_df, prompt)

                if llm_plan_str:
                    # Attempt to parse the plan received from the LLM
                    parsed_plan = parse_llm_plan_robustly(llm_plan_str)

                    if parsed_plan:
                        # Store the raw plan string for debugging if needed
                        parsed_plan['raw_plan_str'] = llm_plan_str
                        with st.spinner("⚙️ Python is executing AI plan..."):
                            # Execute the parsed plan using the data
                            # Pass a copy to avoid modifying original, important if execute_llm_plan adds columns (though data_loader does that now)
                            final_response_to_display = execute_llm_plan(parsed_plan, data_df.copy())

                        # Check if the execution response contains known error phrases from execute_llm_plan
                        known_error_phrases = [
                            "Execution failed:", "Cannot execute", "Invalid or incomplete plan structure",
                            "Filter column", "Target column", "Cannot perform", "No valid data",
                            "Unsupported date filter condition", "Invalid value", "Error applying date filter",
                            "Unsupported filter condition", "No data found matching", "Missing 'column' parameter",
                            "Column", "An unexpected error occurred", "Could not retrieve matching date row data",
                            "Could not find matching date row", "Error executing plan:", # Catch the general execution error too
                            "AI provided a plan, but I had trouble parsing it." # Error specifically from parse_llM_plan_robustly
                        ]
                        is_execution_error_message = any(phrase.lower() in final_response_to_display.lower() for phrase in known_error_phrases)

                        # If it looks like a plan attempt that failed parsing, display as warning
                        if parsed_plan.get('parse_error'):
                             st.chat_message("assistant").warning(final_response_to_display)
                        # If the executed plan returned a specific error message, display as warning
                        elif is_execution_error_message:
                             st.chat_message("assistant").warning(final_response_to_display)
                        # Otherwise, display the successful execution result
                        else:
                            st.chat_message("assistant").markdown(final_response_to_display)

                    else:
                        # Parsing failed, AND it didn't look like an attempted plan structure (parse_llm_plan_robustly returned None).
                        # Treat the raw LLM response as a direct textual response.
                        final_response_to_display = f"AI Response:\n```\n{llm_plan_str.strip()}\n```"
                        st.chat_message("assistant").markdown(final_response_to_display)
                        llm_response_was_textual = True # Redundant now, but kept for clarity


                else:
                    # If the LLM returned nothing or an error from the API call itself
                    final_response_to_display = "The AI could not generate a plan or response for this query."
                    st.chat_message("assistant").warning(final_response_to_display) # Treat API errors as warnings


            # Append message to history AFTER displaying it
            # This prevents error messages from being added as "assistant" messages if the app reruns
            # The displaying logic is now within the if/else block above.

            # Add the final response content to the session state history
            # Check if the last message in history is from the user. If so, add the assistant response.
            # This handles potential reruns not adding duplicates.
            if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
                 st.session_state.chat_messages.append({"role": "assistant", "content": final_response_to_display})
            # If the last message was *not* a user message (e.g. on a rerun), we might still update the content of the last assistant message
            # This is more complex and often not needed. For simplicity, we rely on the check above.


with tab_bonus:
    st.header("🎬 Market Replay")
    if data_df.empty:
        st.warning("No data for animation.")
    else:
        max_anim_frames = len(data_df)

        # Use a unique prefix for session state keys in this tab
        prefix = "replay_tv_pro_v13_"

        # Initialize session state variables specific to this replay instance
        if f'{prefix}active' not in st.session_state:
            st.session_state[f'{prefix}active'] = False # Is the animation running?
        if f'{prefix}index' not in st.session_state:
            st.session_state[f'{prefix}index'] = 0 # Current frame index
        if f'{prefix}speed' not in st.session_state:
            st.session_state[f'{prefix}speed'] = 0.25 # Delay between frames
        if f'{prefix}last_fig' not in st.session_state:
            st.session_state[f'{prefix}last_fig'] = None # Store the last generated figure

        # --- Animation Controls ---
        current_speed = st.session_state[f'{prefix}speed']
        new_speed = st.slider(
            "Replay Speed (seconds per frame)",
            0.05, 2.0, current_speed, 0.05, # Adjusted range for speed
            key=f"{prefix}speed_slider",
            help="Adjust animation speed (higher value is slower)"
        )
        # Update speed if the slider changed
        if new_speed != current_speed:
            st.session_state[f'{prefix}speed'] = new_speed
            # No rerun needed here, the value is updated for the next loop iteration


        cols_ctrl = st.columns(3)
        # Start/Resume button
        if cols_ctrl[0].button("▶️ Start / Resume", key=f"{prefix}start_btn", use_container_width=True):
            st.session_state[f'{prefix}active'] = True
            # If finished, reset index to 0 to start over
            if st.session_state[f'{prefix}index'] >= max_anim_frames:
                st.session_state[f'{prefix}index'] = 0
                st.session_state[f'{prefix}last_fig'] = None # Clear the last figure if starting over
            st.rerun() # Trigger rerun to enter the animation loop

        # Pause button
        if cols_ctrl[1].button("⏸️ Pause", key=f"{prefix}pause_btn", use_container_width=True):
            st.session_state[f'{prefix}active'] = False
            # st.rerun() # Rerun to ensure the paused state is displayed immediately

        # Reset button
        if cols_ctrl[2].button("⏹️ Reset", key=f"{prefix}reset_btn", use_container_width=True):
            st.session_state[f'{prefix}active'] = False # Stop animation
            st.session_state[f'{prefix}index'] = 0 # Reset index to start
            st.session_state[f'{prefix}last_fig'] = None # Clear stored figure
            # Clear the chart display area explicitly before rerun
            if f"{prefix}placeholder" in st.session_state and st.session_state[f"{prefix}placeholder"] is not None:
                 st.session_state[f"{prefix}placeholder"].empty()
            else:
                 # Create and empty a placeholder if it wasn't in session_state yet
                 st.empty().empty()
            st.rerun() # Trigger rerun to show the reset state

        # --- Progress Bar and Chart Placeholder ---
        # Define placeholder consistently before it's used
        if f"{prefix}placeholder" not in st.session_state:
            st.session_state[f"{prefix}placeholder"] = st.empty()

        # Use the direct variable for the current script run's placeholder instance
        chart_display_area = st.session_state[f"{prefix}placeholder"]

        progress_bar_replay = st.progress(0.0)
        current_progress = float(st.session_state[f'{prefix}index']) / max_anim_frames if max_anim_frames > 0 else 0
        progress_bar_replay.progress(min(current_progress, 1.0)) # Ensure it doesn't exceed 1.0


        # --- Animation Logic ---
        if st.session_state[f'{prefix}active'] and st.session_state[f'{prefix}index'] < max_anim_frames:
            current_idx = st.session_state[f'{prefix}index']
            slice_end_idx = current_idx + 1 # Include the current bar

            # Determine the start index for the animated slice to show historical context
            # Show at least 1 bar, up to the last 100 bars plus the current bar
            min_context = max(0, slice_end_idx - 100)
            # Ensure we show at least the first few bars if the slice is very short
            if slice_end_idx < 20:
                 min_context = 0 # Start from the beginning for the first 20 bars


            animated_df_slice = data_df.iloc[min_context:slice_end_idx]

            if not animated_df_slice.empty:
                current_display_date = data_df['Date'].iloc[current_idx]
                anim_title = f"Replay: {current_display_date.strftime('%Y-%m-%d')} (Bar {current_idx+1}/{max_anim_frames})"
                # Create the chart for the current frame's data slice
                anim_fig = create_tv_style_plotly_chart(animated_df_slice.copy(), chart_title=anim_title, chart_height=600, is_animation_frame=True)

                if anim_fig:
                    # Manually set y-axis range to be constant across animation for better comparison
                    # Use the min/max of the *full* dataset for the y-axis range
                    full_min_low = data_df['Low'].min()
                    full_max_high = data_df['High'].max()
                    padding = (full_max_high - full_min_low) * 0.05 # Add a little padding
                    anim_fig.update_yaxes(range=[full_min_low - padding, full_max_high + padding], autorange=False, row=1, col=1)

                    # Store the generated figure in session state
                    st.session_state[f'{prefix}last_fig'] = anim_fig
                    # Display the figure using the placeholder
                    chart_display_area.plotly_chart(anim_fig, use_container_width=True, key=f"{prefix}chart_active_{current_idx}")
                else:
                    # Handle case where chart creation failed for this frame
                     chart_display_area.warning(f"Could not generate chart for {current_display_date.strftime('%Y-%m-%d')}")


            st.session_state[f'{prefix}index'] += 1 # Move to the next frame

            # Check if the animation has reached the end after incrementing the index
            if st.session_state[f'{prefix}index'] >= max_anim_frames:
                st.session_state[f'{prefix}active'] = False # Stop the animation
                st.success("Replay Finished!")
                # Update progress bar one last time to full
                progress_bar_replay.progress(1.0)
                # No immediate rerun here; the last_fig will be shown by the block below

            # If still active and not at the end, wait and rerun
            elif st.session_state[f'{prefix}active']:
                time.sleep(st.session_state[f'{prefix}speed'])
                st.rerun() # Rerun to draw the next frame

        # --- Logic for Paused or Finished State (Displaying the last rendered figure) ---
        # If the animation is not active OR it has finished, and we have a last figure saved, display it
        elif st.session_state[f'{prefix}last_fig'] is not None:
            # Display the stored figure using the placeholder
            chart_display_area.plotly_chart(st.session_state[f'{prefix}last_fig'], use_container_width=True, key=f"{prefix}chart_paused_or_done")
            # Display status messages if paused or finished
            if not st.session_state[f'{prefix}active'] and st.session_state[f'{prefix}index'] > 0 and st.session_state[f'{prefix}index'] < max_anim_frames:
                 st.info(f"Replay paused at bar {st.session_state[f'{prefix}index']} of {max_anim_frames}.")
            # If it finished, the success message is handled above in the active block

        # --- Initial State Message (if no figure has been rendered yet and not active) ---
        if not st.session_state[f'{prefix}active'] and st.session_state[f'{prefix}index'] == 0 and st.session_state[f'{prefix}last_fig'] is None:
             chart_display_area.info("Click 'Start / Resume' to begin animation replay.")
             progress_bar_replay.progress(0.0) # Ensure progress bar is at 0