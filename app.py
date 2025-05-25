import streamlit as st
import pandas as pd
import time 
import re 
import yaml 

from data_loader import load_data
from chart_generator import create_tv_style_plotly_chart 
from chatbot import configure_gemini_model, get_gemini_response 

st.set_page_config(layout="wide", page_title="Pro TSLA Dashboard", initial_sidebar_state="expanded")

DATA_FILE_PATH = "TSLA_data.csv" 
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
    plan = {}
    cleaned_plan_str = plan_str
    if "PLAN:" in cleaned_plan_str.upper(): cleaned_plan_str = cleaned_plan_str.split("PLAN:", 1)[1].strip()
    notes_section = ""; 
    if "notes:" in cleaned_plan_str.lower(): 
        parts = re.split(r"\n\s*notes:", cleaned_plan_str, flags=re.IGNORECASE, maxsplit=1)
        cleaned_plan_str = parts[0].strip(); 
        if len(parts) > 1: notes_section = parts[1].strip()
    try:
        parsed_yaml = yaml.safe_load(cleaned_plan_str)
        if isinstance(parsed_yaml, dict) and 'operation' in parsed_yaml:
            if notes_section: parsed_yaml['notes_from_llm'] = notes_section
            return parsed_yaml
    except: pass 
    operation_match = re.search(r"operation:\s*([\w_]+)", cleaned_plan_str, re.IGNORECASE)
    if operation_match: plan['operation'] = operation_match.group(1).lower()
    else: return None 
    column_match = re.search(r"^\s*column:\s*([\w\.'\"\[\]]+)", cleaned_plan_str, re.IGNORECASE | re.MULTILINE)
    if column_match: plan['column'] = column_match.group(1).strip("'\"")
    output_columns_match = re.search(r"output_columns:\s*\[([^\]]+)\]", cleaned_plan_str, re.IGNORECASE)
    if output_columns_match: plan['output_columns'] = [col.strip().strip("'\"") for col in output_columns_match.group(1).split(',')]
    for key_in_plan in ['group_by_column', 'group_by_period', 'count_condition_column', 'count_condition_value']:
        key_match_regex = rf"^\s*{key_in_plan}:\s*([\w\.'\"\[\]]+)"
        key_val_match = re.search(key_match_regex, cleaned_plan_str, re.IGNORECASE | re.MULTILINE)
        if key_val_match: plan[key_in_plan] = key_val_match.group(1).strip().strip("'\"")
    filters_list = []
    filters_section_match = re.search(r"filters:((\s*-[\s\S]*?)(?=\n\w+:|\Z))", cleaned_plan_str, re.IGNORECASE | re.DOTALL)
    if filters_section_match:
        filters_block_text = filters_section_match.group(1).strip()
        individual_filter_texts = re.split(r"\n\s*-\s+", "- " + filters_block_text) 
        for item_text in individual_filter_texts:
            item_text = item_text.strip()
            if not item_text or not item_text.startswith('-'): item_text = "- " + item_text 
            item_text_cleaned = item_text.lstrip('-').strip()
            if not item_text_cleaned: continue
            filter_item_dict = {}
            col_m = re.search(r"column:\s*([\w\.'\"\[\]]+)", item_text_cleaned, re.IGNORECASE)
            cond_m = re.search(r"condition:\s*([\w_]+)", item_text_cleaned, re.IGNORECASE)
            val_m = re.search(r"value:\s*(.+)", item_text_cleaned, re.IGNORECASE) 
            if col_m: filter_item_dict['column'] = col_m.group(1).strip().strip("'\"")
            if cond_m: filter_item_dict['condition'] = cond_m.group(1).strip().lower()
            if val_m: 
                val_raw = val_m.group(1).strip().strip("'\"")
                try: processed_val = int(val_raw) 
                except ValueError: processed_val = val_raw
                filter_item_dict['value'] = processed_val
            if 'column' in filter_item_dict and 'condition' in filter_item_dict and 'value' in filter_item_dict:
                 filters_list.append(filter_item_dict)
    if filters_list: plan['filters'] = filters_list
    if notes_section: plan['notes_from_llm'] = notes_section
    return plan

def execute_llm_plan(plan: dict, df: pd.DataFrame):
    try:
        if not plan or 'operation' not in plan: return f"Execution failed: Invalid plan. Plan: {plan}"
        operation = plan.get('operation','').lower(); df_exec = df.copy()
        if 'Date' in df_exec.columns and not pd.api.types.is_datetime64_any_dtype(df_exec['Date']):
            df_exec['Date'] = pd.to_datetime(df_exec['Date'], errors='coerce'); df_exec.dropna(subset=['Date'], inplace=True)
        filtered_df = df_exec.copy()
        if 'filters' in plan and isinstance(plan['filters'], list):
            for f_item in plan['filters']:
                if isinstance(f_item, dict) and all(k in f_item for k in ['column','condition','value']):
                    col, cond, val = f_item['column'], f_item['condition'].lower(), f_item['value']
                    actual_col = next((c for c in filtered_df.columns if c.lower()==col.lower()), None)
                    if not actual_col: return f"Filter column '{col}' not found."
                    if cond == 'year_equals':
                        if not pd.api.types.is_datetime64_any_dtype(filtered_df[actual_col]): return f"Cannot filter by year: '{actual_col}' not datetime."
                        filtered_df = filtered_df[filtered_df[actual_col].dt.year == int(val)]
                    elif cond == 'month_equals':
                        if not pd.api.types.is_datetime64_any_dtype(filtered_df[actual_col]): return f"Cannot filter by month: '{actual_col}' not datetime."
                        filtered_df = filtered_df[filtered_df[actual_col].dt.month == int(val)]
                    elif cond == 'quarter_equals':
                        if not pd.api.types.is_datetime64_any_dtype(filtered_df[actual_col]): return f"Cannot filter by quarter: '{actual_col}' not datetime."
                        filtered_df = filtered_df[filtered_df[actual_col].dt.quarter == int(val)]
                    elif cond == 'equals':
                        if filtered_df[actual_col].dtype=='object' or actual_col.lower()=='direction': filtered_df = filtered_df[filtered_df[actual_col].astype(str).str.lower() == str(val).lower()]
                        else: 
                            try: filtered_df = filtered_df[pd.to_numeric(filtered_df[actual_col],errors='coerce') == pd.to_numeric(val,errors='coerce')]
                            except: filtered_df = filtered_df[filtered_df[actual_col].astype(str).str.lower() == str(val).lower()]
        if filtered_df.empty and 'filters' in plan and plan['filters']:
            year_val = next((f['value'] for f in plan['filters'] if isinstance(f,dict) and f.get('column','').lower() == 'date' and f.get('condition') == 'year_equals'), None)
            if year_val: return f"No data found for year {year_val} after applying all filters."
            return "No data found matching all filters."
        if operation == 'count_rows': return f"Matching days: **{len(filtered_df)}**."
        col_plan = plan.get('column'); actual_col_op = next((c for c in filtered_df.columns if c.lower()==col_plan.lower()),None) if col_plan else None
        if operation in ['find_max', 'find_max_with_date', 'find_min_with_date', 'calculate_average']:
            if not actual_col_op or not pd.api.types.is_numeric_dtype(filtered_df[actual_col_op]): return f"Cannot perform '{operation}': numeric column '{col_plan}' not valid or not found."
            if filtered_df.empty or filtered_df[actual_col_op].isnull().all(): return f"No valid data for '{actual_col_op}' for '{operation}'."
        if operation=='find_max' or operation=='find_max_with_date':
            val_op = filtered_df[actual_col_op].max()
            if operation=='find_max_with_date':
                rows_op = filtered_df[filtered_df[actual_col_op] == val_op]
                if not rows_op.empty: 
                    date_op = rows_op['Date'].iloc[0].strftime('%Y-%m-%d')
                    output_cols_plan = plan.get('output_columns', [actual_col_op, 'Date'])
                    res_str = f"Highest '{actual_col_op}': **{val_op:.2f}** on **{date_op}**."
                    details = [f"{actual_out_col}: {rows_op[actual_out_col].iloc[0]:.2f}" if pd.api.types.is_numeric_dtype(rows_op[actual_out_col]) and actual_out_col.lower()!=actual_col_op.lower() and actual_out_col.lower()!='date' else (f"{actual_out_col}: {rows_op[actual_out_col].iloc[0].strftime('%Y-%m-%d')}" if isinstance(rows_op[actual_out_col].iloc[0], pd.Timestamp) and actual_out_col.lower()!='date' else None) for actual_out_col in [next((c for c in rows_op.columns if c.lower() == ocp.lower()), None) for ocp in output_cols_plan] if actual_out_col]
                    details_filtered = [d for d in details if d is not None]
                    if details_filtered: res_str += " Other details: " + "; ".join(details_filtered)
                    return res_str
                return f"Found max {val_op:.2f} for '{actual_col_op}', but no date."
            return f"Highest '{actual_col_op}': **{val_op:.2f}**."
        if operation=='find_min_with_date':
            val_op = filtered_df[actual_col_op].min()
            rows_op = filtered_df[filtered_df[actual_col_op] == val_op]
            if not rows_op.empty: 
                date_op = rows_op['Date'].iloc[0].strftime('%Y-%m-%d')
                output_cols_plan = plan.get('output_columns', [actual_col_op, 'Date'])
                res_str = f"Lowest '{actual_col_op}': **{val_op:.2f}** on **{date_op}**."
                details = [f"{actual_out_col}: {rows_op[actual_out_col].iloc[0]:.2f}" if pd.api.types.is_numeric_dtype(rows_op[actual_out_col]) and actual_out_col.lower()!=actual_col_op.lower() and actual_out_col.lower()!='date' else (f"{actual_out_col}: {rows_op[actual_out_col].iloc[0].strftime('%Y-%m-%d')}" if isinstance(rows_op[actual_out_col].iloc[0], pd.Timestamp) and actual_out_col.lower()!='date' else None) for actual_out_col in [next((c for c in rows_op.columns if c.lower() == ocp.lower()), None) for ocp in output_cols_plan] if actual_out_col]
                details_filtered = [d for d in details if d is not None]
                if details_filtered: res_str += " Other details: " + "; ".join(details_filtered)
                return res_str
            return f"Found min {val_op:.2f} for '{actual_col_op}', but no date."
        if operation=='calculate_average': 
            avg_val = filtered_df[actual_col_op].mean()
            return f"Average '{actual_col_op}': **{avg_val:.2f}**."
        if operation == 'find_group_with_max_count':
            group_col_plan = plan.get('group_by_column'); actual_group_col = next((c for c in filtered_df.columns if c.lower()==group_col_plan.lower()),None) if group_col_plan else None
            group_period = plan.get('group_by_period','').lower()
            count_col_plan = plan.get('count_condition_column'); actual_count_col = next((c for c in filtered_df.columns if c.lower()==count_col_plan.lower()),None) if count_col_plan else None
            count_val_plan = plan.get('count_condition_value')
            if not (actual_group_col and pd.api.types.is_datetime64_any_dtype(filtered_df[actual_group_col]) and group_period == 'month' and actual_count_col and count_val_plan):
                return f"Cannot execute 'find_group_with_max_count': Missing or invalid plan parameters."
            condition_df = filtered_df[filtered_df[actual_count_col].astype(str).str.lower() == str(count_val_plan).lower()].copy()
            if condition_df.empty: return f"No days found where '{actual_count_col}' was '{count_val_plan}' in the filtered period."
            condition_df['grouping_key'] = condition_df[actual_group_col].dt.strftime('%B') 
            monthly_counts = condition_df.groupby('grouping_key').size()
            if monthly_counts.empty: return f"No '{count_val_plan}' days found to group by month."
            max_count = monthly_counts.max()
            best_months_names = monthly_counts[monthly_counts == max_count].index.tolist()
            if len(best_months_names) == 1: return f"The month with the most '{count_val_plan}' days was **{best_months_names[0]}** with **{max_count}** day(s)."
            return f"The months with the most '{count_val_plan}' days (tied at **{max_count}** days each) were: **{', '.join(best_months_names)}**."
        notes_from_llm = plan.get('notes_from_llm', '')
        response_str = f"I understood the AI's plan for operation '{operation}'. Here's the plan from the AI:\n```yaml\n{plan.get('raw_plan_str', 'Plan not fully parsed')}\n```"
        if notes_from_llm: response_str += f"\n\nAI's Notes: {notes_from_llm}"
        return response_str
    except Exception as e: return f"Error executing plan: {e}. Plan: {plan}. Raw: {plan.get('raw_plan_str', 'N/A')}"

with tab2: 
    st.header("🤖 AI-Powered Data Insights")
    if not st.session_state.gemini_model: st.warning("Gemini AI model not available.")
    elif data_df.empty: st.warning("Stock data not loaded for chatbot.")
    else:
        st.info("Ask: 'How many bullish days in 2023?', 'Highest Close in 2022 and date?', 'Lowest Open in Q2 2023?', 'Average Volume in December 2023?', 'Which month had the most bullish days in 2023?'")
        if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
        for msg in st.session_state.chat_messages: st.chat_message(msg["role"]).markdown(msg["content"])
        
        if prompt := st.chat_input("Your question about TSLA data..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            final_response_to_display = "Could not process the request."
            
            # --- Python Direct Calculation Attempt FIRST ---
            python_calculated_answer = None
            # Pattern 1: "How many bullish/LONG days in [YEAR]?"
            year_bullish_match = re.search(r"how many\s+(bullish|long)\s+days\s+in\s+(\d{4})", prompt.lower())
            if year_bullish_match and 'Date' in data_df.columns and pd.api.types.is_datetime64_any_dtype(data_df['Date']) and 'direction' in data_df.columns:
                target_year = int(year_bullish_match.group(2))
                try:
                    df_year = data_df[data_df['Date'].dt.year == target_year]
                    if not df_year.empty:
                        count = df_year[df_year['direction'].astype(str).str.lower() == 'long'].shape[0]
                        python_calculated_answer = f"In {target_year}, there were **{count}** bullish ('LONG') days."
                    else: python_calculated_answer = f"No data found for {target_year}."
                except: pass 

            # Pattern 2: "How many days in [YEAR] did TSLA close lower than it opened?"
            if not python_calculated_answer: # Only if previous pattern didn't provide an answer
                year_closed_lower_match = re.search(r"how many days in (\d{4}).*close lower than.*opened", prompt.lower())
                if year_closed_lower_match and all(c in data_df.columns for c in ['Date', 'Open', 'Close']) and \
                   pd.api.types.is_datetime64_any_dtype(data_df['Date']) and \
                   pd.api.types.is_numeric_dtype(data_df['Open']) and pd.api.types.is_numeric_dtype(data_df['Close']):
                    target_year_cl = int(year_closed_lower_match.group(1))
                    try:
                        df_year_cl = data_df[data_df['Date'].dt.year == target_year_cl].copy()
                        df_year_cl.dropna(subset=['Open', 'Close'], inplace=True)
                        if not df_year_cl.empty:
                            closed_lower_count = df_year_cl[df_year_cl['Close'] < df_year_cl['Open']].shape[0]
                            python_calculated_answer = f"In {target_year_cl}, TSLA closed lower than it opened on **{closed_lower_count}** days."
                        else: python_calculated_answer = f"No data for {target_year_cl} (after cleaning Open/Close)."
                    except: pass
            # --- End Python Direct Calculation Attempt ---

            if python_calculated_answer:
                final_response_to_display = python_calculated_answer
            else: # Fallback to LLM for planning if Python tools didn't handle it
                with st.spinner("🤖 AI is planning..."): 
                    llm_plan_str = get_gemini_response(st.session_state.gemini_model, data_df, prompt)
                if llm_plan_str:
                    parsed_plan = parse_llm_plan_robustly(llm_plan_str)
                    if parsed_plan: 
                        parsed_plan['raw_plan_str'] = llm_plan_str
                        with st.spinner("⚙️ Python is executing AI plan..."): 
                            final_response_to_display = execute_llm_plan(parsed_plan, data_df)
                    else: 
                        final_response_to_display = f"AI provided a plan, but I had trouble parsing it. AI's response:\n```\n{llm_plan_str}\n```"
                else: 
                    final_response_to_display = "The AI could not generate a plan or response for this query."
            
            st.chat_message("assistant").markdown(final_response_to_display)
            st.session_state.chat_messages.append({"role": "assistant", "content": final_response_to_display})

with tab_bonus: 
    st.header("🎬 Market Replay")
    if data_df.empty: 
        st.warning("No data for animation.")
    else:
        max_anim_frames = len(data_df) 
        
        prefix = "replay_tv_pro_v13_" 
        
        # Initialize session state
        if f'{prefix}active' not in st.session_state: st.session_state[f'{prefix}active'] = False
        if f'{prefix}index' not in st.session_state: st.session_state[f'{prefix}index'] = 0
        if f'{prefix}speed' not in st.session_state: st.session_state[f'{prefix}speed'] = 0.25 
        if f'{prefix}last_fig' not in st.session_state: st.session_state[f'{prefix}last_fig'] = None
        
        # --- Animation Controls ---
        current_speed = st.session_state[f'{prefix}speed']
        new_speed = st.slider(
            "Replay Speed (seconds per frame)", 
            0.10, 2.0, current_speed, 0.05, 
            key=f"{prefix}speed_slider", 
            help="Adjust animation speed (higher value is slower)"
        )
        if new_speed != current_speed: # Only update if slider value actually changed
            st.session_state[f'{prefix}speed'] = new_speed
            # No rerun needed here, speed will be picked up by active loop

        cols_ctrl = st.columns(3)
        if cols_ctrl[0].button("▶️ Start / Resume", key=f"{prefix}start_btn", use_container_width=True):
            st.session_state[f'{prefix}active'] = True
            if st.session_state[f'{prefix}index'] >= max_anim_frames: 
                st.session_state[f'{prefix}index'] = 0
                st.session_state[f'{prefix}last_fig'] = None 
            st.rerun() # Trigger animation loop

        if cols_ctrl[1].button("⏸️ Pause", key=f"{prefix}pause_btn", use_container_width=True):
            st.session_state[f'{prefix}active'] = False
            st.rerun() # Rerun to ensure the paused state is displayed immediately with the last chart

        if cols_ctrl[2].button("⏹️ Reset", key=f"{prefix}reset_btn", use_container_width=True):
            st.session_state[f'{prefix}active'] = False
            st.session_state[f'{prefix}index'] = 0
            st.session_state[f'{prefix}last_fig'] = None
            # Get placeholder and empty it. Important to do this before rerun.
            if f"{prefix}placeholder" in st.session_state and st.session_state[f"{prefix}placeholder"] is not None:
                 st.session_state[f"{prefix}placeholder"].empty() 
            else: # If placeholder wasn't in session_state, create it to empty it (or just let it be recreated)
                 st.empty().empty() # A bit of a hack to ensure previous content is cleared if placeholder was lost
            st.rerun() 
        
        # --- Progress Bar and Chart Placeholder ---
        # Define placeholder consistently before it's used in conditional logic
        if f"{prefix}placeholder" not in st.session_state:
            st.session_state[f"{prefix}placeholder"] = st.empty()
        
        # Use the direct variable for the current script run
        chart_display_area = st.session_state[f"{prefix}placeholder"]
        
        progress_bar_replay = st.progress(0.0)
        current_progress = float(st.session_state[f'{prefix}index']) / max_anim_frames if max_anim_frames > 0 else 0
        progress_bar_replay.progress(min(current_progress, 1.0))


        # --- Animation Logic ---
        if st.session_state[f'{prefix}active'] and st.session_state[f'{prefix}index'] < max_anim_frames:
            current_idx = st.session_state[f'{prefix}index']
            slice_end_idx = current_idx + 1
            min_context = max(1, slice_end_idx - 100) 
            if slice_end_idx < 20: min_context = 0
            
            animated_df_slice = data_df.iloc[min_context:slice_end_idx]

            if not animated_df_slice.empty:
                current_display_date = data_df['Date'].iloc[current_idx] 
                anim_title = f"Replay: {current_display_date.strftime('%Y-%m-%d')} (Bar {current_idx+1}/{max_anim_frames})"
                anim_fig = create_tv_style_plotly_chart(animated_df_slice.copy(), chart_title=anim_title, chart_height=600, is_animation_frame=True) 
                
                if anim_fig:
                    full_min_low = data_df['Low'].min(); full_max_high = data_df['High'].max()
                    padding = (full_max_high - full_min_low) * 0.05 
                    anim_fig.update_yaxes(range=[full_min_low - padding, full_max_high + padding], autorange=False, row=1, col=1)
                    
                    st.session_state[f'{prefix}last_fig'] = anim_fig # Store this figure
                    chart_display_area.plotly_chart(anim_fig, use_container_width=True, key=f"{prefix}chart_active_{current_idx}")
            
            st.session_state[f'{prefix}index'] += 1 # Increment index
            
            # Check for end of replay
            if st.session_state[f'{prefix}index'] >= max_anim_frames: 
                st.session_state[f'{prefix}active'] = False 
                st.success("Replay Finished!")
                # Update progress bar one last time to full
                progress_bar_replay.progress(1.0)
                # No immediate rerun here; the last_fig will be shown by the block below
                # If you want an immediate rerun to ensure the "Replay Finished!" message is dominant, add st.rerun()
            elif st.session_state[f'{prefix}active']: # If still active and not at the end
                time.sleep(st.session_state[f'{prefix}speed']) 
                st.rerun() # Rerun for the next frame

        # --- Logic for Paused or Finished State (Displaying the last rendered figure) ---
        elif st.session_state[f'{prefix}last_fig'] is not None: 
            chart_display_area.plotly_chart(st.session_state[f'{prefix}last_fig'], use_container_width=True, key=f"{prefix}chart_paused_or_done")
            if st.session_state[f'{prefix}index'] >= max_anim_frames:
                 # This success message might appear along with the one from the loop end
                 # Consider consolidating status messages outside the main conditional blocks
                 pass # "Replay Finished!" already shown
            elif not st.session_state[f'{prefix}active'] and st.session_state[f'{prefix}index'] > 0:
                 st.info(f"Replay paused at bar {st.session_state[f'{prefix}index']}.")
        
        # --- Initial State Message (if no figure has been rendered yet) ---
        if not st.session_state[f'{prefix}active'] and st.session_state[f'{prefix}index'] == 0 and st.session_state[f'{prefix}last_fig'] is None:
             chart_display_area.info("Click 'Start / Resume' to begin animation replay.")
             progress_bar_replay.progress(0.0) # Ensure progress bar is at 0