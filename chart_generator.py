import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_tv_style_plotly_chart(df: pd.DataFrame, chart_title: str = "TSLA Stock Analysis", chart_height: int = 700, is_animation_frame: bool = False):
    if df.empty:
        st.warning("Chart Generator: Input DataFrame is empty.")
        return None

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols) or \
       not all(pd.api.types.is_numeric_dtype(df[col]) for col in ['Open', 'High', 'Low', 'Close']) or \
       not pd.api.types.is_datetime64_any_dtype(df['Date']):
        st.error(f"Chart Generator: Data not in expected format. Required: {required_cols} with correct types.")
        # st.text(f"Columns: {df.columns.tolist()}"); st.text(f"Data Types:\n{df.dtypes}") # Debug
        return None
    
    df = df.copy() 

    tv_colors = {
        'bg': 'rgb(18, 22, 25)', 'plot_bg': 'rgb(18, 22, 25)', 'text': 'rgb(210, 213, 218)',      
        'grid': 'rgba(59, 69, 85, 0.7)', 'candlestick_up_fill': 'rgb(0, 176, 80)',
        'candlestick_up_line': 'rgb(0, 176, 80)', 'candlestick_down_fill': 'rgb(240, 80, 80)',
        'candlestick_down_line': 'rgb(240, 80, 80)', 'volume_up': 'rgba(0, 176, 80, 0.45)',
        'volume_down': 'rgba(240, 80, 80, 0.45)', 'support_fill': 'rgba(0, 200, 83, 0.08)', 
        'support_line': 'rgba(0, 200, 83, 0.5)', 'resistance_fill': 'rgba(239, 83, 80, 0.08)',
        'resistance_line': 'rgba(239, 83, 80, 0.5)', 'marker_long': 'rgb(0, 220, 100)',
        'marker_short': 'rgb(255, 80, 80)', 'marker_neutral': 'rgb(255, 193, 7)',
        'current_price_line': 'rgba(74, 144, 226, 0.9)'
    }

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.01, row_heights=[0.85, 0.15],
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

    hover_texts_candlestick = [
        f"<b>{d.strftime('%Y-%m-%d')}</b><br>O: {o:.2f} H: {h:.2f} L: {l:.2f} C: {c:.2f}<br>" +
        (f"Vol: {v:,.0f}" if pd.notnull(v) else "Vol: N/A")
        for d, o, h, l, c, v in zip(df['Date'], df['Open'], df['High'], df['Low'], df['Close'], 
                                    df.get('Volume', pd.Series([np.nan]*len(df))))
    ]
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                 name='Price',
                                 increasing=dict(line=dict(color=tv_colors['candlestick_up_line'], width=1), fillcolor=tv_colors['candlestick_up_fill']),
                                 decreasing=dict(line=dict(color=tv_colors['candlestick_down_line'], width=1), fillcolor=tv_colors['candlestick_down_fill']),
                                 hoverlabel=dict(bgcolor=tv_colors['plot_bg'], font_size=11, bordercolor=tv_colors['grid'], namelength=-1, font_color=tv_colors['text']),
                                 hovertext=hover_texts_candlestick, hoverinfo="text"), 
                  row=1, col=1)

    if 'Volume' in df.columns and df['Volume'].notna().any():
        df['Volume_Num'] = pd.to_numeric(df['Volume'], errors='coerce')
        # Ensure Open and Close exist for coloring volume bars
        volume_df = df.dropna(subset=['Date', 'Volume_Num', 'Open', 'Close']) 
        if not volume_df.empty:
            vol_colors = [tv_colors['volume_up'] if row['Close'] >= row['Open'] else tv_colors['volume_down'] for index, row in volume_df.iterrows()]
            fig.add_trace(go.Bar(x=volume_df['Date'], y=volume_df['Volume_Num'], name='Volume', marker_color=vol_colors, showlegend=False,
                                 hoverinfo="skip"), row=2, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1, title_text=None)

    for band_type in ['Support', 'Resistance']:
        lower_col, upper_col = f'{band_type}_Lower', f'{band_type}_Upper'
        fill_c = tv_colors['support_fill'] if band_type == 'Support' else tv_colors['resistance_fill']
        line_c = tv_colors['support_line'] if band_type == 'Support' else tv_colors['resistance_line']
        if lower_col in df.columns and upper_col in df.columns and df[lower_col].notna().any() and df[upper_col].notna().any():
            band_df = df.dropna(subset=['Date', lower_col, upper_col])
            if not band_df.empty:
                fig.add_trace(go.Scatter(x=band_df['Date'], y=band_df[upper_col], line=dict(color=line_c, width=1, dash='solid'), name=f'{band_type} Upper', legendgroup=band_type.lower(), hoverinfo='x+y', showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=band_df['Date'], y=band_df[lower_col], line=dict(color=line_c, width=1, dash='solid'), name=f'{band_type} Band', legendgroup=band_type.lower(), fill='tonexty', fillcolor=fill_c, hoverinfo='x+y'), row=1, col=1)
    
    markers_df = df[df['direction'].notna()].copy()
    if not markers_df.empty:
        # Ensure High and Low are available for ATR calculation
        if 'High' in df.columns and 'Low' in df.columns and df['High'].notna().any() and df['Low'].notna().any():
            atr_series = (df['High'] - df['Low']).rolling(window=10, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill')
        else:
            atr_series = pd.Series(0, index=df.index) # Fallback if High/Low are missing
            st.warning("Chart Generator: High/Low columns missing for ATR calculation for marker offset.")

        marker_offset_multiplier = 0.3 
        marker_configs = [
            {'condition_str': 'long', 'y_col': 'Low', 'name': 'LONG', 'symbol': 'triangle-up', 'color': tv_colors['marker_long'], 'border_color': tv_colors['bg'], 'y_offset_factor': -1, 'size': 9},
            {'condition_str': 'short', 'y_col': 'High', 'name': 'SHORT', 'symbol': 'triangle-down', 'color': tv_colors['marker_short'], 'border_color': tv_colors['bg'], 'y_offset_factor': 1, 'size': 9},
            {'condition_str': ['none', ''], 'y_col': 'Close', 'name': 'NEUTRAL', 'symbol': 'circle', 'color': tv_colors['marker_neutral'], 'border_color': tv_colors['bg'], 'size': 6, 'y_offset_factor': 0}
        ]
        show_legend_flags = {cfg['name']: True for cfg in marker_configs}
        for m_config in marker_configs:
            condition = markers_df['direction'].astype(str).str.lower().isin(m_config['condition_str']) if isinstance(m_config['condition_str'], list) else markers_df['direction'].astype(str).str.lower() == m_config['condition_str']
            m_df_subset = markers_df[condition].copy()
            if not m_df_subset.empty and m_config['y_col'] in m_df_subset.columns:
                m_df_subset['y_value_num'] = pd.to_numeric(m_df_subset[m_config['y_col']], errors='coerce')
                current_atr = atr_series.reindex(m_df_subset.index).fillna(0) # Ensure ATR aligns and handles NaNs
                m_df_subset['marker_y'] = m_df_subset['y_value_num'] + (current_atr * marker_offset_multiplier * m_config['y_offset_factor'])
                m_df_subset.dropna(subset=['Date', 'marker_y'], inplace=True)
                if not m_df_subset.empty:
                    fig.add_trace(go.Scatter(x=m_df_subset['Date'], y=m_df_subset['marker_y'], mode='markers', name=m_config['name'], 
                                             marker=dict(symbol=m_config['symbol'], color=m_config['color'], size=m_config.get('size', 12), line=dict(width=1.5, color=m_config['border_color'])),
                                             hoverinfo='skip', legendgroup='signals', showlegend=show_legend_flags[m_config['name']]), row=1, col=1)
                    show_legend_flags[m_config['name']] = False 
        

    fig.update_layout(
        title=None, 
        xaxis_title=None, yaxis_title=None, 
        yaxis=dict(side='right', showgrid=True, gridwidth=1, gridcolor=tv_colors['grid'], griddash='dot', zeroline=False, showline=False, tickfont=dict(size=10, color=tv_colors['text']), tickformat=".2f", automargin=True, fixedrange=False),
        height=chart_height, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor='rgba(0,0,0,0)', font=dict(color=tv_colors['text'], size=10), tracegroupgap=5),
        margin=dict(l=10, r=45, t=25, b=10), hovermode='x unified', hoverlabel=dict(bgcolor=tv_colors['plot_bg'], font_size=11, bordercolor=tv_colors['grid'], namelength=-1, font_color=tv_colors['text']),
        dragmode='pan', paper_bgcolor=tv_colors['bg'], plot_bgcolor=tv_colors['plot_bg'],
        font=dict(family="Verdana, Arial, sans-serif", size=11, color=tv_colors['text']),
        transition_duration=10 if is_animation_frame else 50, 
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=tv_colors['grid'], gridwidth=1, griddash='dot', zeroline=False, showline=False, tickfont_size=10, color=tv_colors['text'], rangeslider_visible=False, type='date', row=1, col=1) 
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, type='date', tickfont_size=10, color=tv_colors['text'], row=2, col=1)
    
    if not is_animation_frame:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(buttons=list([dict(count=1, label="1m", step="month", stepmode="backward"), dict(count=6, label="6m", step="month", stepmode="backward"), dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label="1y", step="year", stepmode="backward"), dict(step="all")]), bgcolor=tv_colors['plot_bg'], activecolor=tv_colors['current_price_line'], font=dict(color=tv_colors['text'], size=10), bordercolor=tv_colors['grid'], x=0.01, xanchor='left', y=1.08, yanchor='bottom'),
                rangeslider=dict(visible=True, thickness=0.06, bgcolor=tv_colors['plot_bg'], bordercolor=tv_colors['grid'], autorange=True)
            )
        )
        fig.update_layout(xaxis2_rangeslider_visible=False) 
    else: 
        fig.update_layout(xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False, showlegend=False, title=dict(text=chart_title, x=0.5, font=dict(size=16, color=tv_colors['text'])))
        fig.update_xaxes(rangeselector=None, row=1, col=1); fig.update_xaxes(rangeselector=None, row=2, col=1)

    if not df.empty and not is_animation_frame and 'Close' in df.columns and 'Date' in df.columns and len(df['Date']) > 0:
        if len(df['Date']) > 0 and len(df['Close']) > 0 : 
            latest_date, latest_close = df['Date'].iloc[-1], df['Close'].iloc[-1]
            if pd.notna(latest_date) and pd.notna(latest_close):
                fig.add_hline(y=latest_close, line_width=1.5, line_dash="solid", line_color=tv_colors['current_price_line'], opacity=1,
                              annotation_text=f"<b>{latest_close:.2f}</b>", annotation_position="top right",
                              annotation_font_size=10, annotation_font_color=tv_colors['bg'], 
                              annotation_bgcolor=tv_colors['current_price_line'], row=1, col=1)
    
    return fig