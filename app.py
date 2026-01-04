"""
AI Personal Expense Assistant - Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import our modules
from core.preprocessing import clean_data
from core.categorization import load_model as load_categorizer
from ml.train_model import train_categorization_model, save_model
from ml.forecasting import forecast_next_month
from core.anomalies import detect_anomalies
from ai.vibe_engine import VibeEngine

st.set_page_config(
    page_title="AI Personal Expense Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI Personal Expense Assistant")
st.markdown("Upload your expense files and get insights, forecasts, and answers to your questions!")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'categorizer' not in st.session_state:
    st.session_state.categorizer = None

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your expense data file"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Clean data
            df = clean_data(df)
            st.session_state.df = df
            
            st.success(f"✅ Loaded {len(df)} transactions")
            
            # Auto-categorize if model exists or train one
            if 'category' in df.columns and df['category'].nunique() > 1:
                # Try to load existing model
                categorizer = load_categorizer()
                
                if categorizer is None or not categorizer.is_trained:
                    # Train new model
                    with st.spinner("Training categorization model..."):
                        categorizer = train_categorization_model(df)
                        if categorizer.is_trained:
                            save_model(categorizer)
                            st.success("Model trained and saved!")
                else:
                    # Use existing model for predictions
                    unknown_mask = df['category'] == 'unknown'
                    if unknown_mask.any():
                        with st.spinner("Categorizing expenses..."):
                            predictions = categorizer.predict(df[unknown_mask]['description'])
                            df.loc[unknown_mask, 'category'] = predictions
                            st.session_state.df = df
                            st.success("Expenses categorized!")
                
                st.session_state.categorizer = categorizer
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Main content area with tabs
if st.session_state.df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 AI Chat", "🔮 Forecast", "⚠️ Anomalies"])
    
    df = st.session_state.df.copy()
    
    with tab1:
        st.header("Expense Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Expenses", f"${df['amount'].sum():,.2f}")
        with col2:
            st.metric("Total Transactions", len(df))
        with col3:
            st.metric("Average Transaction", f"${df['amount'].mean():,.2f}")
        with col4:
            st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Expenses Over Time")
            daily_expenses = df.groupby('date')['amount'].sum().reset_index()
            fig_time = px.line(daily_expenses, x='date', y='amount', 
                             title='Daily Expenses Trend')
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            st.subheader("Expenses by Category")
            if 'category' in df.columns and df['category'].nunique() > 1:
                category_expenses = df.groupby('category')['amount'].sum().sort_values(ascending=False)
                fig_cat = px.pie(values=category_expenses.values, names=category_expenses.index,
                               title='Expenses Distribution by Category')
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Category data not available")
        
        # Recent transactions table
        st.subheader("Recent Transactions")
        display_df = df[['date', 'description', 'amount', 'category']].tail(20)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.header("AI Chat - Ask Questions About Your Expenses")
        st.markdown("Try queries like: *'restaurant expenses in january'*, *'total expenses'*, *'food above 500'*")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "data" in message:
                    st.dataframe(message["data"], use_container_width=True, hide_index=True)
        
        # Chat input
        query = st.chat_input("Ask about your expenses...")
        
        if query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.write(query)
            
            # Process query
            with st.chat_message("assistant"):
                try:
                    vibe_engine = VibeEngine(df)
                    result_df, summary = vibe_engine.generate_report_from_text(query, df)
                    
                    # Display summary
                    st.write(summary)
                    
                    # Display data if not empty
                    if len(result_df) > 0:
                        # Show summary metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Amount", f"${result_df['amount'].sum():,.2f}")
                        with col2:
                            st.metric("Transactions", len(result_df))
                        
                        st.dataframe(result_df[['date', 'description', 'amount', 'category']], 
                                   use_container_width=True, hide_index=True)
                        
                        # Add chart if applicable
                        if len(result_df) > 1 and 'date' in result_df.columns:
                            # Group by date for better visualization
                            chart_df = result_df.groupby('date')['amount'].sum().reset_index()
                            fig = px.bar(chart_df, x='date', y='amount', 
                                       title='Query Results Over Time')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": summary,
                        "data": result_df[['date', 'description', 'amount', 'category']] if len(result_df) > 0 else None
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    with tab3:
        st.header("Expense Forecast")
        st.markdown("Forecast your future expenses using Prophet time series forecasting")
        
        forecast_periods = st.slider("Forecast Period (days)", 7, 90, 30)
        
        if st.button("Generate Forecast"):
            try:
                with st.spinner("Generating forecast..."):
                    forecast_df = forecast_next_month(df, periods=forecast_periods)
                    
                    if forecast_df is not None:
                        st.success("Forecast generated successfully!")
                        
                        # Display forecast chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=forecast_df['date'],
                            y=forecast_df['forecasted_amount'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_df['date'],
                            y=forecast_df['upper_bound'],
                            mode='lines',
                            name='Upper Bound',
                            line=dict(color='lightblue', dash='dash'),
                            fill=None
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_df['date'],
                            y=forecast_df['lower_bound'],
                            mode='lines',
                            name='Lower Bound',
                            line=dict(color='lightblue', dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(173, 216, 230, 0.2)'
                        ))
                        
                        fig.update_layout(
                            title='Expense Forecast',
                            xaxis_title='Date',
                            yaxis_title='Forecasted Amount ($)',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast table
                        st.subheader("Forecast Details")
                        forecast_df['forecasted_amount'] = forecast_df['forecasted_amount'].round(2)
                        forecast_df['lower_bound'] = forecast_df['lower_bound'].round(2)
                        forecast_df['upper_bound'] = forecast_df['upper_bound'].round(2)
                        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                        
                        # Summary metrics
                        total_forecast = forecast_df['forecasted_amount'].sum()
                        avg_daily = forecast_df['forecasted_amount'].mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Forecasted Expenses", f"${total_forecast:,.2f}")
                        with col2:
                            st.metric("Average Daily Forecast", f"${avg_daily:,.2f}")
                    else:
                        st.warning("Not enough data to generate forecast. Need at least 2 data points.")
                        
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
    
    with tab4:
        st.header("Anomaly Detection")
        st.markdown("Detect unusual transactions and spending patterns")
        
        if st.button("Detect Anomalies"):
            try:
                with st.spinner("Detecting anomalies..."):
                    anomalies_df = detect_anomalies(df)
                    anomaly_count = anomalies_df['is_anomaly'].sum()
                    
                    if anomaly_count > 0:
                        st.warning(f"⚠️ Found {anomaly_count} anomalous transactions")
                        
                        # Display anomalies
                        anomaly_transactions = anomalies_df[anomalies_df['is_anomaly']].copy()
                        st.dataframe(
                            anomaly_transactions[['date', 'description', 'amount', 'category']],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Anomaly statistics
                        st.subheader("Anomaly Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Anomalous Amount", f"${anomaly_transactions['amount'].sum():,.2f}")
                        with col2:
                            st.metric("Average Anomaly Amount", f"${anomaly_transactions['amount'].mean():,.2f}")
                        with col3:
                            st.metric("Largest Anomaly", f"${anomaly_transactions['amount'].max():,.2f}")
                        
                        # Chart
                        st.subheader("Anomalies Visualization")
                        fig = px.scatter(
                            anomalies_df,
                            x='date',
                            y='amount',
                            color='is_anomaly',
                            hover_data=['description', 'category'],
                            title='Transactions with Anomalies Highlighted'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("✅ No anomalies detected. All transactions are within normal ranges.")
                        
            except Exception as e:
                st.error(f"Error detecting anomalies: {str(e)}")

else:
    st.info("👈 Please upload a CSV or Excel file to get started!")
    st.markdown("""
    ### Sample Data Format
    
    Your file should contain columns like:
    - **date**: Transaction date
    - **description**: Transaction description
    - **amount**: Transaction amount
    - **category** (optional): Expense category
    
    You can use the sample file in `data/sample.csv` as a reference.
    """)
