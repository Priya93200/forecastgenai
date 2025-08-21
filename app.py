import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import google.generativeai as genai
import matplotlib.pyplot as plt
from io import BytesIO  # For CSV download


genai.configure(api_key="AIzaSyBrEI3Ve3u6mk681Muk5LNu24VaH5M32ow")  # Replace with your key

st.set_page_config(page_title="ARIMA Forecasting", layout="wide")
st.title("📊 ARIMA Forecasting with Gen AI")


@st.cache_data(show_spinner=False)
def fit_arima(train_series, order=(2,1,3)):
    model = ARIMA(train_series, order=order)
    return model.fit()

uploaded_file = st.file_uploader("Upload your sales dataset (CSV)", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Detect date column
        date_col = next((col for col in df.columns if "date" in col.lower()), None)
        if date_col is None:
            st.error("Could not find a date column. Make sure one column contains 'date' in its name.")
            st.stop()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df.set_index(date_col, inplace=True)
        
        # Detect sales column
        sales_col = next((col for col in df.columns if "sale" in col.lower() or "revenue" in col.lower()), None)
        if sales_col is None:
            st.error("Could not find a sales column. Make sure one column contains 'sales' or 'revenue'.")
            st.stop()
        
        
        tabs = st.tabs(["📄 Data Preview", "🔮 Forecast", "📈 Confidence & Comparison", "💡 AI Insights"])
        
        
        with tabs[0]:
            st.subheader("Data Preview")
            st.dataframe(df.head(20))
            st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        
        
        with tabs[1]:
            st.subheader("ARIMA Forecast Settings")
            forecast_days = st.slider("Forecast Horizon (days)", min_value=1, max_value=365, value=30)
            use_log = st.checkbox("Apply log transform for smoothing", value=True)
            
            series = np.log(df[sales_col].clip(lower=1)) if use_log else df[sales_col]
            series_smooth = series.rolling(window=3, min_periods=1).mean() if use_log else series
            
            model_fit = fit_arima(series_smooth)
            forecast_series = model_fit.forecast(steps=forecast_days)
            forecast_values = np.exp(forecast_series) if use_log else forecast_series
            
            forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({sales_col: forecast_values.values}, index=forecast_index)
            
            # Plot historical + forecast
            combined = pd.concat([df[sales_col], forecast_df])
            st.line_chart(combined)
            
            # Download forecast
            buffer = BytesIO()
            forecast_df.reset_index().to_csv(buffer, index=False)
            st.download_button("Download Forecast CSV", data=buffer, file_name="forecast.csv", mime="text/csv")
        
        
        
        with tabs[2]:
            st.subheader("Forecast Confidence")
            train_size = int(len(series_smooth) * 0.9)
            train, test = series_smooth.iloc[:train_size], series_smooth.iloc[train_size:]
            
            if len(test) == 0:
                st.warning("Dataset too small for confidence calculation")
            else:
                test_forecast_series = model_fit.forecast(steps=len(test))
                test_forecast = np.exp(test_forecast_series) if use_log else test_forecast_series
                test_actual = df[sales_col].iloc[train_size:]
                rmse = np.sqrt(mean_squared_error(test_actual, test_forecast))
                confidence_score = max(0, 100 - (rmse / np.mean(test_actual) * 100))
                
                col1, col2 = st.columns([1,3])
                with col1:
                    st.metric("✅ Confidence Score", f"{confidence_score:.2f}%")
                    st.write("Lower RMSE → Higher confidence")
                with col2:
                    compare_df = pd.DataFrame({
                        "Actual Sales": test_actual.values,
                        "Forecasted Sales": test_forecast.values
                    }, index=test_actual.index)
                    st.line_chart(compare_df)
                    with st.expander("🔍 View Actual vs Forecasted Values"):
                        st.dataframe(compare_df)
        
        
        with tabs[3]:
            st.subheader("AI-Generated Business Insights")
            
            # Multi-language support
            lang = st.selectbox("Select Language for Insights", ["English", "Spanish", "French", "German", "Hindi"])
            
            last_actual = df[sales_col].iloc[-1]
            last_forecast = forecast_df[sales_col].iloc[-1]
            growth = (last_forecast - last_actual) / last_actual * 100
            prompt = f"""
            You are a professional sales analyst.

            Historical Data:
            - Last observed sales: {last_actual:.2f}
            - Time period of historical data: {df.index.min().date()} to {df.index.max().date()}

            Forecasted Predictions:
            - Forecast horizon: {forecast_days} days
            - Forecasted sales at the end of horizon: {last_forecast:.2f}
            - Growth compared to last observed value: {growth:.2f}%

            Business Context / KPIs:
            - Main KPI: Sales performance
            - Goal: Increase sales and optimize revenue trends

            Instructions:
            - Summarize the trend of the past data briefly.
            - Generate exactly 3 concise, actionable business insights.
            - Format insights as bullet points with emojis.
            - Ensure insights are directly relevant to the business context and KPIs.
            - Avoid unrelated information.
            - Optional: Generate insights in {lang}.

            Provide the response in a professional and actionable manner.
            """

            
            model_ai = genai.GenerativeModel("gemini-1.5-flash")
            response = model_ai.generate_content(prompt)
            st.markdown(response.text)
        
    except Exception as e:
        st.error(f" Error: {e}")
