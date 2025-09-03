import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import google.generativeai as genai
import plotly.express as px
from io import BytesIO
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------- CONFIG ----------------
genai.configure(api_key="AIzaSyAIgZcXNSK0PBeqtAnCeYtZCDbDGbRGXio")  # Replace with your key

st.set_page_config(page_title="ARIMA Forecasting", layout="wide")
st.title("📊 AI insights generator using ARIMA models forecasting")

# ---------------- Model Helpers ----------------
@st.cache_data(show_spinner=False)
def fit_arima(train_series, order=(2, 1, 3)):
    model = ARIMA(train_series, order=order)
    return model.fit()

# ---------------- File Upload ----------------
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

        # Group by date to ensure unique index
        df = df.groupby(date_col).sum(numeric_only=True)
        df.index.name = date_col  # set index name

        # Detect sales column
        sales_col = next((col for col in df.columns if "sale" in col.lower() or "revenue" in col.lower()), None)
        if sales_col is None:
            st.error("Could not find a sales column. Make sure one column contains 'sales' or 'revenue'.")
            st.stop()

        # ---------------- Tabs ----------------
        tabs = st.tabs(["📄 Data Preview", "🔮 Forecast", "📌 Anomaly Detection", "💡 AI Insights"])

        # ---------------- Tab 1: Data Preview ----------------
        with tabs[0]:
            st.subheader("Data Preview")
            st.dataframe(df.head(20))
            st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

        # ---------------- Tab 2: Forecast ----------------
        with tabs[1]:
            st.subheader("ARIMA Forecast")

            forecast_days = st.slider("Forecast Horizon (days)", min_value=1, max_value=365, value=30)
            use_log = st.checkbox("Apply log transform for smoothing", value=True)

            series = np.log(df[sales_col].clip(lower=1)) if use_log else df[sales_col]

            # Fixed ARIMA order
            model_fit = fit_arima(series, order=(2, 1, 3))
            forecast_series = model_fit.forecast(steps=forecast_days)
            forecast_values = np.exp(forecast_series) if use_log else forecast_series

            # Forecast DataFrame
            forecast_index = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({"Forecasted Sales": forecast_values}, index=forecast_index)

            # Combine actual and forecast
            actual_df = df[[sales_col]].rename(columns={sales_col: "Actual Sales"})
            full_df = pd.concat([actual_df, forecast_df], axis=1)

            # ---------------- Chart ----------------
            st.markdown("### 📈 Chart: Actual and Forecast Visualization")
            st.line_chart(full_df)

            # ---------------- Table ----------------
            def highlight_forecast_row(row):
                if pd.isna(row.get("Actual Sales")):
                    return ["background-color: lightgreen"] * len(row)
                else:
                    return [""] * len(row)

            st.markdown("### 📌 Actual vs Forecast Table")
            styled_display = full_df.style.format("{:,.2f}").apply(highlight_forecast_row, axis=1)
            st.dataframe(styled_display, use_container_width=True)

            # ---------------- CSV Download ----------------
            csv_bytes = full_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Actual vs Forecast CSV", data=csv_bytes, file_name="actual_vs_forecast.csv", mime="text/csv")

            # ---------------- Last Forecast Summary ----------------
            last_actual = df[sales_col].iloc[-1]
            last_forecast_value = forecast_df["Forecasted Sales"].iloc[-1]
            st.success(f"✅ Forecasted Sales after {forecast_days} days: {last_forecast_value:,.2f}")

        # ---------------- Tab 3: Anomaly Detection ----------------
        with tabs[2]:
            st.subheader("Anomaly Detection in Historical Sales")
            try:
                decomp = seasonal_decompose(df[sales_col], period=7, model="additive", extrapolate_trend="freq")
                resid = decomp.resid.dropna()
                std = resid.std()
                anomalies = resid[np.abs(resid) > 2 * std]

                st.write("Detected anomalies (sudden spikes/drops):")
                st.dataframe(anomalies)

                if not anomalies.empty:
                    anomaly_prompt = f"""
                    You are a supply chain and sales data expert.
                    We detected anomalies (unusual spikes or drops) in sales data at these time points: {anomalies.to_dict()}.

                    Please analyze and provide:
                    1. Possible **business reasons** (e.g., sudden demand, promotions, competitor moves, supply shortages, seasonal effects).  
                    2. The **potential impact** of these anomalies on revenue and operations.  
                    3. **Recommendations** for management to handle or prevent such anomalies in the future.  

                    Keep the explanation short, clear, and business-oriented.
                    """

                    model_ai = genai.GenerativeModel("gemini-1.5-flash")
                    anomaly_response = model_ai.generate_content(anomaly_prompt)
                    st.markdown("### 🤖 Explanation of Anomalies")
                    st.write(anomaly_response.text)
            except Exception as e:
                st.warning(f"Could not perform anomaly detection: {e}")

        # ---------------- Tab 4: AI Insights ----------------
        with tabs[3]:
            st.subheader("AI-Generated Business Insights")
            lang = st.selectbox("Select Language for Insights", ["English", "Spanish", "French", "German", "Hindi"])

            last_actual = df[sales_col].iloc[-1]
            last_forecast = forecast_df["Forecasted Sales"].iloc[-1]
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
            - Optional: Generate insights in {lang}.
            """

            model_ai = genai.GenerativeModel("gemini-1.5-flash")
            response = model_ai.generate_content(prompt)
            st.markdown(response.text)

    except Exception as e:
        st.error(f" Error: {e}")
