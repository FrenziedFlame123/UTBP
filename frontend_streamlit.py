
#   streamlit run frontend_streamlit.py

import streamlit as st
import requests
import pandas as pd
import sys
import pathlib

working_dir = str(pathlib.Path().absolute())
print(working_dir)
sys.path.append(working_dir)

from my_pandas_extensions.business import plot_optim_thresh

ENDPOINT = 'http://localhost:8000'

st.title(":bar_chart: Utility Tool for Business Problems (UTBP)")
st.markdown("""
Welcome to the UTBP Web App created by **Ahmet Selçuk Arslan**.  
Analyze your data, run scoring analysis, and download insights!
""")

uploaded_file = st.file_uploader("Upload Your Dataset", type = ["csv"], accept_multiple_files=False)

@st.cache_data()
def load_data(filename):
    leads_df = pd.read_csv(uploaded_file)
    return leads_df


if uploaded_file:
    
    leads_df = load_data(uploaded_file)
    full_data_json = leads_df.to_json()

    if st.checkbox("Show raw data"):
        st.subheader("Sample of Raw Data (First 10 rows)")
        st.write(leads_df.head(10))
    
    st.write("---")
    st.markdown("#### Scoring Analysis")

    estimated_periodic_sales = st.number_input(
        "Average sales per period ($)", 0, value=250000, step=1000, help="Enter your estimated average sales per month."
    )
    periodic_sales_reduction_safeguard = st.slider(
        "Periodic sales safeguard (%)", 0., 1., 0.9, step=0.01, help="Percentage of sales to maintain as a safeguard."
    )
    
    sales_limit = "${:,.0f}".format(periodic_sales_reduction_safeguard * estimated_periodic_sales)
    st.subheader(f"Periodic sales will not go below: {sales_limit}")
    
    
    if st.button("Run Analysis"):
    
        with st.spinner("Scoring in progress... This may take a minute."):
            
            res = requests.post(
                url = f"{ENDPOINT}/calculate_strategy",
                json = full_data_json,
                params = dict(
                    periodic_sales_reduction_safeguard=float(periodic_sales_reduction_safeguard),
                    list_size=100000,
                    unsub_rate_per_periodic_campaign=0.005,
                    periodic_campaign_count=5,
                    avg_sales_per_period=float(estimated_periodic_sales),
                    avg_periodic_campaign_count=5,
                    customer_conversion_rate=0.05,
                    avg_customer_value=2000.0
                )
            )
            
            print("Raw Response Text:", res.text)
            
            print(res.json().keys())
            
            lead_strategy_df = pd.read_json(res.json()["lead_strategy"])
            expected_value_df = pd.read_json(res.json()["expected_value"])
            thresh_optim_table_df = pd.read_json(res.json()["thresh_optim_table"])
            
            st.success("Success! Scoring is complete. Download the results below.")
            
            st.subheader("Strategy Summary:")
            st.write(expected_value_df)
            
            st.subheader("Expected Value Plot")
            st.plotly_chart(
                plot_optim_thresh(
                    thresh_optim_df=thresh_optim_table_df,
                    periodic_sales_reduction_safeguard=periodic_sales_reduction_safeguard  
                )
            )
            
            st.subheader("Sample of Strategy (First 10 Rows)")
            st.write(lead_strategy_df.head(10))
            
            st.download_button(
                label = "Download Scoring Strategy",
                data = lead_strategy_df.to_csv(index = False),
                file_name = "strategy.csv",
                mime = "text/csv",
                key = "download-csv"
            )