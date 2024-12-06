from utils import *
import streamlit as st
import pandas as pd
import concurrent.futures
import requests
from requests.auth import HTTPBasicAuth
# import optuna
import datetime
import time
# mlflow.set_tracking_uri("http://127.0.0.1:5000")



# Set the page config to wide mode
st.set_page_config(layout="wide")
# Set the title of the Streamlit app
st.title("MLOPS Dashboard")

# Add a header for data ingestion
st.header("Data Ingestion")



# Initial start date
# config = load_config()
start_date = '2013-01-01'
# st.write(config)

# Streamlit code to handle button press and state
if 'end_date' not in st.session_state:
    # config['data_ingestion']['end_date'] = '2014-02-01'
    # Initialize end_date to 4 months from start_date
    st.session_state.end_date = pd.to_datetime(start_date) + pd.DateOffset(months=13)
    # save_config(config)

if 'fig1' not in st.session_state:
    st.session_state.fig1 = None
if 'fig2' not in st.session_state:
    st.session_state.fig2 = None
if 'fig_oil' not in st.session_state:
    st.session_state.fig_oil = None
if 'fig_distribution' not in st.session_state:
    st.session_state.fig_distribution = None

if 'fig_seasonality' not in st.session_state:
    st.session_state.fig_seasonality = None



# Create a sidebar
st.sidebar.header("Control Panel")

st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            height: 50px;
            font-size: 16px;

            color: white;
            border-radius: 5px;
        }
        .stButton {
            margin-bottom: 55px;
        }
            
    </style>
""", unsafe_allow_html=True)

if st.sidebar.button("Data Ingestion"):

        # Airflow API URL and credentials
    url = "http://localhost:8081/api/v1/dags/ingest_data_DAG/dagRuns"
    auth = HTTPBasicAuth("admin", "admin")

    payload = {
        "conf": {
            "start_date": start_date,  # Replace with actual start date
            "end_date": st.session_state.end_date.strftime('%Y-%m-%d'),     # Replace with actual end date
        }
    }

    # Trigger a DAG run
    response = requests.post(url, auth=auth, json=payload,headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        # st.write(response.json())
        st.success("Task triggered")
    else:
        st.error(f"Failed to trigger DAG: {response.text}")

    dag_run_id = response.json()['dag_run_id']

    # Polling the status of the DAG run
    status_url = f"http://localhost:8081/api/v1/dags/ingest_data_DAG/dagRuns/{dag_run_id}"

    # Keep checking the status until the DAG run is completed
    st.success("Task running")
    while True:
        
        status_response = requests.get(status_url, auth=auth)
        # st.write(status_response.json())
        if status_response.status_code == 200:
            dag_run_status = status_response.json()['state']
            if dag_run_status in ['success', 'failed']:
                if dag_run_status == 'success':
                    st.success("Task completed successfully!")
                else:
                    st.error("DAG failed!")
                break
        else:
            st.error(f"Failed to fetch DAG status: {status_response.text}")
            break
        
    
    # Call the ingest_data function
    df_train =pd.read_csv('data/train.csv',parse_dates=['date'])
    df_oil =pd.read_csv('data/oil.csv',parse_dates=['date'])
    
    st.session_state.df_train = df_train
    
    # Define family groups
    group1 = ['GROCERY I', 'BEVERAGES', 'CLEANING','PRODUCE','DAIRY','FROZEN FOODS']
    group2 = df_train['family'].unique().tolist()
    group2 = [fam for fam in group2 if fam not in group1]
    
    # Plot and display the data for group1
    # Generate the plots and store them in the session state
    st.session_state.fig1 = plot_sales_with_moving_avg(df_train, group1)
    st.session_state.fig2 = plot_sales_with_moving_avg(df_train, group2)
    st.session_state.fig_oil = display_oil_data(df_oil)

    
    st.session_state.end_date += pd.DateOffset(months=1)
    # config['data_ingestion']['end_date'] =st.session_state.end_date.strftime('%Y-%m-%d')
    # save_config(config)


# Create two columns for side-by-side display with specified width ratios
# col1, col2 = st.columns([2, 2])  # Each column takes equal width

# Display the first plot in the first column with increased width

if st.session_state.fig1 is not None:
    st.plotly_chart(st.session_state.fig1, use_container_width=True, height=500)

# Display the second plot in the second column with increased width
if st.session_state.fig2 is not None:
    st.plotly_chart(st.session_state.fig2, use_container_width=True, height=500)

# Display oil data
if st.session_state.fig_oil is not None:
    st.plotly_chart(st.session_state.fig_oil, use_container_width=True, height=500)

    # Add button for analyzing new data
st.header("Data Analysis : compare latest data. ")
if st.sidebar.button("Analyze New Data"):
    st.success("Task triggered")
    df_train = st.session_state.df_train

    families_to_analyze = ['EGGS', 'BEAUTY', 'FROZEN FOODS', 'BREAD/BAKERY', 'CLEANING', 'SEAFOOD']  # Replace with actual family names
    
    # Call the function to plot the sales distribution overlay

    st.session_state.fig_distribution = plot_sales_distribution_overlay(df_train, families_to_analyze) 


    st.session_state.fig_seasonality = plot_sales_seasonality(df_train, families_to_analyze) 
    st.success("Task running")
    time.sleep(2)
    st.success("Task completed")
# Display the sales distribution overlay
if st.session_state.fig_distribution is not None:
    st.plotly_chart(st.session_state.fig_distribution, use_container_width=True, height=500)
    st.plotly_chart(st.session_state.fig_seasonality, use_container_width=True, height=500)# Define a dummy function for training models






if "optuna_studies" in st.session_state:
    del st.session_state["optuna_studies"]

st.session_state.families = st.sidebar.multiselect(
"Select Models to Train", 
options=["BEAUTY", "CLEANING", "GROCERY I", "SEAFOOD"],  # Add more family options here
default=[]  # Default selection
)
st.header("Train Models")
if st.sidebar.button("Train Models"):
    st.success("Task triggered")
    # st.write('Training starting...')
    df_train = st.session_state.df_train
    
    families = list(st.session_state.families)
    # st.write(families)
    st.success("training started")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each family to be trained in parallel
        futures = [executor.submit(train_family, family, df_train) for family in families]
        
        # Wait for all to complete and retrieve results
        results = []
        for future in futures:
            try:
                results.append(future.result())   # Fetch any return values if needed
            except Exception as e:
                st.error(f"An error occurred during training: {e}")
    st.success("training completed")
    # Sequential execution of training for each family
    # results = []
    # for family in families:
    #     try:
    #         result = train_family(family, df_train)  # Call the function directly
    #         results.append(result)  # Append the result if needed
    #     except Exception as e:
    #         st.error(f"An error occurred during training: {e}")
    
    for i, result in enumerate(results):
        st.header(f"{families[i]} Models")  # Display heading for each family
        for figs in result:
            st.plotly_chart(figs)  # Plot the chart for the respective family
            # st.button(f'publish {families[i]} model')
    # st.button(f'publish models')

if st.sidebar.button("Publish Models"):
    st.success("Model published!")