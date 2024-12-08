from airflow import DAG, Dataset
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import logging
import pandas as pd

# Define paths for ingested datasets
train_ingested_data_path = "data/train.csv"
oil_ingested_data_path = "data/oil.csv"

# Define the Dataset objects to register datasets
data_ingestion_dataset = Dataset(train_ingested_data_path)
oil_ingestion_dataset = Dataset(oil_ingested_data_path)

# Task 1: Get time frame from the configuration
def get_time_frame(**context):
    logging.info("Fetching time frame configuration (start_date, end_date)...")
    
    # Access DAG run configuration (start_date and end_date)
    conf = context["dag_run"].conf
    logging.info(f"Configuration: {conf}")
    
    start_date = pd.to_datetime(conf['start_date'])
    end_date = pd.to_datetime(conf['end_date'])
    
    # Pass the start_date and end_date to the next tasks in the XCom
    context['ti'].xcom_push(key='start_date', value=start_date.isoformat())
    context['ti'].xcom_push(key='end_date', value=end_date.isoformat())
    
    return "Time frame fetched!"

# Task 2: Ingest data from the source (train and oil data)
def ingest_data_task(**context):
    from helpers import ingest_data

    logging.info("INGESTING DATA")

    # Fetch start_date and end_date from XCom (previous task)
    start_date = context['ti'].xcom_pull(task_ids='get_time_frame', key='start_date')
    end_date = context['ti'].xcom_pull(task_ids='get_time_frame', key='end_date')
    
    logging.info(f"Using start_date: {start_date} and end_date: {end_date}")
    
    # Ingest data
    df_train, df_oil = ingest_data(start_date, end_date)
    
    # Push dataframes to XCom for use by other tasks if necessary
    context['ti'].xcom_push(key='df_train', value=df_train)
    context['ti'].xcom_push(key='df_oil', value=df_oil)

    return "Data ingested!"

# Task 3: Write train data to CSV
def write_train_data_to_csv(**context):
    logging.info("Writing training data to CSV")

    # Fetch df_train from XCom (previous task)
    df_train = context['ti'].xcom_pull(task_ids='ingest_data_task', key='df_train')

    # Save the training data to CSV
    df_train.to_csv(train_ingested_data_path, index=True)
    logging.info(f"Training data saved to {train_ingested_data_path}")
    
    # Register the dataset
    # data_ingestion_dataset.push(df_train)
    
    return "Training data written to CSV!"

# Task 4: Write oil data to CSV
def write_oil_data_to_csv(**context):
    logging.info("Writing oil data to CSV")

    # Fetch df_oil from XCom (previous task)
    df_oil = context['ti'].xcom_pull(task_ids='ingest_data_task', key='df_oil')

    # Save the oil data to CSV
    df_oil.to_csv(oil_ingested_data_path, index=False)
    logging.info(f"Oil data saved to {oil_ingested_data_path}")

    # Register the dataset
    # oil_ingestion_dataset.push(df_oil)
    
    return "Oil data written to CSV!"

# Define the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'ingest_data_DAG',
    default_args=default_args,
    description='MLOps Dashboard DAG with Granular Subtasks',
    schedule_interval=None,  # Set to None for manual triggering
)

# Define tasks and subtasks
get_time_frame_task = PythonOperator(
    task_id='get_time_frame',
    python_callable=get_time_frame,
    dag=dag,
    provide_context=True
)

ingest_data_task = PythonOperator(
    task_id='ingest_data_task',
    python_callable=ingest_data_task,
    dag=dag,
    provide_context=True
)

write_train_data_task = PythonOperator(
    task_id='write_train_data_to_csv',
    python_callable=write_train_data_to_csv,
    dag=dag,
    provide_context=True,
    outlets=[data_ingestion_dataset]
)

write_oil_data_task = PythonOperator(
    task_id='write_oil_data_to_csv',
    python_callable=write_oil_data_to_csv,
    dag=dag,
    provide_context=True,
    outlets=[oil_ingestion_dataset]
)

# Define task dependencies (execution order)
get_time_frame_task >> ingest_data_task
ingest_data_task >> write_train_data_task
ingest_data_task >> write_oil_data_task
