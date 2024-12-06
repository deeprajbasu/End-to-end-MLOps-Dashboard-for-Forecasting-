import streamlit as st
import pandas as  pd
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore

from darts.dataprocessing.transformers import MissingValuesFiller
from darts import TimeSeries
from darts.models import Theta
import optuna
from darts.models import Prophet
from darts.utils.utils import ModelMode, SeasonalityMode

import numpy as np

from darts.models import ExponentialSmoothing

import yaml

import concurrent.futures
import optuna

opuna_storage_url = "postgresql://postgres:deepraj@localhost:5432/optuna_studies"

def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

# Save configuration to YAML file
def save_config(config):
    with open("config.yaml", "w") as file:
        yaml.dump(config, file)

# Function to ingest data
@st.cache_data
def ingest_data(start_date, end_date):
    # Load and filter the train.csv data
    df_train = (
        pd.read_csv('train.csv', parse_dates=['date'])
        .query('date >= @start_date and date <= @end_date')
    )

    df_oil = (
        pd.read_csv('oil.csv', parse_dates=['date'])
        .query('date >= @start_date and date <= @end_date')
    )

    return df_train,df_oil


# Function to plot daily sales with moving averages
@st.cache_data
def plot_sales_with_moving_avg(df_train, selected_families):
    # Filter data for selected families
    df_selected = df_train[df_train['family'].isin(selected_families)]
    
    # Calculate daily total sales
    df_grouped = df_selected.set_index("date").groupby("family").resample("D").sales.sum().reset_index()

    # Calculate the 7-day moving average with minimum periods set to 2
    df_grouped['moving_avg'] = df_grouped.groupby('family')['sales'].transform(lambda x: x.rolling(window=7, min_periods=2).mean())
    
    # Create a new Plotly figure
    fig = px.line(df_grouped, x="date", y="sales", color="family", title=f"Daily Total Sales of {', '.join(selected_families)}")
    # Set low opacity for the raw data traces
    for trace in fig.data:
        trace.opacity = 0.2
    # Overlay the moving averages
    for family in df_grouped['family'].unique():
        family_data = df_grouped[df_grouped['family'] == family]
        fig.add_scatter(x=family_data['date'], y=family_data['moving_avg'], mode='lines', name=f'{family} Moving Avg')
    
    # Calculate date range for last month
    max_date = df_grouped['date'].max()
    start_last_month = max_date - timedelta(days=30)

    # Add a vertical rectangle to highlight the last month
    fig.add_vrect(x0=start_last_month, x1=max_date, fillcolor="LightSalmon", opacity=0.3, line_width=0)
    
    # Update layout for better visualization
    fig.update_layout(title=f"Daily Total Sales with Moving Averages ({', '.join(selected_families)})", 
                      xaxis_title='Date', yaxis_title='Sales')


    return fig

# Function to display oil data
@st.cache_data
def display_oil_data(df_oil):
    # Fill missing dates with NaN, then forward-fill values
    df_oil = df_oil.set_index("date").resample("D").ffill().reset_index()

    # Melt the DataFrame for plotting
    p = df_oil.melt(id_vars=['date'] + list(df_oil.keys()[5:]), var_name='Legend')
    
    # Create a new Plotly figure for the oil data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p['date'],
        y=p['value'],
        mode='lines+markers',
        marker=dict(size=5),
        line=dict(width=1),
        text=p['Legend'],
        name='Oil Price',
        showlegend=True,
    ))

    # Calculate the date range for the last month
    max_date = df_oil['date'].max()
    start_last_month = max_date - timedelta(days=30)

    # Add a vertical rectangle for the last month
    fig.add_vrect(x0=start_last_month, x1=max_date, 
                  fillcolor="LightSalmon", opacity=0.3, line_width=0)

    # Update layout for better visualization
    fig.update_layout(title="Oil Prices Over Time", 
                      xaxis_title='Date', 
                      yaxis_title='Price', 
                      legend_title='Oil Price Categories',
                      showlegend=False)

    # Return the figure
    return fig

# Function to plot sales distribution overlay
def plot_sales_distribution_overlay(df_train, families):
    # Ensure the date column is in datetime format
    df_train['date'] = pd.to_datetime(df_train['date'])
    
    # Create a copy of the DataFrame for processing
    df_copy = df_train.copy()
    
    # Define date ranges
    end_date = df_copy['date'].max()
    last_month_start = end_date - timedelta(days=30)
    last_six_months_start = end_date - timedelta(days=90)
    last_year_start = end_date - timedelta(days=365)
    
    # Prepare data for the last month, last six months, and last year
    last_month_data = df_copy[df_copy['date'] >= last_month_start]
    last_six_months_data = df_copy[(df_copy['date'] < last_month_start) & 
                                    (df_copy['date'] >= last_six_months_start)]
    last_year_data = df_copy[df_copy['date'] >= last_year_start]

    # Create subplots: two columns per row
    num_families = len(families)
    num_columns = 2
    num_rows = (num_families + num_columns - 1) // num_columns  # Calculate number of rows needed

    fig = make_subplots(rows=num_rows, cols=num_columns,
                        subplot_titles=families)

    # Plot normal distribution for each family
    for i, family in enumerate(families):
        # Get sales data for the last month, last six months, and last year
        last_month_sales = last_month_data[last_month_data['family'] == family]['sales']
        last_six_months_sales = last_six_months_data[last_six_months_data['family'] == family]['sales']
        last_year_sales = last_year_data[last_year_data['family'] == family]['sales']

        # Calculate row and column for the current family
        row = i // num_columns + 1
        col = i % num_columns + 1

        # Last Month Distribution
        if not last_month_sales.empty:
            mu_last_month, std_last_month = last_month_sales.mean(), last_month_sales.std()
            x_last_month = np.linspace(mu_last_month - 4*std_last_month, mu_last_month + 4*std_last_month, 100)
            y_last_month = stats.norm.pdf(x_last_month, mu_last_month, std_last_month)

            # Add the last month plot to the subplot
            fig.add_trace(go.Scatter(
                x=x_last_month,
                y=y_last_month,
                mode='lines',
                name=f'latest',
                line=dict(width=2)
            ), row=row, col=col)

        # Last Six Months Distribution
        if not last_six_months_sales.empty:
            mu_last_six_months, std_last_six_months = last_six_months_sales.mean(), last_six_months_sales.std()
            x_last_six_months = np.linspace(mu_last_six_months - 4*std_last_six_months, mu_last_six_months + 4*std_last_six_months, 100)
            y_last_six_months = stats.norm.pdf(x_last_six_months, mu_last_six_months, std_last_six_months)

            # Add the last six months plot to the subplot
            fig.add_trace(go.Scatter(
                x=x_last_six_months,
                y=y_last_six_months,
                mode='lines',
                name=f'-6 Months',
                line=dict(width=2, dash='dash')  # Dashed line for last six months
            ), row=row, col=col)

        # Last Year Distribution
        if not last_year_sales.empty:
            mu_last_year, std_last_year = last_year_sales.mean(), last_year_sales.std()
            x_last_year = np.linspace(mu_last_year - 4*std_last_year, mu_last_year + 4*std_last_year, 100)
            y_last_year = stats.norm.pdf(x_last_year, mu_last_year, std_last_year)

            # Add the last year plot to the subplot
            fig.add_trace(go.Scatter(
                x=x_last_year,
                y=y_last_year,
                mode='lines',
                name=f'-1Year',
                line=dict(width=2, dash='dot')  # Dotted line for last year
            ), row=row, col=col)

    # Update layout for better visualization
    fig.update_layout(
        title="Sales Distribution: Overlay of Last Month, Last 6 Months, and Last Year",
        xaxis_title='Sales',
        yaxis_title='Density',
        template='plotly_white',
        height=300 * num_rows,
        showlegend=False  # Adjust height based on the number of rows
    )

    # Show the plot
    return fig

def plot_sales_seasonality(df_train, families):
    # Ensure the date column is in datetime format
    df_train['date'] = pd.to_datetime(df_train['date'])
    
    # Create a copy of the DataFrame for processing
    df_copy = df_train.copy()
    
    # Define date ranges
    end_date = df_copy['date'].max()
    last_month_start = end_date - timedelta(days=30)
    last_six_months_start = end_date - timedelta(days=90)
    last_year_start = end_date - timedelta(days=365)
    
    # Create subplots: two columns per row for each family's seasonality plot
    num_families = len(families)
    num_columns = 2
    num_rows = (num_families + num_columns - 1) // num_columns  # Calculate the number of rows needed

    # Seasonality Figure
    seasonality_fig = make_subplots(rows=num_rows, cols=num_columns,
                                    subplot_titles=[f"{family} Seasonality" for family in families])

    for i, family in enumerate(families):
        # Get sales data for each period
        family_data = df_copy[df_copy['family'] == family]
        last_month_data = family_data[family_data['date'] >= last_month_start]
        last_six_months_data = family_data[(family_data['date'] >= last_six_months_start) & 
                                           (family_data['date'] < last_month_start)]
        last_year_data = family_data[family_data['date'] >= last_year_start]

        # Calculate row and column for the current family
        row = i // num_columns + 1
        col = i % num_columns + 1

        for period_name, data in [('Last Month', last_month_data), 
                                  ('Last 6 Months', last_six_months_data), 
                                  ('Last Year', last_year_data)]:
            if len(data) > 10:  # Ensuring we have enough data points for decomposition
                # Resample to daily data if needed
                data_resampled = data.set_index('date').resample('D').sum().fillna(0)

                # Decompose the time series into seasonal components
                decomposition = seasonal_decompose(data_resampled['sales'], model='additive', period=15)
                
                # Normalize the seasonal component using z-score
                seasonal_normalized = zscore(decomposition.seasonal)

                # Plot the normalized seasonality component
                seasonality_fig.add_trace(go.Scatter(
                    x=data_resampled.index[-180:],
                    y=seasonal_normalized,
                    mode='lines',
                    name=f'{family} {period_name} Seasonality (Normalized)',
                    line=dict(width=2, dash='solid' if period_name == 'Last Month' else 'dash' if period_name == 'Last 6 Months' else 'dot')
                ), row=row, col=col)

    # Update layout for seasonality plot
    seasonality_fig.update_layout(
        title="Normalized Sales Seasonality: Overlay of Last Month, Last 6 Months, and Last Year",
        xaxis_title='Date',
        yaxis_title='Normalized Seasonality',
        template='plotly_white',
        height=300 * num_rows,
        showlegend=False  # Adjust height based on the number of rows
    )

    # Show the seasonality plot
    return seasonality_fig



# Objective function for hyperparameter tuning
def prophet_objective(trial, time_series):
    # Define the hyperparameters to tune
    seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
    changepoint_prior_scale = trial.suggest_float("changepoint_prior_scale", 0.01, 0.5)
    seasonality_prior_scale = trial.suggest_float("seasonality_prior_scale", 0.01, 10.0)
    weekly_seasonality = trial.suggest_categorical("weekly_seasonality", [True, False])

    # Initialize and fit the model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        weekly_seasonality=weekly_seasonality,
    )
    
    model.fit(time_series)

    # Perform historical forecasts for backtesting
    backtest_results = model.historical_forecasts(
        time_series,
        start=int(len(time_series) / 1.5),
        forecast_horizon=60,
        stride=15,
        retrain=True,
        last_points_only=False,
        verbose=False,
    )

    # Calculate the average error
    cumulative_error = 0
    for b in backtest_results:
        actual_values = time_series.slice(b.time_index[0], b.time_index[-1]).values().flatten()
        forecast_values = b.values().flatten()

        max_actual = actual_values.max()
        divisor = 10 ** (len(str(int(max_actual))) - 1)  # Calculate divisor based on number of zeros


        # Calculate error while ignoring NaN values
        mask = ~np.isnan(actual_values)
        cumulative_error += (np.mean(abs(actual_values[mask] - forecast_values[mask])))/divisor

    avg_error = cumulative_error / len(backtest_results)

        # Log trial parameters and error to MLflow
    # mlflow.log_params({
    #     "seasonality_mode": seasonality_mode,
    #     "changepoint_prior_scale": changepoint_prior_scale,
    #     "seasonality_prior_scale": seasonality_prior_scale,
    #     "weekly_seasonality": weekly_seasonality,
    # })
    # mlflow.log_metric("avg_error", avg_error)
    
    
    return avg_error  # Lower is better

def exp_objective(trial, time_series):
    # Define the hyperparameters with serialized values
    trend = trial.suggest_categorical("trend", ["none", "additive",])
    seasonal = trial.suggest_categorical("seasonal", ["none", "additive", "multiplicative"])
    seasonal_periods = trial.suggest_int("seasonal_periods", 7, 30)  # Adjust range as needed
    
    # Map string selections back to enums as needed by the ExponentialSmoothing model
    trend_mode = None if trend == "none" else ModelMode[trend.upper()]
    seasonal_mode = None if seasonal == "none" else SeasonalityMode[seasonal.upper()]
    
    model = ExponentialSmoothing(
        trend=trend_mode,
        seasonal=seasonal_mode,
        seasonal_periods=seasonal_periods,
    )
    
    model.fit(time_series)
    
    # Perform historical forecasts for backtesting
    backtest_results = model.historical_forecasts(
        time_series,
        start=int(len(time_series) / 1.5),
        forecast_horizon=60,
        stride=15,
        retrain=True,
        last_points_only=False,
    )

    # Calculate the average error
    cumulative_error = 0
    for b in backtest_results:
        actual_values = time_series.slice(b.time_index[0], b.time_index[-1]).values().flatten()
        forecast_values = b.values().flatten()

        max_actual = actual_values.max()
        divisor = 10 ** (len(str(int(max_actual))) - 1)  # Calculate divisor based on number of zeros


        # Calculate error while ignoring NaN values
        mask = ~np.isnan(actual_values)
        cumulative_error += (np.mean(abs(actual_values[mask] - forecast_values[mask])))/divisor

    avg_error = cumulative_error / len(backtest_results)
    return avg_error  # Lower is better

def theta_objective(trial,time_series):
    # Define the hyperparameters to tune
    theta = trial.suggest_float("theta", 0.1, 1.0)  # Theta parameter

    # Initialize and fit the model
    model = Theta(
        theta=theta,
    )
    
    model.fit(time_series)

    # Perform historical forecasts for backtesting
    backtest_results = model.historical_forecasts(
        time_series,
        start=int(len(time_series) / 1.5),
        forecast_horizon=60,
        stride=15,
        retrain=True,
        last_points_only=False,
    )

    # Calculate the average error
    cumulative_error = 0
    for b in backtest_results:
        actual_values = time_series.slice(b.time_index[0], b.time_index[-1]).values().flatten()
        forecast_values = b.values().flatten()

        max_actual = actual_values.max()
        divisor = 10 ** (len(str(int(max_actual))) - 1)  # Calculate divisor based on number of zeros


        # Calculate error while ignoring NaN values
        mask = ~np.isnan(actual_values)
        cumulative_error += (np.mean(abs(actual_values[mask] - forecast_values[mask])))/divisor

    avg_error = cumulative_error / len(backtest_results)
    
    return avg_error  # Lower is better





def plot_model(model, model_ts,family):

    # Step 5: Make future forecast predictions
    forecast = model.predict(n=60)  # Forecast the next 60 days

    # Step 6: Perform historical forecasts for backtesting
    backtest_results = model.historical_forecasts(
        model_ts,
        start=int(len(model_ts) / 1.5),
        forecast_horizon=60,
        stride=15,
        retrain=True,
        last_points_only=False,
    )

    # Step 7: Create a Plotly figure
    fig = go.Figure()

    # Plot actual sales data
    fig.add_trace(go.Scatter(x=model_ts.time_index, y=model_ts.values().flatten(), mode='lines', name='Sales',opacity=0.55,
    line=dict(width=3)))

    # Plot each backtest forecast result
    error_cum = 0
    for i, b in enumerate(backtest_results):
        # Extract the corresponding actual values from the TimeSeries for the forecast period
        actual_values = model_ts.slice(b.time_index[0], b.time_index[-1]).values().flatten()

        max_actual = actual_values.max()
        divisor = 10 ** (len(str(int(max_actual))) - 1)  # Calculate divisor based on number of zeros

        err = np.mean(abs(actual_values - b.values().flatten()))/divisor
        error_cum += err
        err_str = f"{err:.2f}"
        fig.add_trace(go.Scatter(x=b.time_index, y=b.values().flatten(), name='Backtest ' + str(i) + ' Mean Error: ' + err_str))

    # Plot the future forecast (next 60 days)
    fig.add_trace(go.Scatter(x=forecast.time_index, y=forecast.values().flatten(),
                             mode='lines', name='Future Forecast (Next 60 Days)', line=dict(color='orange')))

    # Update layout
    fig.update_layout(
        title=f'{family} Forecast with Backtests using Optimized {model.__class__.__name__} | Mean Error: {error_cum / len(backtest_results):.2f}',
        xaxis_title='Date',
        yaxis_title='Sales',
        template='plotly_white'
    )

    return fig  # Return the figure to be displayed or used later

def plot_ensemble(theta_forecast,prophet_forecast,exp_forecast,ts):
    fig = go.Figure()

    # Plot actual sales data
    fig.add_trace(go.Scatter(
        x=ts.time_index,
        y=ts.values().flatten(),
        mode='lines',
        name='Sales'
    ))

    # Assuming thetaforecast, prophetforecast, and expforecast are Darts TimeSeries objects
    # Convert them to pandas DataFrame if necessary
    theta_values = theta_forecast.values().flatten()
    prophet_values = prophet_forecast.values().flatten()
    exp_values = exp_forecast.values().flatten()

    # Calculate min and max forecasts for each day
    min_forecast = np.minimum.reduce([theta_values, prophet_values, exp_values])
    max_forecast = np.maximum.reduce([theta_values, prophet_values, exp_values])

    spread_factor = .0012 # Adjust this factor to control the spread
    min_forecast_spread = min_forecast * np.exp(-spread_factor * np.arange(len(min_forecast)))
    max_forecast_spread = max_forecast * np.exp(spread_factor * np.arange(len(max_forecast)))


    # Create time index for forecasts
    forecast_time_index = theta_forecast.time_index  # Assuming all forecasts have the same time index

    # Overlay min and max regions
    fig.add_trace(go.Scatter(
        x=forecast_time_index,
        y=min_forecast_spread,
        mode='lines',
        line=dict(color='rgba(0, 0, 0, 0)', width=0),  # Invisible line for min
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast_time_index,
        y=max_forecast_spread,
        mode='lines',
        line=dict(color='rgba(0, 0, 0, 0)', width=0),  # Invisible line for max
        fill='tonexty',
        name='Forecast Range',
        fillcolor='rgba(53, 230, 200, 0.7)'  # Adjust transparency as needed
    ))

    # Update layout
    fig.update_layout(
        title='Sales Data with Combined Forecast Regions',
        xaxis_title='Date',
        yaxis_title='Sales',
        template='plotly_white'
    )

    # Show the plot
    return fig

# Function to train Prophet model (example)
def optimize_prophet(model_ts):
    study = optuna.create_study(direction="minimize", storage=opuna_storage_url,)
    study.optimize(lambda trial: prophet_objective(trial, model_ts), n_trials=20)
    return study.best_params, study.best_value

# Function to train Exponential Smoothing model (example)
def optimize_exponential_smoothing(model_ts):
    study = optuna.create_study(direction="minimize", storage=opuna_storage_url,)
    study.optimize(lambda trial: exp_objective(trial, model_ts), n_trials=20)
    return study.best_params, study.best_value

# Function to train Theta model (example)
def optimize_theta(model_ts):
    study = optuna.create_study(direction="minimize", storage=opuna_storage_url,)
    study.optimize(lambda trial: theta_objective(trial, model_ts), n_trials=20)
    return study.best_params, study.best_value


# The main function that trains models and handles the Optuna studies
def train_models(df_train, family):
    # Filter and prepare the family-specific data (same as before)
    df_model = df_train[df_train['family'] == family]
    df_model = df_model.resample("D", on="date").agg({
        'sales': 'sum',
        'family': 'first'
    }).reset_index()
    df_model['date'] = pd.to_datetime(df_model['date'])
    df_model.set_index('date', inplace=True)
    df_model['sales'] = df_model['sales'].rolling(window=3).mean()
    df_model['sales'] = df_model['sales'].fillna(df_model['sales'].mean())
    # Convert the DataFrame to TimeSeries for the models (same as before)
    model_ts = TimeSeries.from_dataframe(df_model, value_cols='sales', fill_missing_dates=True, freq='D')
    filler = MissingValuesFiller()
    model_ts = filler.transform(model_ts)

    # Parallelize the Optuna studies for Prophet, Exponential Smoothing, and Theta models
    st.write(f'{family} study started')
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            'prophet': executor.submit(optimize_prophet, model_ts),
            'exponential_smoothing': executor.submit(optimize_exponential_smoothing, model_ts),
            'theta': executor.submit(optimize_theta, model_ts)
        }

        # Wait for all studies to complete and get their results
        results = {}
        for model, future in futures.items():
            try:
                best_params, best_value = future.result()
                results[model] = {'params': best_params, 'value': best_value}
            except Exception as e:
                st.error(f"An error occurred during {model} optimization: {e}")

    # Once studies are complete, train the models using the best parameters
    # Prophet model
    prophet_params = results['prophet']['params']
    prophet_model = Prophet(
        changepoint_prior_scale=prophet_params['changepoint_prior_scale'],
        seasonality_prior_scale=prophet_params['seasonality_prior_scale'],
        seasonality_mode=prophet_params['seasonality_mode'],
        weekly_seasonality=prophet_params['weekly_seasonality']
    )
    prophet_model.fit(model_ts)
    prophet_forecast = prophet_model.predict(n=60)
    prophet_model_fig = plot_model(prophet_model, model_ts, family)

    # Exponential Smoothing model
    exp_params = results['exponential_smoothing']['params']
    trend_mode = None if exp_params["trend"] == "none" else ModelMode[exp_params["trend"].upper()]
    seasonal_mode = None if exp_params["seasonal"] == "none" else SeasonalityMode[exp_params["seasonal"].upper()]
    seasonal_periods = exp_params['seasonal_periods']

    exp_model = ExponentialSmoothing(
        trend=trend_mode,
        seasonal=seasonal_mode,
        seasonal_periods=seasonal_periods
    )
    exp_model.fit(model_ts)
    exp_forecast = exp_model.predict(n=60)
    exp_model_fig = plot_model(exp_model, model_ts, family)

    # Theta model
    theta_params = results['theta']['params']
    theta_model = Theta(theta=theta_params['theta'])
    theta_model.fit(model_ts)
    theta_forecast = theta_model.predict(n=60)
    theta_model_fig = plot_model(theta_model, model_ts, family)

    # Ensemble forecast (using the 3 models' forecasts)
    ensemble_fig = plot_ensemble(theta_forecast, prophet_forecast, exp_forecast, model_ts)

    # Return the ensemble figure to be plotted later
    return [prophet_model_fig,exp_model_fig,theta_model_fig,ensemble_fig]


# Wrapper function for model training and logging within Streamlit
def train_family(family_name, df_train):

    st.write(f'Starting training for {family_name}')
    return train_models(df_train, family_name)
