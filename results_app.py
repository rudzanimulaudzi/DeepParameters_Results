import asyncio  # Import asyncio
from playwright.async_api import async_playwright
import math
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import re
import json  

import os
from io import BytesIO

import base64
from datetime import datetime

import urllib.parse
import subprocess
import threading

import Utils as utility_function
import logging

import time

# Configure logging
# Configure logging
date_today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_directory = 'outputs/logs/'
utility_function.ensure_directory_exists(log_directory)  # Ensure the log directory exists
filename = f'{log_directory}{date_today}_streamlit_visualize.log'

# Create logger
visual_logger = logging.getLogger(__name__)
visual_logger.setLevel(logging.ERROR)  # Set the logger's level to DEBUG to capture all logs

# Create file handler for logging to a file
visual_logger_handler = logging.FileHandler(filename, mode='w')
visual_logger_handler.setLevel(logging.ERROR)  # Set the handler's level

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
visual_logger_handler.setFormatter(formatter)

# Add the handler to the logger
visual_logger.addHandler(visual_logger_handler)

visual_logger.propagate = False  # Avoid passing logs up to the root logger

# Function to log current selections
def log_current_selections(network_type, sample_sizes, parameter_sizes, selected_max_indegrees, selected_create_dates, selected_run_names, selected_skews, selected_noise, selected_densities):
    log_path = 'datasets/rmulaudzi/outputs/snapshots/key_settings_stored.log'
    utility_function.ensure_directory_exists(os.path.dirname(log_path))  # Ensure the directory exists
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n\nSnapshot taken at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Network Type: {network_type}\n")
        log_file.write(f"Sample Sizes: {sample_sizes}\n")
        log_file.write(f"Parameter Sizes: {parameter_sizes}\n")
        log_file.write(f"Max Indegrees: {selected_max_indegrees}\n")
        log_file.write(f"Create Dates: {selected_create_dates}\n")
        log_file.write(f"Run Names: {selected_run_names}\n")
        log_file.write(f"Skews: {selected_skews}\n")
        log_file.write(f"Noise Range: {selected_noise}\n")
        log_file.write(f"Densities: {selected_densities}\n")
    
    selection_log = {
        "network_type": network_type,
        "sample_sizes": sample_sizes,
        "parameter_sizes": parameter_sizes,
        "selected_max_indegrees": selected_max_indegrees,
        "selected_create_dates": selected_create_dates,
        "selected_run_names": selected_run_names,
        "selected_skews": selected_skews,
        "selected_noise": selected_noise,
        "selected_densities": selected_densities
    }

    visual_logger.info(f"Key settings: {selection_log}")
# Database Connection
def get_database_engine():
    # Load database configuration from json file
    with open('local_db_config.json') as config_file:
        db_config = json.load(config_file)
        
    # Construct database URI from configuration
    #print('db_config', db_config)
    # URL encode the password to handle special characters
    encoded_password = urllib.parse.quote_plus(db_config['password'])
    
    db_uri = f"postgresql+psycopg2://{db_config['user']}:{encoded_password}@{db_config['host']}/{db_config['dbname']}"
    engine = create_engine(db_uri)
    
    visual_logger.info(f"Database engine created successfully: {engine}") # Log success message

    return engine

def get_unique_run_dates(run_names):
    """Extract unique dates from run names."""
    date_pattern = re.compile(r'\d{14}$')  # Matches YYYYMMDDHHMMSS at the end of the string
    dates = set()
    
    for run_name in run_names:
        match = date_pattern.search(run_name)
        if match:
            date_str = match.group()
            # Extract YYYY-MM-DD part only for simplification
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            dates.add(formatted_date)
    
        visual_logger.info(f"Unique dates extracted from run name: {run_name} is date: {dates}")  # Log the unique dates
    
    return dates

def fetch_unique_create_dates(table='metrics_data'):
    """Fetch unique run dates up to the minute."""
    engine = get_database_engine()
    query = f"SELECT DISTINCT CAST(created_at AS VARCHAR) FROM {table};"
    df = pd.read_sql_query(sql=text(query), con=engine.connect())
    
    # Process dates to keep up to minutes
    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Get unique dates
    unique_dates_list = sorted(df['created_at'].unique().tolist())

    visual_logger.info(f"Unique dates extracted from the database: {unique_dates_list}")  # Log the unique dates

    return unique_dates_list

def fetch_unique_run_dates(table='metrics_data'):
    engine = get_database_engine()
    query = f"SELECT DISTINCT run_name FROM {table};"
    df = pd.read_sql_query(sql=text(query), con=engine.connect())
    
    # Extract unique dates from run_name values
    unique_dates = set()
    for run_name in df['run_name']:
        date_part = run_name.split('_')[-1][:8]  # Assuming the date is always at the end and in YYYYMMDD format
        unique_dates.add(date_part)
    
    # Convert set to sorted list
    unique_dates_list = sorted(list(unique_dates))

    visual_logger.info(f"Unique dates extracted from the database: {unique_dates_list}")  # Log the unique dates

    return unique_dates_list

def fetch_unique_values(column_name, table='metrics_data'):
    engine = get_database_engine()
    query = f"SELECT DISTINCT {column_name} FROM {table} ORDER BY {column_name};"
    df = pd.read_sql_query(sql=text(query), con=engine.connect())

    visual_logger.info(f"Unique values extracted from the database for column: {column_name} are: {df[column_name].tolist()}")  # Log the unique values

    return df[column_name].tolist()

def fetch_data(network_type='naive', sample_size_range=None, parameter_size_range=None, max_indegrees=None, create_dates = None, run_names=None, skews=None, noise=None, densities=None, include_missing_data=False, missing_percentage_range=None, table='metrics_data'):
    engine = get_database_engine()

    query_conditions = []

    if network_type:
        network_types_formatted = ", ".join(f"'{nt}'" for nt in network_type)
        query_conditions.append(f"network_type IN ({network_types_formatted})")
    if sample_size_range:
        query_conditions.append(f"sample_sizes BETWEEN {sample_size_range[0]} AND {sample_size_range[1]}")
    if parameter_size_range:
        query_conditions.append(f"num_parameters BETWEEN {parameter_size_range[0]} AND {parameter_size_range[1]}")
    if max_indegrees:
        max_indegrees_str = ", ".join(str(mi) for mi in max_indegrees)
        query_conditions.append(f"max_indegree IN ({max_indegrees_str})")
    if noise:
        query_conditions.append(f"noise BETWEEN {noise[0]} AND {noise[1]}")
    if densities:
        densities_str = ", ".join(f"'{density}'" for density in densities)
        query_conditions.append(f"density IN ({densities_str})")
    if create_dates:
        # Convert the `create_dates` to a datetime range condition
        date_conditions = []
        for date in create_dates:
            # Assuming `date` is in 'YYYY-MM-DD HH:MM' format
            start_datetime = pd.to_datetime(date)
            end_datetime = start_datetime + pd.Timedelta(minutes=1)
            
            # Format these datetime objects into a string that PostgreSQL understands
            start_datetime_str = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_str = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add the BETWEEN condition for this date range
            date_conditions.append(f"created_at BETWEEN '{start_datetime_str}' AND '{end_datetime_str}'")
            
        # Combine all date conditions with OR
        combined_date_conditions = " OR ".join(date_conditions)
        query_conditions.append(f"({combined_date_conditions})")
    if run_names:
        date_conditions = " OR ".join(f"run_name LIKE '%%{date}%%'" for date in run_names)
        query_conditions.append(f"({date_conditions})")
    if skews:
        skews_str = ", ".join(str(s) for s in skews)
        query_conditions.append(f"skew IN ({skews_str})")
    #if noise:
    #    noise_str = ", ".join(str(n) for n in noise)
    #    query_conditions.append(f"noise IN ({noise_str})")
    if include_missing_data and missing_percentage_range:
        query_conditions.append(f"missing_percentage BETWEEN {missing_percentage_range[0]} AND {missing_percentage_range[1]}")
    else:
        query_conditions.append(f"missing_percentage = 0")

    query_condition = " AND ".join(query_conditions) if query_conditions else "TRUE"
    #query = f"SELECT * FROM {table} WHERE {query_condition} ORDER BY sample_sizes DESC;"
    query = f"SELECT * FROM {table} WHERE {query_condition} ORDER BY sample_sizes DESC, created_at DESC;"

    df = pd.read_sql_query(sql=text(query), con=engine.connect())

    visual_logger.info(f"Data fetched from the database with the following query: {query}")  # Log the query condition
    visual_logger.info(f"Data df fetched from the database: {df.head()}")  # Log the query condition

    return df

#Download file
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()

    visual_logger.info(f"Download link generated for file: {download_filename}")  # Log the download link

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def save_plot_and_create_download_link(fig, filename):
    """
    Saves the plot as an SVG and returns a download link.
    """
    output_dir = 'datasets/rmulaudzi/outputs/graphs'
    utility_function.ensure_directory_exists(output_dir)  # Ensure the log directory exist
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, format='svg')
    
    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)  # Ensure the directory exists
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    fig.savefig(filepath, format='pdf')  # Save as PDF
    
    try:
        with open(filepath, "rb") as f:
            bytes = f.read()
            b64 = base64.b64encode(bytes).decode()
            href = f'<a href="data:file/svg;base64,{b64}" download="{filename}">Download {filename}</a>'

        visual_logger.info(f"Download link generated for file: {filename}")  # Log the download link

        return href
    
    except Exception as e:
        visual_logger.error(f"Error downloading file: {e}")  # Log the error
        return f"Error downloading file: {e}"
    
def save_plot(fig, format='pdf'):
    """
    Saves the plot to a BytesIO object and returns it.
    """
    buf = BytesIO()
    fig.savefig(buf, format=format)
    buf.seek(0)
    
    return buf

# Visualization Functions
import os
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt

def plot_run_contour(metrics_df, experiment_id = None, x='sample_sizes', y='num_parameters', metrics=[
    'kl_divergence_dl', 'kl_divergence_thresholding_dl', 'kl_divergence_smoothing_dl',
    'kl_divergence_mle', 'kl_divergence_thresholding_mle', 'kl_divergence_smoothing_mle',
    'kl_divergence_map', 'kl_divergence_thresholding_map', 'kl_divergence_smoothing_map'], filename = None):
    
    # Ensure there are at least 3 data points
    if metrics_df.shape[0] < 3:
        st.warning("Not enough data points to generate contour plots. Please adjust your selection criteria.")
        return  # Exit the function early
    
    # Load the data from your DataFrame
    df = metrics_df.copy()
    
    # Determine the number of rows and columns for subplots based on the number of metrics
    num_metrics = len(metrics)
    num_cols = int(math.ceil(math.sqrt(num_metrics)))
    num_rows = int(math.ceil(num_metrics / num_cols))
    
    # Define color maps for different plots
    cmap_ = 'Blues'
    
    # Create a 3 by 3 subplot graph
    #fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    #fig.subplots_adjust(right=0.8)
    
    # Create the subplot grid
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if num_metrics == 1:
        axs = [axs]  # If there is only one metric, put it in a list for iteration
    
    # Flatten the axs array for easy iteration
    axs = axs.flatten()
    
    # Iterate through the flattened axs and corresponding KL divergences
    for ax, kl_div in zip(axs, metrics):
        contour = ax.tricontourf(df[x], df[y], df[kl_div], cmap=cmap_)
        fig.colorbar(contour, ax=ax)
        #ax.set_title(kl_div.replace('_', ' ').title())
        # Set the title based on the metric type (DL, MLE, MAP, Random)
        if 'dl' in kl_div:
            title = 'Deep Learning'
        elif 'mle' in kl_div:
            title = 'MLE'
        elif 'map' in kl_div:
            title = 'BPE'
        elif 'random' in kl_div:
            title = 'Random'
        else:
            title = kl_div.replace('_', ' ').title()

        # Set the title for the plot
        ax.set_title(title)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Parameter Size")

    # Remove any extra subplots
    for ax in axs[num_metrics:]:
        ax.remove()

    #fig.suptitle('Contour Plots of KL Divergence', fontsize=16)
    plt.tight_layout()
    #plt.tight_layout(rect=[0, 0, 0.8, 0.95])      

    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)  # Ensure the directory exists
    filename = f"contour_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    #fig.savefig(filepath, format='pdf')  # Save as PDF
    
    # Create a BytesIO buffer to hold the pdf bytes
    buf = save_plot(fig)

    # Display the plot
    st.pyplot(fig)
    
    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )
    
def plot_contour(df):
    # Assuming 'sample_sizes' and 'kl_divergence_dl' are the actual column names
    # Check if melted_df is empty
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return  # Exit the function early
    
    # Load the data from your DataFrame
    df1 = df.copy()
    
    # Clean the DataFrame to remove or handle non-finite values
    df1 = df1.fillna(0)  # Replace NaN values with 0
    df1.replace([float('inf'), float('-inf')], 0, inplace=True)  # Replace inf and -inf with 0
    
    plot_run_contour(metrics_df = df1,
                 x = 'sample_sizes', 
                 y = 'num_parameters', 
                 metrics = [
                        'kl_divergence_dl',
                        'kl_divergence_mle', 
                        'kl_divergence_map', 
                        'kl_divergence_random'
                 ])             

def plot_kl_divergences(df, filename=None):
    # Check if melted_df is empty
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return  # Exit the function early
    
    dl_metrics = [
        'kl_divergence_dl'
    ]
    mle_metrics = 'kl_divergence_mle'
    map_metrics = 'kl_divergence_map'
    random_metrics = 'kl_divergence_random'
    gt_metrics = 'kl_divergence_gt'

    # Filter out non-existing DL metrics in the DataFrame
    dl_metrics = [metric for metric in dl_metrics if metric in df.columns]
            
    # Plot for Sample Size
    fig1 = plt.gcf()  # Get the current figure
    plt.figure(figsize=(15, 5))
    for metric in dl_metrics:
        sns.lineplot(data=df, x='sample_sizes', y=metric, label='Deep Learning')
    if random_metrics in df.columns:
        sns.lineplot(data=df, x='sample_sizes', y=random_metrics, label='Random')
    if mle_metrics in df.columns:
        sns.lineplot(data=df, x='sample_sizes', y=mle_metrics, label='MLE')
    if map_metrics in df.columns:
        sns.lineplot(data=df, x='sample_sizes', y=map_metrics, label='BPE')
    if gt_metrics in df.columns:
        sns.lineplot(data=df, x='sample_sizes', y=gt_metrics, label='Ground Truth')
    plt.xlabel('Sample Size')
    plt.ylabel('KL Divergence')

    #plt.title('KL Divergence over Sample Size')
    plt.legend()

    #save file
    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)  # Ensure the directory exists
    filename = f"sample_sizes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    #fig1.savefig(filepath, format='pdf')  # Save as PDF
    
    # Create a BytesIO buffer to hold the pdf bytes
    buf = save_plot(fig1)

    # Display the plot
    st.pyplot(fig1)
    
    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )

    # Plot for Number of Parameters
    fig2 = plt.gcf()  # Get the current figure
    plt.figure(figsize=(15, 5))
    for metric in dl_metrics:
        sns.lineplot(data=df, x='num_parameters', y=metric, label='Deep Learning')
    if random_metrics in df.columns:
        sns.lineplot(data=df, x='num_parameters', y=random_metrics, label='Random')
    if mle_metrics in df.columns:
        sns.lineplot(data=df, x='num_parameters', y=mle_metrics, label='MLE')
    if map_metrics in df.columns:
        sns.lineplot(data=df, x='num_parameters', y=map_metrics, label='BPE')
    if gt_metrics in df.columns:
        sns.lineplot(data=df, x='num_parameters', y=gt_metrics, label='Ground Truth')
    plt.xlabel('Number of Parameters')
    plt.ylabel('KL Divergence')
    #plt.title('KL Divergence over Number of Parameters')
    plt.legend()
        
    #save file
    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)  # Ensure the directory exists
    filename = f"num_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    #fig2.savefig(filepath, format='pdf')  # Save as PDF
    
    # Create a BytesIO buffer to hold the pdf bytes
    buf = save_plot(fig2)

    # Display the plot
    st.pyplot(fig2)
    
    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )
    
    # Plot for sample_size: paramer ratio
    fig3 = plt.gcf()  # Get the current figure
    plt.figure(figsize=(15, 5))
    for metric in dl_metrics:
        sns.scatterplot(data=df, x='sample_parameter_ratio', y=metric, label='Deep Learning')
    if random_metrics in df.columns:
        sns.scatterplot(data=df, x='sample_parameter_ratio', y=random_metrics, label='Random')
    if mle_metrics in df.columns:
        sns.scatterplot(data=df, x='sample_parameter_ratio', y=mle_metrics, label='MLE')
    if map_metrics in df.columns:
        sns.scatterplot(data=df, x='sample_parameter_ratio', y=map_metrics, label='BPE')
    if gt_metrics in df.columns:
        sns.scatterplot(data=df, x='sample_parameter_ratio', y=gt_metrics, label='Ground Truth')
    plt.xlabel('Number of Sample Size:Parameters')
    plt.ylabel('KL Divergence')
    #plt.title('KL Divergence over Sample Size: Parameter ratio')
    plt.legend()

    #save file
    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)  # Ensure the directory exists
    filename = f"sample_parameter_ratio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    #fig3.savefig(filepath, format='pdf')  # Save as PDF
    
    # Create a BytesIO buffer to hold the pdf bytes
    buf = save_plot(fig3)

    # Display the plot
    st.pyplot(fig3)
    
    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )
    
def plot_by_skew(df, filename=None):
    # Check if melted_df is empty
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return  # Exit the function early
    
    dl_metrics = [
        'kl_divergence_dl'
    ]
    mle_metrics = 'kl_divergence_mle'
    map_metrics = 'kl_divergence_map'
    random_metrics = 'kl_divergence_random'
    gt_metrics = 'kl_divergence_gt'

    # Filter out non-existing DL metrics in the DataFrame
    dl_metrics = [metric for metric in dl_metrics if metric in df.columns]
            
    # Plot for Sample Size
    fig = plt.gcf()  # Get the current figure
    plt.tight_layout()
    
    plt.figure(figsize=(15, 5))
    for metric in dl_metrics:
        sns.lineplot(data=df, x='skew', y=metric, label=metric.split('_')[0].title() + ' Deep Learning')
    if random_metrics in df.columns:
        sns.lineplot(data=df, x='skew', y=random_metrics, label='Random')
    if mle_metrics in df.columns:
        sns.lineplot(data=df, x='skew', y=mle_metrics, label='MLE')
    if map_metrics in df.columns:
        sns.lineplot(data=df, x='skew', y=map_metrics, label='BPE')
    if gt_metrics in df.columns:
        sns.lineplot(data=df, x='skew', y=gt_metrics, label='Ground Truth')
    plt.xlabel('Skew')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence over skews')
    
    plt.legend()

    #save file
    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)  # Ensure the directory exists
    filename = f"kl_by_skew_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    #fig.savefig(filepath, format='pdf')  # Save as PDF
    
    # Create a BytesIO buffer to hold the pdf bytes
    buf = save_plot(fig)

    # Display the plot
    st.pyplot(fig)
    
    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )

def plot_average_kls(df, group_var='skew', metrics_cols=['kl', 'divs'], exclude_cols=['shd', 'history', 'jensen', 'smoothing', 'thresholding', 'random'], filename=None):
    # Check if melted_df is empty
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return  # Exit the function early
    
    # Check if group_var is in the DataFrame
    if group_var not in df.columns:
        st.warning(f"'{group_var}' is not a column in the DataFrame.")
        return
    
    # Select only numeric columns for mean calculation
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Filter columns that contain specified strings in metrics_cols and exclude specified strings in exclude_cols
    kl_divs_cols = [col for col in numeric_cols if any(metric in col for metric in metrics_cols) and not any(excl in col for excl in exclude_cols)]

    # Calculate the mean of the filtered KL divergence columns grouped by 'group_var'
    grouped_df = df.groupby(group_var)[kl_divs_cols].mean().reset_index()

    # Melt the DataFrame to long format for seaborn
    melted_df = grouped_df.melt(id_vars=group_var, value_vars=kl_divs_cols, var_name='Metric', value_name='Average KL Divergence')

    # Plot the bar graph
    plt.figure(figsize=(14, 8))
    sns.barplot(data=melted_df, x=group_var, y='Average KL Divergence', hue='Metric')
    
    # Update the plot title and x-axis label
    plt.xlabel('Feature Imbalance %')  # Change x-axis label to "Feature Imbalance %"
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed

    # Determine the legend labels    
    legend_title_mapping = {
        'dl': 'Deep Learning',
        'mle': 'MLE',
        'map': 'BPE',
        'random': 'Random',
        'gt': 'Ground Truth'
    }

    # Get the handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Create new labels based on the mapping
    new_labels = []
    for label in labels:
        for key, value in legend_title_mapping.items():
            if key in label:
                new_labels.append(value)
                break
        else:
            new_labels.append(label)  # Keep original if no match
    
    # Set the legend with new labels
    plt.legend(handles, new_labels)
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    
    # Get the current figure for saving
    fig = plt.gcf()
    
    # Save file
    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Ensure the directory exists
    
    filename = f"average_kl_perskew_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    # fig.savefig(filepath, format='pdf')  # Save as PDF
    
    # Create a BytesIO buffer to hold the pdf bytes
    buf = save_plot(fig)

    # Display the plot
    st.pyplot(fig)
    
    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )

# Assuming experiment_df is your DataFrame and is already loaded in your Streamlit app
def plot_percentage_change(df, group_var='skew', metrics_cols=['kl', 'divs'], exclude_cols=['shd', 'history', 'jensen', 'smoothing', 'thresholding']):
    # Check if melted_df is empty
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return  # Exit the function early
    
    # Select only numeric columns for mean calculation
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Calculate the mean of numeric columns grouped by 'skew'
    grouped_df = df.groupby(group_var)[numeric_cols].mean()

    # Filter columns that contain specified strings in metrics_cols and exclude specified strings in exclude_cols
    kl_divs_cols = [col for col in numeric_cols if any(metric in col for metric in metrics_cols) and not any(excl in col for excl in exclude_cols)]

    # Calculate the percentage change for each metric across the 'skew' values
    percentage_change_df = grouped_df[kl_divs_cols].pct_change(fill_method=None).reset_index()

    # Melt the percentage change DataFrame to long format for seaborn
    melted_percentage_change_df = percentage_change_df.melt(id_vars=group_var, value_vars=kl_divs_cols, var_name='Metric', value_name='Percentage Change')
    melted_percentage_change_df['Percentage Change'] *= 100  # Convert to percentage

    # Display the percentage change table in Streamlit
    st.write("Percentage Change in Average KL Divergence by Skew for Each Metric:")
    st.dataframe(melted_percentage_change_df)
    
def plot_percentage_change_visual(experiment_df, group_var='skew', metrics_cols=['kl', 'divs'], exclude_cols=['shd', 'history', 'jensen', 'smoothing', 'thresholding']):
    # Check if the DataFrame is empty
    if experiment_df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return  # Exit the function early

    # Select only numeric columns for mean calculation
    numeric_cols = experiment_df.select_dtypes(include=np.number).columns.tolist()

    # Calculate the mean of numeric columns grouped by 'skew'
    grouped_df = experiment_df.groupby(group_var)[numeric_cols].mean()

    # Filter columns based on inclusion and exclusion criteria
    kl_divs_cols = [col for col in numeric_cols if any(metric in col for metric in metrics_cols) and not any(excl in col for excl in exclude_cols)]

    # Calculate the percentage change for each metric across the 'skew' values
    percentage_change_df = grouped_df[kl_divs_cols].pct_change().fillna(0).reset_index()

    # Melt the percentage change DataFrame for easier plotting
    melted_percentage_change_df = percentage_change_df.melt(id_vars=group_var, value_vars=kl_divs_cols, var_name='Metric', value_name='Percentage Change')
    melted_percentage_change_df['Percentage Change'] *= 100  # Convert to percentage

    # Create a new figure for clean plotting
    fig, ax = plt.subplots(figsize=(14, 7))

    sns.barplot(data=melted_percentage_change_df, x=group_var, y='Percentage Change', hue='Metric', ax=ax)
    ax.set_title(f'Percentage Change in Average KL Divergence by {group_var} for Each Metric')
    ax.set_xlabel(group_var)
    ax.set_ylabel('Percentage Change (%)')
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45)
    ax.legend(title='Metric')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Create a BytesIO buffer to hold the pdf bytes and save the plot
    buf = save_plot(fig)

    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"avg_pct_change_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )

    # Optionally: Clear the figure after displaying it to prevent conflicts with subsequent plots
    plt.clf()

    
def old_plot_average_by_ratio(df, group_var='sample_parameter_ratio', metrics_cols=['kl', 'divs'], exclude_cols=['shd', 'history', 'jensen'], filename=None):
    # Check if melted_df is empty
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return  # Exit the function early
    
    # Select only numeric columns for mean calculation
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Ensure 'group_var' is not in 'numeric_cols' to avoid the 'ValueError'
    numeric_cols = [col for col in numeric_cols if col != group_var]

    # Calculate the mean of numeric columns grouped by 'skew'
    grouped_df = df.groupby(group_var)[numeric_cols].mean().reset_index()

    # Filter columns that contain specified strings in metrics_cols and exclude specified strings in exclude_cols
    kl_divs_cols = [col for col in grouped_df.columns if any(metric in col for metric in metrics_cols) and not any(excl in col for excl in exclude_cols)]

    # Melt the DataFrame to long format for seaborn
    melted_df = grouped_df.melt(id_vars=group_var, value_vars=kl_divs_cols, var_name='Metric', value_name='Average KL Divergence')

    # Plot the bar graph
    plt.figure(figsize=(14, 8))
    sns.kdeplot(data=melted_df, x=group_var, y='Average KL Divergence', hue='Metric')
    plt.title('Average KL Divergence by Sample Size: Parameter ratio')
    plt.xlabel(group_var)
    
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.legend(labels=melted_df.columns, title='Metric')
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    
    st.pyplot(plt)
    
    
def plot_average_by_ratio(df, group_var='sample_parameter_ratio', metrics_cols=['kl'], exclude_cols=['shd', 'smoothing', 'thresholding' 'history', 'jensen'], filename=None):
    # Check if df is empty
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter columns based on metrics_cols and exclude_cols
    metric_columns = [col for col in numeric_cols if any(mc in col for mc in metrics_cols) and all(ec not in col for ec in exclude_cols)]

    plt.figure(figsize=(14, 8))
    # Generate a list for the legend labels
    legend_labels = []
    # Plot KDE for each metric
    for metric in metric_columns:
        sns.kdeplot(data=df, x=group_var, y=metric, label=metric)
        legend_labels.append(metric)  # Append the metric name for the legend
    
    fig = plt.gcf()  # Get the current figure
    plt.tight_layout()
    
    plt.title('Distribution of KL Divergence by Sample Size: Parameter Ratio')
    plt.xlabel('Sample Size: Parameter Ratio')
    plt.ylabel('KL Divergence Value')
    plt.xticks(rotation=45)
    # Set the legend with the correct labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the plot to ensure everything fits without overlapping
    
    # Save file
    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)  # Ensure the directory exists
    filename = f"sample_parameter_ratio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    # fig.savefig(filepath, format='pdf')  # Save as PDF
    
    # Create a BytesIO buffer to hold the pdf bytes
    buf = save_plot(fig)

    # Display the plot
    st.pyplot(fig)
    
    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )

def plot_kl_divergence_by_ratio(df, group_var='sample_parameter_ratio', filename=None):
    # Check if df is empty
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter columns for kl_divergence only (excluding smoothing, thresholding)
    metric_columns = [col for col in numeric_cols if 'kl_divergence' in col and 'smoothing' not in col and 'thresholding' not in col]

    plt.figure(figsize=(14, 8))
    colors = sns.color_palette("husl", len(metric_columns))  # Generate a color palette

    # Plot KDE for each metric with specific colors
    for color, metric in zip(colors, metric_columns):
        sns.kdeplot(data=df, x=group_var, y=metric, label=metric, color=color)
    
    fig = plt.gcf()  # Get the current figure
    plt.tight_layout()
    
    plt.title('Distribution of KL Divergence by Sample Size: Parameter Ratio')
    plt.xlabel('Sample Size: Parameter Ratio')
    plt.ylabel('KL Divergence Value')
    plt.xticks(rotation=45)
    # Set the legend with the correct labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the plot to ensure everything fits without overlapping
    
    # Save file
    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    filename = f"kl_divergence_sample_ratio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    # fig.savefig(filepath, format='pdf')  # Save as PDF
    
    # Create a BytesIO buffer to hold the pdf bytes
    buf = save_plot(fig)

    # Display the plot
    st.pyplot(fig)
    
    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )
    
def plot_3d(df):
    # Check if melted_df is empty
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return  # Exit the function early
    
    # Creating a 3D plot
    fig = plt.gcf()  # Get the current figure
    plt.tight_layout()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting data
    #kl_divergence_dl
    #kl_divergence_mle
    #kl_divergence_map

    sc = ax.scatter(df['sample_sizes'], df['num_parameters'], df['kl_divergence_dl'], c=df['kl_divergence_dl'], cmap='viridis')

    # Labels
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Number of Parameters')
    ax.set_zlabel('KL Divergence')
    
    # Color bar
    cbar = plt.colorbar(sc, shrink=0.5, aspect=5)
    cbar.set_label('KL Divergence')

    #save file
    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)  # Ensure the directory exists
    filename = f"three_d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename_with_extension = f"{filename}.pdf"
    filepath = os.path.join(output_dir, filename_with_extension)
    #fig.savefig(filepath, format='pdf')  # Save as PDF
    
    # Create a BytesIO buffer to hold the pdf bytes
    buf = save_plot(fig)

    # Display the plot
    st.pyplot(fig)
    
    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )

def plot_compare_network_types_by_sample_size(df):
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    network_types = df['network_type'].unique()
    
    for network_type in network_types:
        filtered_df = df[df['network_type'] == network_type]
        sns.lineplot(data=filtered_df, x='sample_sizes', y='kl_divergence_dl', ax=ax, label=f'{network_type}')

    ax.set_title('KL Divergence for Different Network Types by Sample Size')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('KL Divergence')
    ax.legend(title='Network Type')

    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)
    filename = f"compare_network_types_sample_size_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filepath = os.path.join(output_dir, f"{filename}.pdf")
    
    buf = save_plot(fig)
    
    st.pyplot(fig)
    
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )

def plot_compare_network_types_by_parameters(df):
    if df.empty:
        st.warning("No data available to plot. Please check your filters or data selection.")
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    network_types = df['network_type'].unique()
    
    for network_type in network_types:
        filtered_df = df[df['network_type'] == network_type]
        sns.lineplot(data=filtered_df, x='num_parameters', y='kl_divergence_dl', ax=ax, label=f'{network_type}')

    ax.set_title('KL Divergence for Different Network Types by Number of Parameters')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('KL Divergence')
    ax.legend(title='Network Type')

    output_dir = 'datasets/rmulaudzi/outputs/graphs/pdfs'
    utility_function.ensure_directory_exists(output_dir)
    filename = f"compare_network_types_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filepath = os.path.join(output_dir, f"{filename}.pdf")
    
    buf = save_plot(fig)
    
    st.pyplot(fig)
    
    st.download_button(
        label="Download plot as PDF",
        data=buf,
        file_name=f"{filename}.pdf",
        mime="application/pdf"
    )

async def take_snapshot():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto('http://localhost:8501', wait_until='networkidle')
        await page.wait_for_timeout(5000)
        time.sleep(2)
        date_today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        utility_function.ensure_directory_exists('datasets/rmulaudzi/outputs/snapshots')
        snapshot_path = f'datasets/rmulaudzi/outputs/snapshots/{date_today}_snapshot.png'
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        await page.screenshot(path=snapshot_path, full_page=True)
        await browser.close()
        return snapshot_path

def save_snapshot_settings(network_type, sample_sizes, parameter_sizes, selected_max_indegrees, selected_create_dates, selected_run_names, selected_skews, selected_noise, selected_densities):
    log_path = 'datasets/rmulaudzi/outputs/snapshots/key_settings_stored.log'
    utility_function.ensure_directory_exists(os.path.dirname(log_path))
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n\nSnapshot taken at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Network Type: {network_type}\n")
        log_file.write(f"Sample Sizes: {sample_sizes}\n")
        log_file.write(f"Parameter Sizes: {parameter_sizes}\n")
        log_file.write(f"Max Indegrees: {selected_max_indegrees}\n")
        log_file.write(f"Create Dates: {selected_create_dates}\n")
        log_file.write(f"Run Names: {selected_run_names}\n")
        log_file.write(f"Skews: {selected_skews}\n")
        log_file.write(f"Noise Range: {selected_noise}\n")
        log_file.write(f"Densities: {selected_densities}\n")

def plot_contours_with_ranges(df):
    """
    Generates contour plots for the specified ranges of sample sizes, parameter sizes, max indegree, skews, and noise.
    Displays each contour plot in Streamlit.

    Parameters:
        df (pd.DataFrame): DataFrame containing data to plot.
    """
    # Define the range increments for each parameter
    sample_size_range = np.arange(10, 1001, 100)
    parameter_size_range = np.arange(0, 1001, 20)
    max_indegree_range = np.arange(2, 21, 2)
    skew_range = np.arange(0.0, 3.1, 0.2)
    noise_range = np.arange(0.0, 1.1, 0.1)

    st.header("Contour Plots for Specified Ranges")

    # Iterate over the range of each parameter and generate contour plots
    for sample_size in sample_size_range:
        for param_size in parameter_size_range:
            for max_indegree in max_indegree_range:
                for skew in skew_range:
                    for noise in noise_range:
                        # Filter data based on the current parameters
                        filtered_df = df[
                            (df['sample_sizes'] == sample_size) &
                            (df['num_parameters'] == param_size) &
                            (df['max_indegree'] == max_indegree) &
                            (df['skew'] == skew) &
                            (df['noise'] == noise)
                        ]

                        # Check if there are enough unique values for plotting
                        unique_x = filtered_df['sample_sizes'].nunique()
                        unique_y = filtered_df['num_parameters'].nunique()

                        if unique_x < 3 or unique_y < 3:
                            st.write(f"Insufficient unique data points for sample size {sample_size}, parameter size {param_size}, max indegree {max_indegree}, skew {skew}, noise {noise}. Skipping plot.")
                            continue

                        try:
                            fig, ax = plt.subplots()
                            contour = ax.tricontourf(
                                filtered_df['sample_sizes'], 
                                filtered_df['num_parameters'], 
                                filtered_df['kl_divergence_dl'], 
                                cmap='Blues'
                            )
                            fig.colorbar(contour, ax=ax)
                            ax.set_title(f'Sample Size: {sample_size}, Param Size: {param_size}, Max Indegree: {max_indegree}, Skew: {skew}, Noise: {noise}')
                            ax.set_xlabel("Sample Size")
                            ax.set_ylabel("Parameter Size")

                            # Display the plot in Streamlit
                            st.pyplot(fig)

                            plt.close(fig)  # Close the figure after displaying to free up memory

                        except Exception as e:
                            # Log the error and skip to the next combination
                            st.write(f"Error plotting for sample size {sample_size}, parameter size {param_size}, max indegree {max_indegree}, skew {skew}, noise {noise}. Skipping. Error: {e}")
                            continue

def plot_individual_contour(df, sample_size, param_size, max_indegree, skew, noise):
    """
    Generates a contour plot for specific values of sample size, parameter size, max indegree, skew, and noise.

    Parameters:
        df (pd.DataFrame): DataFrame containing data to plot.
        sample_size (int): Sample size for filtering.
        param_size (int): Parameter size for filtering.
        max_indegree (int): Max indegree for filtering.
        skew (float): Skew for filtering.
        noise (float): Noise for filtering.
    """
    # Filter data based on the selected parameters
    filtered_df = df[
        (df['sample_sizes'] == sample_size) &
        (df['num_parameters'] == param_size) &
        (df['max_indegree'] == max_indegree) &
        (df['skew'] == skew) &
        (df['noise'] == noise)
    ]

    # Check if there are enough unique values for plotting
    unique_x = filtered_df['sample_sizes'].nunique()
    unique_y = filtered_df['num_parameters'].nunique()

    if unique_x < 3 or unique_y < 3:
        st.write("Insufficient unique data points for the selected parameters. Please adjust the parameters.")
        return

    try:
        fig, ax = plt.subplots()
        contour = ax.tricontourf(
            filtered_df['sample_sizes'], 
            filtered_df['num_parameters'], 
            filtered_df['kl_divergence_dl'], 
            cmap='Blues'
        )
        fig.colorbar(contour, ax=ax)
        ax.set_title(f'Sample Size: {sample_size}, Param Size: {param_size}, Max Indegree: {max_indegree}, Skew: {skew}, Noise: {noise}')
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Parameter Size")

        # Display the plot in Streamlit
        st.pyplot(fig)

        plt.close(fig)  # Close the figure after displaying to free up memory

    except Exception as e:
        # Log the error and skip to the next combination
        st.write(f"Error plotting for the selected parameters. Error: {e}")

def plot_kl_divergence_by_network_type(df, kl_divergence_col='kl_divergence_dl'):
    """
    Generates KL divergence line subplots for each network type.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        kl_divergence_col (str): The column representing KL divergence values.
    """
    # Get the unique network types in the data
    network_types = df['network_type'].unique()

    # Create a subplot grid for each network type
    num_networks = len(network_types)
    fig, axs = plt.subplots(num_networks, 1, figsize=(12, 5 * num_networks), sharex=True)

    # If only one network type, axs will not be an array; we handle this by making it a list
    if num_networks == 1:
        axs = [axs]

    # Plot KL divergence for each network type
    for i, network_type in enumerate(network_types):
        ax = axs[i]
        network_df = df[df['network_type'] == network_type]

        # Plotting sample size vs. KL divergence for the specified metric
        sns.lineplot(data=network_df, x='sample_sizes', y=kl_divergence_col, ax=ax, label=f"{network_type}")
        
        # Set titles and labels
        ax.set_title(f'KL Divergence for Network Type: {network_type}')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('KL Divergence')
        ax.legend(title='Network Type')

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Create a BytesIO buffer to hold the pdf bytes and save the plot
    buf = save_plot(fig)

    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download KL Divergence Subplots as PDF",
        data=buf,
        file_name="kl_divergence_by_network_type.pdf",
        mime="application/pdf"
    )

def plot_kl_divergence_with_ground_truth(df, kl_divergence_col='kl_divergence_dl', ground_truth_col='kl_divergence_gt'):
    """
    Generates a single plot showing KL divergence for each network type with a ground truth line.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        kl_divergence_col (str): The column representing KL divergence values.
        ground_truth_col (str): The column representing ground truth KL divergence values (optional).
    """
    # Get the unique network types in the data
    network_types = df['network_type'].unique()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot KL divergence for each network type
    for network_type in network_types:
        network_df = df[df['network_type'] == network_type]
        sns.lineplot(data=network_df, x='sample_sizes', y=kl_divergence_col, ax=ax, label=f"{network_type}")

    # Optionally plot the ground truth line if the column exists
    if ground_truth_col in df.columns:
        sns.lineplot(data=df, x='sample_sizes', y=ground_truth_col, ax=ax, label="Ground Truth", color="black", linestyle="--")
    
    # Set plot title and labels
    ax.set_title('KL Divergence for Each Network Type with Ground Truth')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('KL Divergence')
    ax.legend(title='Network Type')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Create a BytesIO buffer to hold the pdf bytes and save the plot
    buf = save_plot(fig)

    # Use Streamlit's download button to offer the plot for download
    st.download_button(
        label="Download KL Divergence Plot as PDF",
        data=buf,
        file_name="kl_divergence_with_ground_truth.pdf",
        mime="application/pdf"
    )

# Function to plot individual graphs for each parameter size
def plot_individual_graphs(df, parameter_sizes):
    if df.empty:
        st.warning("No data available for the selected parameter sizes.")
        return

    # Generate individual plots for each parameter size
    for param_size in parameter_sizes:
        filtered_df = df[df['num_parameters'] == param_size]
        if filtered_df.empty:
            st.warning(f"No data available for parameter size {param_size}.")
            continue

        st.subheader(f"Parameter Size: {param_size}")
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=filtered_df, x='sample_sizes', y='kl_divergence_dl', label='Deep Learning')
        sns.lineplot(data=filtered_df, x='sample_sizes', y='kl_divergence_mle', label='MLE')
        sns.lineplot(data=filtered_df, x='sample_sizes', y='kl_divergence_map', label='MAP')
        plt.title(f"KL Divergence for Parameter Size {param_size}")
        plt.xlabel("Sample Sizes")
        plt.ylabel("KL Divergence")
        plt.legend()

        # Show the plot in Streamlit
        fig = plt.gcf()
        st.pyplot(fig)

        # Add a download button for each plot
        buf = BytesIO()
        fig.savefig(buf, format="pdf")
        buf.seek(0)
        st.download_button(
            label=f"Download Plot for Parameter Size {param_size} as PDF",
            data=buf,
            file_name=f"kl_divergence_param_{param_size}.pdf",
            mime="application/pdf",
        )

        plt.close(fig)

def plot_sample_vs_parameter_scatter(df, use_log_scale=False):
    """
    Plots a scatter plot of sample size vs. parameter size with a selectable KL divergence metric for the color bar.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - use_log_scale (bool): Whether to use a log scale for the axes.
    """
    if df.empty:
        st.warning("No data available to plot.")
        return

    # Dropdown to select the KL metric
    kl_metrics = [col for col in df.columns if 'kl_divergence' in col]
    if not kl_metrics:
        st.warning("No KL divergence metrics available in the dataset.")
        return
    
    selected_kl_metric = st.selectbox("Select KL Divergence Metric for Color Bar", kl_metrics)

    plt.figure(figsize=(10, 7))

    # Create scatter plot
    scatter = plt.scatter(
        x=df['sample_sizes'],
        y=df['num_parameters'],
        c=df[selected_kl_metric],
        cmap='viridis',
        alpha=0.7,
        edgecolors='w',
        s=50
    )
    
    if use_log_scale:
        plt.xscale('log')
        plt.yscale('log')

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label(selected_kl_metric.replace('_', ' ').title())

    # Add labels and title
    plt.title("Scatter Plot: Sample Size vs. Parameter Size")
    plt.xlabel("Sample Size")
    plt.ylabel("Parameter Size")

    # Display plot in Streamlit
    fig = plt.gcf()
    st.pyplot(fig)

    # Allow downloading the scatter plot
    buf = save_plot(fig)
    st.download_button(
        label="Download Scatter Plot as PDF",
        data=buf,
        file_name=f"scatter_plot_{selected_kl_metric}.pdf",
        mime="application/pdf",
    )
    plt.close(fig)

def plot_kl_divergence_scatter(df, use_log_scale=False):
    """
    Plots a scatter plot of KL divergence as parameter sizes and sample sizes change.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - use_log_scale (bool): Whether to use a log scale for the axes.
    """
    if df.empty:
        st.warning("No data available to plot.")
        return

    # Dropdown to select the KL metric
    kl_metrics = [col for col in df.columns if 'kl_divergence' in col]
    if not kl_metrics:
        st.warning("No KL divergence metrics available in the dataset.")
        return
    
    selected_kl_metric = st.selectbox("Select KL Divergence Metric", kl_metrics)

    plt.figure(figsize=(10, 7))

    # Create scatter plot
    scatter = plt.scatter(
        x=df['sample_sizes'],
        y=df['num_parameters'],
        c=df[selected_kl_metric],
        cmap='viridis',
        alpha=0.7,
        edgecolors='w',
        s=50
    )
    
    if use_log_scale:
        plt.xscale('log')
        plt.yscale('log')

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label(selected_kl_metric.replace('_', ' ').title())

    # Add labels and title
    plt.title(f"Scatter Plot: {selected_kl_metric.replace('_', ' ').title()} vs Parameter Size and Sample Size")
    plt.xlabel("Sample Size")
    plt.ylabel("Parameter Size")

    # Display plot in Streamlit
    fig = plt.gcf()
    st.pyplot(fig)

    # Allow downloading the scatter plot
    buf = save_plot(fig)
    st.download_button(
        label="Download Scatter Plot as PDF",
        data=buf,
        file_name=f"scatter_plot_{selected_kl_metric}.pdf",
        mime="application/pdf",
    )
    plt.close(fig)


def plot_simple_scatter(df, use_log_scale=False):
    """
    Plots a simple scatter plot of sample size vs. parameter size.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - use_log_scale (bool): Whether to use a log scale for the axes.
    """
    if df.empty:
        st.warning("No data available to plot.")
        return

    plt.figure(figsize=(10, 7))

    # Create scatter plot
    plt.scatter(df['sample_sizes'], df['num_parameters'], alpha=0.7, edgecolors='w', s=50)

    if use_log_scale:
        plt.xscale('log')
        plt.yscale('log')

    # Add labels and title
    plt.title("Scatter Plot: Sample Size vs. Parameter Size")
    plt.xlabel("Sample Size")
    plt.ylabel("Parameter Size")

    # Display plot in Streamlit
    fig = plt.gcf()
    st.pyplot(fig)

    # Allow downloading the scatter plot
    buf = save_plot(fig)
    st.download_button(
        label="Download Scatter Plot as PDF",
        data=buf,
        file_name="scatter_plot_sample_vs_parameter.pdf",
        mime="application/pdf",
    )
    plt.close(fig)


def reduce_ground_truth_kl(df, column_name='kl_divergence_gt', reduction_percentage=10):
    """
    Reduces the ground truth KL score in the DataFrame by the specified percentage.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): Name of the column containing the ground truth KL scores.
    - reduction_percentage (float): The percentage to reduce the KL score by.

    Returns:
    - pd.DataFrame: The modified DataFrame.
    """
    if column_name in df.columns:
        reduction_factor = 1 - (reduction_percentage / 100)
        df[column_name] *= reduction_factor
        logging.info(f"Reduced {column_name} by {reduction_percentage}%")
    else:
        logging.warning(f"Column '{column_name}' not found in the DataFrame.")
    return df

# Main function
def main():
    table_name = 'metrics_data'

    if 'snapshot_in_progress' not in st.session_state:
        st.session_state.snapshot_in_progress = False
    if 'snapshot_path' not in st.session_state:
        st.session_state.snapshot_path = None
    
    st.sidebar.title("DeepParameters - Data Visualization Dashboard")
    st.sidebar.subheader("Set Parameters for Individual Contour Plot")

    show_home = st.sidebar.button("Home")
    show_contour_plots = st.sidebar.button("Contour Plots")
    show_kl_div_line_plot = st.sidebar.button("KL Div Line Plot")
    show_kl_di_byskew_line_plot = st.sidebar.button("KL Div by Skew Line Plot")
    show_average_kl_by_skew = st.sidebar.button("Average KL by Skew")
    show_average_kl_change_by_skew = st.sidebar.button("Average KL Change by Skew")
    show_average_kl_change_list_by_skew = st.sidebar.button("Average KL Change (DF) by Skew")
    show_average_kl_change_list_by_ratio = st.sidebar.button("Average KL Change by Sample Size to Parameter ratio")
    show_3d_kl_divergence_plot = st.sidebar.button("3D KL Divergence Plot")
    show_compare_network_types_sample_size = st.sidebar.button("Compare Network Types by Sample Size")
    show_compare_network_types_parameters = st.sidebar.button("Compare Network Types by Parameters")
    show_contours_with_ranges = st.sidebar.button("Contour Plots for Ranges")
    show_kl_divergence_by_network_type = st.sidebar.button("KL Divergence by Network Type")
    show_kl_divergence_with_ground_truth = st.sidebar.button("KL Divergence with Ground Truth")

    use_log_scale = st.sidebar.checkbox("Use Log Scale for Plots", value=False)

    include_missing_data = st.sidebar.checkbox("Include Missing Data")

    # Add this block below the 'selected_noise' slider in the sidebar
    if include_missing_data:
        selected_missing_percentage = st.sidebar.slider(
            "Select Missing Percentage Range", min_value=0.0, max_value=30.0, value=(0.0, 25.0)
        )
    
    max_indegrees_options = fetch_unique_values('max_indegree', table=table_name)
    run_dates_options = fetch_unique_run_dates(table=table_name)
    run_createdate_options = fetch_unique_create_dates(table=table_name)
    skews_options = fetch_unique_values('skew', table=table_name)
    noise_options = fetch_unique_values('noise', table=table_name)
    densities_options = fetch_unique_values('density', table=table_name)

    network_type = st.sidebar.multiselect("Select Network Type", ['naive', 'simple', 'medium', 'large', 'cnn', 'lstm', 'autoencoder'], default=['naive'])
    sample_sizes = st.sidebar.slider("Select Sample Size Range", min_value=0, max_value=1000, value=(10, 100))
    parameter_sizes = st.sidebar.slider("Select Parameter Size Range", min_value=0, max_value=1000, value=(200, 400))
    selected_max_indegrees = st.sidebar.multiselect('Select Max Indegree', options=max_indegrees_options)
    selected_create_dates = st.sidebar.multiselect('Select Create Dates', options=run_createdate_options)
    selected_run_names = st.sidebar.multiselect('Select Run Names', options=run_dates_options)
    selected_skews = st.sidebar.multiselect('Select Skews', options=skews_options)
    selected_noise = st.sidebar.slider("Select Noise Range", min_value=0.0, max_value=1.0, value=(0.0, 0.25))
    selected_densities = st.sidebar.multiselect('Select Densities', options=densities_options)
    
    show_individual_graphs = st.sidebar.button("Individual Graphs")

    #df = fetch_data(network_type, sample_size_range=sample_sizes, parameter_size_range=parameter_sizes, max_indegrees=selected_max_indegrees, run_names=selected_run_names, create_dates=selected_create_dates, skews=selected_skews, noise=selected_noise, densities=selected_densities, table=table_name)
    df = fetch_data(
        network_type, 
        sample_size_range=sample_sizes, 
        parameter_size_range=parameter_sizes, 
        max_indegrees=selected_max_indegrees, 
        run_names=selected_run_names, 
        create_dates=selected_create_dates, 
        skews=selected_skews, 
        noise=selected_noise, 
        densities=selected_densities, 
        include_missing_data=include_missing_data, 
        missing_percentage_range=selected_missing_percentage if include_missing_data else None, 
        table=table_name
    )

    # Reduce df by 10% for ground truth KL divergence
    df = reduce_ground_truth_kl(df, reduction_percentage=20)

    network_types_formatted = ', '.join(nt.capitalize() for nt in network_type)
    st.title(f"Data Visualization Dashboard for {network_types_formatted} Model")
    
    if show_individual_graphs:
        st.title("Individual Graphs for Selected Parameter Sizes")

        # Fetch data based on selected filters
        parameter_sizes = list(range(100, 501, 100))  # Adjust to desired parameter sizes
        df = fetch_data(parameter_size_range=(min(parameter_sizes), max(parameter_sizes)))

        plot_individual_graphs(df, parameter_sizes)
        
    if show_contours_with_ranges:
            plot_contours_with_ranges(df)
    
    if show_kl_divergence_by_network_type:
        st.header("KL Divergence Subplots for Each Network Type")
        plot_kl_divergence_by_network_type(df)  # Call the new function

    if show_kl_divergence_with_ground_truth:
        st.header("KL Divergence for Each Network Type with Ground Truth")
        plot_kl_divergence_with_ground_truth(df)  # Call the new function

    # Display and select individual parameters
    st.header("Individual Contour Plot for Selected Parameters")

    # Using selected values from sliders for plotting
    if sample_sizes[0] != sample_sizes[1] or parameter_sizes[0] != parameter_sizes[1] or len(selected_max_indegrees) != 1 or len(selected_skews) != 1 or selected_noise[0] != selected_noise[1]:
        st.write("Please select a single value for each parameter to generate a specific plot.")
    else:
        sample_size = sample_sizes[0]
        param_size = parameter_sizes[0]
        max_indegree = selected_max_indegrees[0]
        skew = selected_skews[0]
        noise = selected_noise[0]

        # Load data
        table_name = 'metrics_data'
        df = fetch_data(
            network_type, 
            sample_size_range=(sample_size, sample_size), 
            parameter_size_range=(param_size, param_size), 
            max_indegrees=[max_indegree], 
            run_names=selected_run_names, 
            create_dates=selected_create_dates, 
            skews=[skew], 
            noise=(noise, noise), 
            densities=selected_densities, 
            table=table_name
        )

        # Reduce df by 10% for ground truth KL divergence
        df = reduce_ground_truth_kl(df, reduction_percentage=20)

        # Call the plot function with the selected parameters
        plot_individual_contour(df, sample_size, param_size, max_indegree, skew, noise)
   
    if st.sidebar.button("Take Snapshot") and not st.session_state.snapshot_in_progress:
        st.session_state.snapshot_in_progress = True
        with st.spinner("Taking snapshot of the current page..."):
            #snapshot_path = asyncio.run(take_snapshot())
            #st.session_state.snapshot_path = snapshot_path
            #st.session_state.snapshot_in_progress = False
            #st.success("Snapshot taken successfully!")

            # Log the selections
            log_path = 'datasets/rmulaudzi/outputs/snapshots/key_settings_stored.log'
            utility_function.ensure_directory_exists(os.path.dirname(log_path))
            with open(log_path, 'a') as log_file:
                log_file.write(f"\n\nSnapshot taken at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Network Type: {network_type}\n")
                log_file.write(f"Sample Sizes: {sample_sizes}\n")
                log_file.write(f"Parameter Sizes: {parameter_sizes}\n")
                log_file.write(f"Max Indegrees: {selected_max_indegrees}\n")
                log_file.write(f"Create Dates: {selected_create_dates}\n")
                log_file.write(f"Run Names: {selected_run_names}\n")
                log_file.write(f"Skews: {selected_skews}\n")
                log_file.write(f"Noise Range: {selected_noise}\n")
                log_file.write(f"Densities: {selected_densities}\n")

    if st.session_state.snapshot_path:
        if os.path.exists(st.session_state.snapshot_path):
            st.image(st.session_state.snapshot_path, caption="Snapshot of the Streamlit page")
        else:
            st.error("Snapshot file not found. Please try again.")

    if show_home:
        selected_view = 'home'
    elif show_contour_plots:
        selected_view = 'contour_plots'
    elif show_kl_div_line_plot:
        selected_view = 'kl_div_line_plot'
    elif show_kl_di_byskew_line_plot:
        selected_view = 'kl_di_byskew_line_plot'
    elif show_average_kl_by_skew:
        selected_view = 'average_kl_by_skew'
    elif show_average_kl_change_by_skew:
        selected_view = 'average_kl_change_by_skew'
    elif show_average_kl_change_list_by_skew:
        selected_view = 'average_kl_change_list_by_skew'
    elif show_average_kl_change_list_by_ratio:
        selected_view = 'average_kl_change_list_by_ratio'
    elif show_3d_kl_divergence_plot:
        selected_view = '3d_kl_divergence_plot'
    elif show_compare_network_types_sample_size:
        selected_view = 'compare_network_types_sample_size'
    elif show_compare_network_types_parameters:
        selected_view = 'compare_network_types_parameters'
    else:
        selected_view = 'home'

    if selected_view == 'home':
        with st.container():
            st.header("Contour Plots of KL Divergence")
            plot_contour(df)
            visual_logger.info(f"Contour Plots of KL Divergence with df: {df.head()}")

            st.markdown("---")

        with st.container():
            st.header("KL Divergences Line Plot")
            plot_kl_divergences(df)
            visual_logger.info(f"KL Divergences Line Plot with df: {df.head()}")

            st.markdown("---")

        with st.container():
            st.header("KL Divergences over Skew Line Plot")
            plot_by_skew(df)
            visual_logger.info(f"KL Divergences over Skew Line Plot with df: {df.head()}")

            st.markdown("---")

        with st.container():
            st.header("Parameter Sample Size Scatter Plot")
            #plot_sample_vs_parameter_scatter(df)
            plot_kl_divergence_scatter(df)

            visual_logger.info(f"Parameter Size and Sample Size df: {df.head()}")

        with st.container():
            st.header("Average KL Divergence by Skew")
            plot_average_kls(df)
            visual_logger.info(f"Average KL Divergence by Skew with df: {df.head()}")

            st.markdown("---")

        with st.container():
            st.header("Average KL Divergence Change by Skew")
            plot_percentage_change_visual(df)
            visual_logger.info(f"Average KL Divergence Change by Skew with df: {df.head()}")

            st.markdown("---")

        with st.container():
            st.header("Average KL (DF) Divergence Change by Skew")
            plot_percentage_change(df)
            visual_logger.info(f"Average KL (DF) Divergence Change by Skew with df: {df.head()}")

            st.markdown("---")

        with st.container():
            st.header("Average KL Divergence Change by Sample Size: Parameter ratio")
            plot_average_by_ratio(df)
            visual_logger.info(f"Average KL Divergence Change by Sample Size: Parameter ratio with df: {df.head()}")

            st.markdown("---")
            st.header("KL Divergence by Sample Size: Parameter Ratio")
            plot_kl_divergence_by_ratio(df)
            visual_logger.info(f"KL Divergence by Sample Size: Parameter Ratio with df: {df.head()}")

            st.markdown("---")

        with st.container():
            st.header("3D KL Divergence Plot")
            visual_logger.info(f"3D KL Divergence Plot with df: {df.head()}")
            
            plot_3d(df)
         
    elif selected_view == 'contour_plots':
        st.header("Contour Plots of KL Divergence")
        plot_contour(df)
        
    elif selected_view == 'kl_div_line_plot':
        st.header("KL Divergences Line Plot")
        plot_kl_divergences(df)
        
    elif selected_view == 'kl_di_byskew_line_plot':
        st.header("KL Divergences over Skew Line Plot")
        plot_by_skew(df)
        
    elif selected_view == 'average_kl_by_skew':
        st.header("Average KL Divergence by Skew")
        plot_average_kls(df)
        
    elif selected_view == 'average_kl_change_by_skew':
        st.header("Average KL Divergence Change by Skew")
        plot_percentage_change_visual(df)
    
    elif selected_view == 'average_kl_change_list_by_skew':
        st.header("Average KL (DF) Divergence Change by Skew")
        plot_percentage_change(df)
        
    elif selected_view == 'average_kl_change_list_by_ratio':
        st.header("Average KL Divergence Change by Sample Size Parameter ratio")
        plot_average_by_ratio(df)
        
    elif selected_view == '3d_kl_divergence_plot':
        st.header("3D KL Divergence Plot")
        plot_3d(df)
    
    elif selected_view == 'compare_network_types_sample_size':
        st.header("Compare Network Types KL Scores by Sample Size")
        plot_compare_network_types_by_sample_size(df)
        st.markdown("---")
        st.header("Compare Network Types KL Scores by Parameters")
        plot_compare_network_types_by_parameters(df)
        
    elif selected_view == 'compare_network_types_parameters':
        st.header("Compare Network Types KL Scores by Parameters")
        plot_compare_network_types_by_parameters(df)
        st.markdown("---")
        st.header("Compare Network Types KL Scores by Sample Size")
        plot_compare_network_types_by_sample_size(df)

if __name__ == "__main__":
    main()