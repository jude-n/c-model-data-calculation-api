import json

from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta
import gspread
import os
from oauth2client.service_account import ServiceAccountCredentials
from rest_framework.decorators import api_view
# Custom for using env
import environ
# Initialise environment variables
env = environ.Env()
environ.Env.read_env()

def local_google_connection():
    # Get directory path
    DIRNAME = os.path.dirname(__file__)
    # Scope of connection to google sheets
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    # Connect to google sheets using credentials from json
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join(DIRNAME, "creds.json"), scope)
    # Authorize connection
    client = gspread.authorize(creds)
    return client

def heroku_googl_connection():
    # Get directory path
    # Scope of connection to google sheets
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    credentials_json = env('GOOGLE_APPLICATION_CREDENTIALS')

    # Connect to google sheets using credentials from json
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_json, scope)
    # Authorize connection
    client = gspread.authorize(creds)
    return client

def connect_to_gsheet(gsheet):
    client = heroku_googl_connection()
    if gsheet == "Russ":
        # Google sheet name and the specific sheet name selected
        sheet = client.open("Russ Time Series").worksheet("rawData")  # Open the spreadsheet
        # Get data in specified sheet
        data = sheet.get_all_records()  # Get a list of all records
        client_raw_data = pd.DataFrame.from_records(data)
        return client_raw_data
    else:
        # Google sheet name and the specific sheet name selected
        sheet = client.open("New Cmodel Product").worksheet("data")  # Open the spreadsheet
        # Get data in specified sheet
        data = sheet.get_all_values()  # Get a list of all records
        client_raw_data = pd.DataFrame.from_records(data)
        client_raw_data_header = client_raw_data.iloc[0] #grab the first row for the header
        client_raw_data = client_raw_data[1:] #take the data less the header row
        client_raw_data.columns = client_raw_data_header #set the header row as the df header

# def connect_to_new_cmodel_time_sheet():

# print(client_raw_data.head())
# Mapped data from DB
#

def standard_data_clean(client_raw_data):
    # In probability column remove percentage signs
    client_raw_data['Probability'] = client_raw_data['Probability'].apply(lambda x: x.replace("%", ""))
    # Change probability values to decimals and if no data replace with 0
    client_raw_data['Probability'] = client_raw_data['Probability'].apply(lambda x: int(x) / 100 if x.isnumeric() else 0)
    # Format date columns
    client_raw_data['Created Date'] = client_raw_data['Created Date'].apply(lambda x: pd.to_datetime(x))
    client_raw_data['Close Date'] = client_raw_data['Close Date'].apply(lambda x: pd.to_datetime(x))
    client_raw_data['Last Activity Date'] = client_raw_data['Last Activity Date'].apply(lambda x: pd.to_datetime(x))
    # Format age column to be a number else replace with 0
    client_raw_data['Age'] = client_raw_data['Age'].apply(lambda x: int(x) if type(x) == int else 0)
    # Remove dollar signs and commas
    client_raw_data['ARR'] = client_raw_data['ARR'].apply(lambda x: x.replace("$", "").replace(",", ""))
    # Check ARR is a number else replace with 0
    client_raw_data['ARR'] = client_raw_data['ARR'].apply(lambda x: x if x.isnumeric() else 0)
    # Make sure ARR is a float type
    client_raw_data['ARR'] = client_raw_data['ARR'].astype(float)
    return client_raw_data


def filter_data_for_number_of_years(cleaned_data):
    # Get current year
    current_year = dt.date.today().year
    length_of_time_from_current_year = 2  # Should have a master file to pull these from
    # Copy data
    cleaned_client_data_for_two_years = cleaned_data.copy()
    # Filter data by specified number of numbers
    cleaned_client_data_for_two_years = cleaned_client_data_for_two_years[
        (cleaned_client_data_for_two_years["Close Date"].dt.year >= (current_year - length_of_time_from_current_year)) & \
        (cleaned_client_data_for_two_years["Close Date"].dt.year < current_year)]
    return cleaned_client_data_for_two_years

def fetch_data_for_product(cleaned_client_data_for_two_years, selected_product):
    filter_by = 'Closed Won'
    # Get data for selected stage
    filtered_by_stage_data = cleaned_client_data_for_two_years[cleaned_client_data_for_two_years['Stage'] == filter_by]
    # Get columns that will be used and data for specified product
    filtered_by_selected_product_data = filtered_by_stage_data[['Close Date', 'ARR', 'Group Name']][filtered_by_stage_data['Group Name'] == selected_product]
    # Modify date to get just month and year and store in new column
    filtered_by_selected_product_data['Close Month'] = filtered_by_selected_product_data['Close Date'].dt.strftime('%m-%Y')
    filtered_by_selected_product_data = filtered_by_selected_product_data.groupby(['Close Month'])['ARR'].sum().reset_index()
    return filtered_by_selected_product_data

def copy_data(data_to_copy):
    return data_to_copy.copy()

def forecast_months(summed_data):
    # Get last month in data
    last_month = (summed_data['Close Month'].iat[-1])
    last_month = dt.datetime.strptime(last_month, "%m-%Y")

    # Get months for the next year
    forecasted_months_list = list()
    for i in range(1, 13):
        date = last_month + relativedelta(months=+i)
        forecasted_months_list.append([date])

    # turn forecasted months into data frame
    forecasted_months = pd.DataFrame(forecasted_months_list)
    # Gave column a name
    forecasted_months.columns = ['Forecast Month']
    # pull just months and store in new column
    forecasted_months['month'] = forecasted_months['Forecast Month'].dt.month
    return forecasted_months

def get_yearly_totals_for_data(summed_data):
    # copy data
    summed_data_copy = copy_data(summed_data)
    # Pull year into a new column
    summed_data_copy['year'] = pd.to_datetime(summed_data_copy['Close Month']).dt.year
    # Sum total price and group based on year
    summed_data_yearly_total = summed_data_copy.groupby(['year'])['ARR'].sum().reset_index(name='sum ARR')
    return summed_data_yearly_total

def get_median_for_each_month(summed_data, summed_data_yearly_total):
    # copy data
    monthly_median_for_data = copy_data(summed_data)
    # Pull year into a new column
    monthly_median_for_data['year'] = pd.to_datetime(monthly_median_for_data['Close Month']).dt.year
    # Merge current data with summed yearly data based on year
    monthly_median_for_data = pd.merge(monthly_median_for_data, summed_data_yearly_total, how="inner",
                                           on=["year"])
    # Divide product total month price by product total year price
    monthly_median_for_data['year month average'] = monthly_median_for_data['ARR'] / \
                                                        monthly_median_for_data['sum ARR']
    # Pull month into a new column
    monthly_median_for_data['month'] = pd.to_datetime(monthly_median_for_data['Close Month']).dt.month
    # Group by month and find median
    monthly_median_for_data = monthly_median_for_data.groupby('month')[
        'year month average'].median().reset_index(name='median ARR')
    return monthly_median_for_data

def get_mean_for_each_month(summed_data):
    monthly_mean_for_data = copy_data(summed_data)
    monthly_mean_for_data['month'] = pd.to_datetime(monthly_mean_for_data['Close Month']).dt.month
    monthly_mean_for_data = monthly_mean_for_data.groupby(['month'])['ARR'].mean().reset_index(name='mean ARR')
    return monthly_mean_for_data

def add_desired_growth_to_forecast(monthly_median_for_data, desired_growth, monthly_mean_for_data):
    desired_growth = int(desired_growth)/100
    initial_forecast = copy_data(monthly_median_for_data)
    initial_forecast['median ARR x desired_growth'] = initial_forecast['median ARR'] * desired_growth
    initial_forecast = pd.merge(initial_forecast, monthly_mean_for_data, how="inner", on=["month"])
    initial_forecast['c-model'] = initial_forecast['median ARR x desired_growth'] + initial_forecast['mean ARR']
    return initial_forecast

def format_data_into_graph_structure(data_forecast,selected_product):
    product = selected_product.replace(' ', '_')
    product = product.lower()
    forecast_year = data_forecast['year']
    forecasted_months = data_forecast['month']
    forecast_values = data_forecast['c-model']
    initial_graph_data = []
    graph_data = {}
    for i in range(len(forecasted_months)):
        product_object = {'year': forecast_year[i], 'month': forecasted_months[i], product: forecast_values[i]}
        initial_graph_data.append(product_object)
    graph_data['data'] = initial_graph_data
    return graph_data

def run_cmodel(filtered_data, desired_growth):
    yearly_totals = get_yearly_totals_for_data(filtered_data)
    calculated_median = get_median_for_each_month(filtered_data)
    calculated_mean = get_mean_for_each_month(filtered_data)
    calculated_forecast = add_desired_growth_to_forecast(calculated_median, desired_growth, calculated_mean)
    return calculated_forecast

def merge_data_with_forecast_months(future_months, forecast_data):
    merged_forecast = pd.merge(future_months, forecast_data[['month', 'c-model']], on='month', how='left')
    # merged_forecast = merged_forecast.loc[:, merged_forecast.columns != 'month']
    merged_forecast['year'] = pd.to_datetime(merged_forecast['Forecast Month']).dt.year
    return merged_forecast

def creating_columns_for_graph(graph_data):
    columns = []
    graph_columns = {}
    for key, value in graph_data['data'][0].items():
        column_object = {"name": key, 'label': key, "options": {"filter": True, "sort": True}}
        columns.append(column_object)
    graph_data['columns'] = columns
    return graph_data
    # merged_forecast.head()
# Required columns for targets model calculation
required_columns_for_model = ["Group Name", 'Opportunity Owner', 'Opportunity Name', 'Created Date', 'Close Date',
                              'Close Quarter', 'Age', 'Stage', 'Probability', 'Forecast Stage', 'ARR',
                              'Last Activity Date']

# # Convert data into a data frame
# clientraw_data = pd.DataFrame.from_records(data)
# print(clientraw_data.head())
# Fetch required columns and their data
# clientraw_data = clientraw_data[required_columns_for_model]

# # Make a copy of the data
# client_raw_data_copy = copy_data(client_raw_data)
# # Clean data
# client_data_clean = standard_data_clean(client_raw_data_copy)
# print(client_data_clean.head())
# selected_product = 'Product 1'
# clean_filtered_data = filter_data_for_number_of_years(client_data_clean)
# selected_product_data = fetch_data_for_product(clean_filtered_data, selected_product)
# forecasted_months = forecast_months(selected_product_data)
# yearly_totals = get_yearly_totals_for_data(selected_product_data)
# median_data = get_median_for_each_month(selected_product_data, yearly_totals)
# mean_data = get_mean_for_each_month(selected_product_data)
# initial_forecast = add_desired_growth_to_forecast(median_data, 0, mean_data)
# print(initial_forecast.head())

# Summary Table Calculation
#

@api_view(['GET'])
def run_targets(request):
    # return Response({'key': request}, status=status.HTTP_200_OK)
    # return Response(request.GET["product"])
    # Convert data into a data frame
    data = connect_to_gsheet('Russ')
    product = request.GET["product"].replace('_',' ')
    product = product.capitalize()
    client_raw_data = pd.DataFrame.from_records(data)
    client_raw_data_copy = copy_data(client_raw_data)
    client_data_clean = standard_data_clean(client_raw_data_copy)
    clean_filtered_data = filter_data_for_number_of_years(client_data_clean)
    selected_product_data = fetch_data_for_product(clean_filtered_data, product)
    forecasted_months = forecast_months(selected_product_data)
    yearly_totals = get_yearly_totals_for_data(selected_product_data)
    median_data = get_median_for_each_month(selected_product_data, yearly_totals)
    mean_data = get_mean_for_each_month(selected_product_data)
    initial_forecast = add_desired_growth_to_forecast(median_data, request.GET["growth"], mean_data)
    merged_forecasted_data = merge_data_with_forecast_months(forecasted_months, initial_forecast)
    graph_data = format_data_into_graph_structure(merged_forecasted_data, product)
    mapped_graph_data = creating_columns_for_graph(graph_data)
    # return JsonResponse(json.dump(graph_data), safe=False)
    return Response(mapped_graph_data)
# for v2 file will be s3 path for now:
# read file
# get necessary variables
# pass it to function
# return results from function

