from Data_Preprocessing.data_processing import standardize_dataset
from Constants.constants import Credentials, Commodities

import pandas as pd
import io
import boto3


def store_forecast(bucket_name, folder_name, datasets):
    """
    Stores multiple datasets in S3 as JSON files.

    Parameters:
    - bucket_name: Name of the S3 bucket.
    - folder_name: Folder path within the S3 bucket.
    - datasets: A dictionary where keys are filenames and values are pandas DataFrames.
    """
    try:
        s3_client = boto3.client('s3', aws_access_key_id=Credentials.aws_access_key_id,
                                 aws_secret_access_key=Credentials.aws_secret_access_key)

        for filename, df in datasets.items():
            json_data = df.to_json(orient='records', date_format='iso')
            file_path = f"{folder_name}/{filename}.json"
            s3_client.put_object(Bucket=bucket_name, Key=file_path, Body=json_data)
            print(f"Uploaded {filename} to s3://{bucket_name}/{file_path}")
    except Exception as e:
        raise Exception(f"Failed to store forecast: {e}")


def read_forecast(commodity_name):
    """
    Fetches and reads the forecast and actual values from S3 for a given commodity.

    Parameters:
    - commodity_name: The name of the commodity to retrieve the forecasts for.

    Returns:
    - A dictionary containing pandas DataFrames for the actual and forecast data.
    """
    folder_name = f'{commodity_name}/'  # Use the commodity name directly as the folder name

    try:
        s3_client = boto3.client('s3', aws_access_key_id=Credentials.aws_access_key_id,
                                 aws_secret_access_key=Credentials.aws_secret_access_key)
        response = s3_client.list_objects_v2(Bucket=Commodities.FORECAST_STORAGE, Prefix=folder_name)

        if 'Contents' not in response:
            raise Exception("No files found in the specified folder.")

        data_files = {}
        for item in response['Contents']:
            file_key = item['Key']
            if file_key.endswith('.json'):
                # Download the JSON data
                json_obj = s3_client.get_object(Bucket=Commodities.FORECAST_STORAGE, Key=file_key)
                data = pd.read_json(io.BytesIO(json_obj['Body'].read()), orient='records')

                # Identify the type of file (actual or forecast)
                file_type = 'actual' if 'actual' in file_key else 'forecast'
                data_files[file_type] = data

        if len(data_files) != 2:
            raise Exception("Expected both actual and forecast files.")

        return data_files

    except Exception as e:
        raise Exception(f"Failed to read forecast data from S3: {e}")


def read_data_s3(bucket_name, folder_name):
    """Function to read and process data from an S3 bucket folder.

    Parameters:
    - bucket_name: Name of the S3 bucket.
    - folder_name: Folder path within the S3 bucket.
    """

    try:
        def custom_date_parser(date_string):
            return pd.to_datetime(date_string, format='%d/%m/%y')

        # Initialize S3 client
        s3_client = boto3.client('s3', aws_access_key_id=Credentials.aws_access_key_id,
                                 aws_secret_access_key=Credentials.aws_secret_access_key)

        # List files in the specified folder
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
        files = [item['Key'] for item in response['Contents'] if item['Key'].endswith('.csv')]

        all_data = []
        for file_key in files:
            csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            body = csv_obj['Body']
            data = pd.read_csv(io.BytesIO(body.read()), parse_dates=['Date'], date_parser=custom_date_parser)
            all_data.append(data)

        # print(all_data)
        standardized_datasets = []

        for df in all_data:
            standardized_df = standardize_dataset(df, 'Date', df.columns.drop('Date'))
            standardized_datasets.append(standardized_df)

        all_dates = pd.date_range(start=min(df['Date'].min() for df in standardized_datasets),
                                  end=max(df['Date'].max() for df in standardized_datasets))

        date_df = pd.DataFrame(all_dates, columns=['Date'])
        for df in standardized_datasets:
            date_df = pd.merge(date_df, df, on='Date', how='left')

        date_column = date_df['Date']
        date_df['Date'] = pd.to_datetime(date_df['Date'])
        date_df = date_df[date_df['Date'].dt.dayofweek < 5]
        date_df.drop('Date', axis=1, inplace=True)
        date_df.interpolate(method='linear', inplace=True)
        date_df['Date'] = date_column
        date_df = date_df[['Date'] + [col for col in date_df.columns if col != 'Date']]

        return date_df
    except Exception as e:
        raise Exception(f"Failed to read data from S3: {e}")
