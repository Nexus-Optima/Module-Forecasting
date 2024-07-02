from Constants.constants import Credentials, Commodities

import pandas as pd
import io
import boto3


def read_forecast(commodity_name, forecast_type):
    """
    Fetches and reads the forecast and actual values from S3 for a given commodity.

    Parameters:
    - commodity_name: The name of the commodity to retrieve the forecasts for.

    Returns:
    - A dictionary containing pandas DataFrames for the actual and forecast data.
    """
    folder_name = f'{commodity_name}/{forecast_type}/'  # Use the commodity name directly as the folder name

    try:
        s3_client = boto3.client('s3', aws_access_key_id=Credentials.aws_access_key_id,
                                 aws_secret_access_key=Credentials.aws_secret_access_key)
        response = s3_client.list_objects_v2(Bucket=Commodities.FORECAST_STORAGE, Prefix=folder_name)

        if 'Contents' not in response:
            raise Exception("No files found in the specified folder.")

        # Dictionary to store data
        data_files = {
            '7D': {'actual': None, 'predictions': None, 'forecast': None},
            '15D': {'actual': None, 'predictions': None, 'forecast': None},
            '30D': {'actual': None, 'predictions': None, 'forecast': None}
        }

        periods = ['7D', '15D', '30D']

        file_types = ['actual_values.json', 'prediction_values.json', 'forecast_values.json']

        for period in periods:
            for file_type in file_types:
                file_key = f'{commodity_name}/{forecast_type}/{period}/{file_type}'
                try:
                    # Download the JSON data
                    json_obj = s3_client.get_object(Bucket=Commodities.FORECAST_STORAGE, Key=file_key)
                    data = pd.read_json(io.BytesIO(json_obj['Body'].read()), orient='records')

                    if 'actual_values.json' in file_type:
                        data_files[period]['actual'] = data
                    elif 'prediction_values.json' in file_type:
                        data_files[period]['predictions'] = data
                    elif 'forecast_values.json' in file_type:
                        data_files[period]['forecast'] = data
                except s3_client.exceptions.NoSuchKey:
                    print(f"File {file_key} does not exist.")
                    continue


        if len(data_files) < 2:
            raise Exception("Expected both actual and forecast files.")

        return data_files

    except Exception as e:
        raise Exception(f"Failed to read forecast data from S3: {e}")
