import boto3
from Constants.constants import Credentials

dynamodb = boto3.resource("dynamodb",
                          region_name="ap-south-1",
                          aws_access_key_id=Credentials.aws_access_key_id,
                          aws_secret_access_key=Credentials.aws_secret_access_key
                          )

# DynamoDB Table Name and S3 Bucket Name
DYNAMODB_TABLE_NAME = 'model-details'
S3_BUCKET_NAME = 'b3ll-curve-model-storage'


def fetch_all_model_details():
    """Retrieve all model details from DynamoDB."""
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    try:
        response = table.scan()  # This scans the entire table and retrieves all items
        items = response.get('Items', [])

        # Handle pagination if the dataset is large
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))

        return items
    except Exception as e:
        print(f"Failed to fetch model details from DynamoDB: {e}")
        return None
