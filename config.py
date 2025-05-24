import os
from dotenv import load_dotenv
import boto3
from sqlalchemy import create_engine

# load config
load_dotenv(override = True)

# AWS Settings
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION_NAME = os.getenv("AWS_REGION")

# RDS Settings
DB_HOST = os.getenv("RDB_HOST")
DB_USER = os.getenv("RDB_USER")
DB_PASSWORD = os.getenv("RDB_PASSWORD")
DB_NAME = os.getenv("RDB_NAME")
DB_PORT = int(os.getenv("RDB_PORT"))

# EC2
EC2_API_URL = os.getenv("EC2_PUBLIC_ADDR")

# S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION_NAME,
)

# SQLAlchemy engine
engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)