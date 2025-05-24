from flask import Flask,render_template, redirect, url_for, flash, jsonify, request, session
from dotenv import load_dotenv
from datetime import datetime


import boto3
import requests
import os
import requests
import os
import uuid
import json
import pickle
import botocore
from modules.auth import authenticate, find_user_by_credentials, handle_login_success
from modules.ec2_utils import send_to_ec2
from modules.user import is_duplicate, register_user, update_user_info, get_user_info, extract_user_data, calculate_age_group, save_user_to_s3
from modules.s3_utils import get_json_from_s3, list_s3_objects, put_json_to_s3, get_s3_signed_urls
from modules.rds_utils import get_images_by_travel_ids, find_nearest_users, get_user_recommended_images_and_areas, get_meta_photo_info, default_travel_plans, travel_plans
from modules.form_utils import extract_travel_styles, get_presigned_image_urls
from gnn_module import *

# import torch.multiprocessing as mp
# mp.set_sharing_strategy("file_system")  # macOS에서 안전

