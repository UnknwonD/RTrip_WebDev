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
from modules.user import update_user_info
from modules.s3_utils import get_user_recommended_images_and_areas, get_user_info
from modules.ec2_utils import send_to_ec2

from lee import find_nearest_neighbors
from gnn_module import *

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")  # macOS에서 안전