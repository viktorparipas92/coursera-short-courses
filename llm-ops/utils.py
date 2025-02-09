import base64
import json
import os

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials


def get_service_account_key() -> str:
    as_string_b64 = os.getenv('SERVICE_ACCOUNT_KEY')

    as_bytes_b64 = as_string_b64.encode('ascii')
    as_string_bytes = base64.b64decode(as_bytes_b64)
    as_string = as_string_bytes.decode('ascii')
    return as_string


def authenticate():
    # return 'DLAI_CREDENTIALS', 'DLAI_PROJECT_ID'
    load_dotenv()

    service_account_key = get_service_account_key()

    # Create credentials based on key from service account
    # Make sure your account has the roles listed in the Google Cloud Setup section
    credentials = Credentials.from_service_account_info(
        service_account_key,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    if credentials.expired:
        credentials.refresh(Request())

    PROJECT_ID = os.getenv('PROJECT_ID')
    return credentials, PROJECT_ID
