import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = os.getenv('DEBUG', False)
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data')

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
