import uuid
from OCR import *
from dotenv import load_dotenv
load_dotenv()

def get_database():
    from pymongo import MongoClient
    import pymongo

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = os.getenv('DB_URI')

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    from pymongo import MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client['NPR']

dbname = get_database()
number_plates = dbname["NumberPlates"]

'{}.jpg'.format(uuid.uuid1())
def save_results(text, region, folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())
    
    cv2.imwrite(os.path.join(folder_path, img_name), region)
    
    data = {"img_name": img_name, "text": text}
    number_plates.insert_one(data)