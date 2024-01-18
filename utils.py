import json


def get_json(file_path):    # Read json file, return as dict
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data