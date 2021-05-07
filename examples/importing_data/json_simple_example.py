import json
import os

file_path = os.path.dirname(os.path.abspath(__file__)) + "/"
# Load JSON: json_data
with open(file_path + "a_movie.json") as json_file:
    json_data = json.load(json_file)

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ": ", json_data[k])
