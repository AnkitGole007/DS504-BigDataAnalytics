import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

# Create a directory for the small test data
test_dir = "./test_data"
os.makedirs(test_dir, exist_ok=True)

# Number of CSV files to generate and rows per file
num_files = 3
num_rows = 100

# Define the columns based on the training dataset format
columns = ["plate", "longitude", "latitude", "time", "status"]

# Use a base date similar to the training data (e.g., 2016-08-07)
base_date = datetime(2016, 8, 7)

def random_time():
    """Generate a random time on the base_date in 24-hour format."""
    # Random number of seconds from 0 to 86399 (total seconds in a day)
    seconds = np.random.randint(0, 86400)
    random_timestamp = base_date + timedelta(seconds=int(seconds))
    return random_timestamp.strftime("%Y-%m-%d %H:%M:%S")

# Generate synthetic test data files
for i in range(num_files):
    data = []
    for j in range(num_rows):
        plate = np.random.randint(0, 5)  # Assuming 5 drivers (0 to 4)
        longitude = np.random.uniform(114.0, 115.0)
        latitude = np.random.uniform(22.0, 23.0)
        time_str = random_time()
        status = np.random.choice([0, 1])
        data.append([plate, longitude, latitude, time_str, status])
    
    df = pd.DataFrame(data, columns=columns)
    file_path = os.path.join(test_dir, f"small_test_day_{i+1}.csv")
    df.to_csv(file_path, index=False)

# Create a zip file containing the generated test data
zip_file_path = "./test_data/small_test_data.zip"
shutil.make_archive(zip_file_path.replace(".zip", ""), 'zip', test_dir)

# List the contents of the directory to confirm the zip file creation
print("Generated files:", os.listdir("./test_data/"))
