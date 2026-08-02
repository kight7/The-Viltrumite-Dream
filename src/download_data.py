import os
import requests
import pandas as pd
from requests.exceptions import RequestException

URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
OUTPUT_FILE = "data/cumulative.csv"

def download_data():
    print(f"Downloading data from NASA Exoplanet Archive...")
    try:
        response = requests.get(URL, stream=True, timeout=60)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Data successfully downloaded to {OUTPUT_FILE}")

        df = pd.read_csv(OUTPUT_FILE)
        print(f"Dataset successfully loaded.")
        print(f"Row count: {df.shape[0]}")
        print(f"Column count: {df.shape[1]}")
        
    except RequestException as e:
        print(f"Error downloading data: {e}")
    except pd.errors.EmptyDataError:
        print(f"Error: The downloaded CSV file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_data()
