import os.path
import random
import time
import matplotlib.pyplot as plt
import requests
import json
from configs import *

key = GITHUB_APY_KEY

headers ={
      'Authorization': key,
    }

GITHUB_JSON_DATA = './data/data.json'

def check_rate_limit():
    """Check GitHub API rate limit and pause if needed."""
    url = "https://api.github.com/rate_limit"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        rate_data = response.json()
        remaining = rate_data['rate']['remaining']
        reset_time = rate_data['rate']['reset']  # UNIX timestamp

        if remaining == 0:
            sleep_time = reset_time - time.time()  # Calculate how long to sleep
            if sleep_time > 0:
                print(f"Rate limit exceeded. Sleeping for {int(sleep_time)} seconds.")
                time.sleep(sleep_time)

        return remaining
    else:
        print(f"Error checking rate limit: {response.status_code}")
        return 5000

def fetch_users(id, count):
    response = requests.get(f'https://api.github.com/users?since={id}&per_page={count}', headers=headers)

    if response.status_code == 200:
        users = response.json()
        save_data(users)
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

def load_data(filename=GITHUB_JSON_DATA):
    if os.path.exists(filename):
        with open(filename,'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []
    return []

def save_data(new_data, filename=GITHUB_JSON_DATA):
    existing_data = load_data(filename)

    dict = {user['id']:user for user in existing_data}

    for user in new_data:
        dict[user['id']] = user

    with open(filename, 'w') as file:
        json.dump(list(dict.values()), file, indent=2)

def sampling_users(start_id, end_id, num_samples):
    return random.sample(range(start_id,end_id), num_samples)

def valid_users(sampled_users, data):
    valid_users = 0
    fetched_ids = {user['id'] for user in data}

    for uid in sampled_users:
        check_rate_limit()
        if uid in fetched_ids:
            valid_users += 1
        else:
            response = requests.get(f'https://api.github.com/user/{uid}', headers=headers)
            if response.status_code == 200:
                valid_users += 1

        time.sleep(1)

    return valid_users

def estimate_valid_users(sampled_users, valid, total_ids):
    sample_size = len(sampled_users)
    estimated_users = (valid / sample_size) * total_ids

    return estimated_users

def evaluate_unbiasedness(runs=10, num_samples=50, total_ids=10000):
    estimates = []
    data = fetch_users(0, 100)  # Fetch sample data before running tests

    for _ in range(runs):
        sampled_ids = sampling_users(1, total_ids, num_samples)
        valid_count = valid_users(sampled_ids, data)  # Pass data here
        estimated_users = estimate_valid_users(sampled_ids, valid_count, total_ids)
        estimates.append(estimated_users)

    # Box plot visualization
    plt.figure(figsize=(10, 6))
    plt.boxplot(estimates, patch_artist=True,
                boxprops=dict(color="blue"), medianprops=dict(color="red"),
                flierprops=dict(marker='o', color='red', alpha=0.5))
    plt.axhline(y=sum(estimates) / len(estimates), color='blue', linestyle='-', label="True Estimated Count")
    plt.xlabel("Run Number")
    plt.ylabel("Estimated Number of GitHub Users")
    plt.title("Box Plot of Estimated GitHub User Count")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def main():
    since_id = 0
    requests_count = 100
    for i in range(requests_count):
        data = fetch_users(since_id, 100)
        if not data:
            break
        since_id = data[-1]['id']
        time.sleep(1)

    data = load_data()
    sample_size = min(500, len(data))  # Dynamically adjust sample size
    sampled_ids = sampling_users(1, 10000, sample_size)

    valid_count = valid_users(sampled_ids, data)

    estimated_users = estimate_valid_users(sampled_ids, valid_count, total_ids=10000)
    print(f"Estimated valid users in range 1-10,000: {estimated_users}")
    #evaluate_unbiasedness()

if __name__ == '__main__':
    main()