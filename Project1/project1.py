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

def fetch_users(id, count):
    response = requests.get(f'https://api.github.com/users?since={id}&per_page={count}', headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

def check_rate_limit():
    """Check GitHub API rate limit and sleep if needed."""
    url = "https://api.github.com/rate_limit"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        rate_data = response.json()
        remaining = rate_data['rate']['remaining']
        reset_time = rate_data['rate']['reset']

        if remaining == 0:
            sleep_time = reset_time - time.time()
            if sleep_time > 0:
                print(f"Rate limit exceeded. Sleeping for {int(sleep_time)} seconds.")
                time.sleep(sleep_time)

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
    data = fetch_users(0, 100)
    print(json.dumps(data, indent=2))

    sample_size = min(500, len(data))  # Dynamically adjust sample size
    sampled_ids = sampling_users(1, 10000, sample_size)
    print(sampled_ids)

    valid_count = valid_users(sampled_ids, data)

    estimated_users = estimate_valid_users(sampled_ids, valid_count, total_ids=10000)
    print(f"Estimated valid users in range 1-10,000: {estimated_users}")
    evaluate_unbiasedness()

if __name__ == '__main__':
    main()