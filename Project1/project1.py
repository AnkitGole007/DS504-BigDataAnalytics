import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from configs import *

key = GITHUB_APY_KEY

headers = {
      'Authorization': key,
    }

GITHUB_JSON_DATA = './data/data.json'

def check_rate_limit():
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

        return remaining
    else:
        print(f"Error checking rate limit: {response.status_code}")
        return 5000

def fetch_users(total_users):
    all_users = load_data()
    fetched_users = len(all_users)

    if not all_users:
        since_id = 0
        print(f'ID initialized: {since_id}')
    else:
        since_id = all_users[-1]['id']
        print(f'ID fetched: {since_id}')

    while fetched_users < total_users:
        check_rate_limit()

        response = requests.get(f'https://api.github.com/users?since={since_id}&per_page=100', headers=headers)

        if response.status_code == 200:
            users = response.json()
            if not users:
                break

            save_data(users)
            since_id = users[-1]['id']
            fetched_users = len(load_data())
            print(f"Fetched {fetched_users}/{total_users} users")
        else:
            print(f"Error {response.status_code}: {response.text}")
            break

        time.sleep(0.5)
    print(f"Finished fetching {len(load_data())} users.")

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

    d = {user['id']: user for user in existing_data}

    for user in new_data:
        d[user['id']] = user

    with open(filename, 'w') as file:
        json.dump(list(d.values()), file, indent=2)

def sampling_users(start_id, end_id, num_samples):
    return np.random.randint(start_id, end_id + 1, size=num_samples)

def valid_users(sampled_users, data):
    valid_user = np.isin(sampled_users, list(data)).sum()

    return valid_user

def estimate_valid_users(sampled_users, valid, total_ids):
    sample_size = len(sampled_users)
    estimated_users = (valid / sample_size) * total_ids

    return estimated_users

def evaluate_unbiasedness(ground_truth, runs, num_samples,total_ids=10000):
    sample_sizes = list(range(500,num_samples,500))
    all_estimates = []
    data = load_data()
    fetch_users = {user['id'] for user in data}

    for m in sample_sizes:
        print(f"Processing sample size: {m}")
        estimates = []
        for _ in range(runs):
            sampled_ids = sampling_users(1, total_ids, m)
            valid_count = valid_users(sampled_ids, fetch_users)
            estimated_users = estimate_valid_users(sampled_ids, valid_count, total_ids)
            estimates.append(estimated_users)
        all_estimates.append(estimates)

    # Box plot visualization
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_estimates, patch_artist=True,
                boxprops=dict(color="blue"), medianprops=dict(color="red"),
                flierprops=dict(marker='o', color='red', alpha=0.5))
    plt.xticks(range(1, len(sample_sizes) + 1), labels=sample_sizes)
    plt.xlabel("Sample Size (m)")
    plt.ylabel("Estimated Number of GitHub Users")
    plt.title("Distribution of User Estimates by Sample Size (100 runs per size)")

    if ground_truth is not None:
        plt.axhline(y=ground_truth, color='green', linestyle='--',
                    linewidth=2, label=f'Ground Truth: {ground_truth:.1f}')
        plt.legend()

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def main():
    fetch_users(total_users=10000)

    data = load_data()
    fetched_users = {user['id'] for user in data}
    sample_size = 4000
    sampled_ids = sampling_users(1, 10000, sample_size)

    valid_count = valid_users(sampled_ids, fetched_users)

    estimated_users = estimate_valid_users(sampled_ids, valid_count, total_ids=10000)
    print(f"Estimated valid users in range 1-10,000: {estimated_users}")
    evaluate_unbiasedness(ground_truth=estimated_users, runs=100, num_samples=sample_size)

if __name__ == '__main__':
    main()