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
# collect data by users API
# id_ = 0
# response = requests.get('https://api.github.com/users?since='+str(id_),headers=headers)
# data = response.json()
#
# # collect data by search API
# # response = requests.get('https://api.github.com/search/users?q=created:>=2025-01-22&sort=joined&order=desc',headers=headers)
# # data = response.json()
#
# json_formatted_str = json.dumps(data, indent=2)
# print(json_formatted_str)

def fetch_users(id, count):
    response = requests.get(f'https://api.github.com/users?since={id}&per_page={count}', headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

def sampling_users(start_id, end_id, num_samples):
    return random.sample(range(start_id,end_id), num_samples)

def valid_users(sampled_users):
    valid_users = 0
    for uid in sampled_users:
        response = requests.get(f'https://api.github.com/user/{uid}')

        if response.status_code == 200:
            valid_users += 1
        elif response.status_code == 404:
            pass

        time.sleep(1)

    return valid_users

def estimate_valid_users(sampled_users, valid, total_ids):
    sample_size = len(sampled_users)
    estimated_users = (valid / sample_size) * total_ids

    return estimated_users

def evaluate_unbiasedness(runs=10, num_samples=50, total_ids=10000):
    estimates = []

    for _ in range(runs):
        sampled_ids = sampling_users(1, total_ids, num_samples)
        valid_count = valid_users(sampled_ids)
        estimated_users = estimate_valid_users(sampled_ids, valid_count, total_ids)
        estimates.append(estimated_users)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.scatter(range(runs), estimates, label="Estimates")
    plt.axhline(y=sum(estimates) / len(estimates), color='r', linestyle='--', label="Average Estimate")
    plt.xlabel("Run")
    plt.ylabel("Estimated Users")
    plt.title("Unbiasedness of the Estimator")
    plt.legend()
    plt.show()

def main():
    data = fetch_users(0,100)
    print(json.dumps(data, indent=2))

    # Select a small known range (1-10,000)

    sampled_ids = sampling_users(1, 10000, 50)

    # Get valid users from the sample
    valid_count = valid_users(sampled_ids)

    # Estimate total valid users
    estimated_users = estimate_valid_users(sampled_ids, valid_count, total_ids=10000)
    print(f"Estimated valid users in range 1-10,000: {estimated_users}")
    evaluate_unbiasedness()

if __name__ == '__main__':
    main()