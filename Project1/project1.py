import requests
import json

GITHUB_API = 'Token'

headers ={
      'Authorization': GITHUB_API, # replace <TOKEN> with your token
    }
# collect data by users API
id_ = 0
response = requests.get('https://api.github.com/users?since='+str(id_),headers=headers)
data = response.json()

# collect data by search API
# response = requests.get('https://api.github.com/search/users?q=created:>=2025-01-22&sort=joined&order=desc',headers=headers)
# data = response.json()

json_formatted_str = json.dumps(data, indent=2)
print(json_formatted_str)