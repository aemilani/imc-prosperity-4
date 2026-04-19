import requests
import time
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime


def get_leaderboard(leaderboard_type: str = 'OVERALL'):
    """
    type: 'OVERALL', 'ALGO', 'MANUAL'
    """
    url = 'https://3dzqiahkw1.execute-api.eu-west-1.amazonaws.com/prod/leaderboard'

    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'origin': 'https://prosperity.imc.com',
        'referer': 'https://prosperity.imc.com/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
    }

    params = {
        'type': 'OVERALL',
        'page': '1',
        'limit': '100'
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    pages = int(np.ceil(data['data']['total'] / data['data']['pageSize']))

    all_data = []
    for page in tqdm(range(1, pages + 1)):
        params = {
            'type': leaderboard_type,
            'page': str(page),
            'limit': '100'
        }
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        all_data.extend(data['data']['items'])

        time.sleep(np.random.uniform(0.4, 0.6))

    return all_data


if __name__ == '__main__':
    for leaderboard_type in ['OVERALL', 'ALGO', 'MANUAL']:
        print(f"Loading {leaderboard_type} leaderboard data ...")
        data = get_leaderboard(leaderboard_type)

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        with open(f'../leaderboard/raw/{leaderboard_type}_{timestamp}.json', 'w') as f:
            json.dump(data, f, indent=4)
        print(f"{leaderboard_type} leaderboard saved.\n")