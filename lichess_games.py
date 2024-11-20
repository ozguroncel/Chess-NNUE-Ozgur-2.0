import requests
import os

# Directory setup
base_dir = None # make sure to define the desired location
historic_games_dir = os.path.join(base_dir, 'historic_games')
if not os.path.exists(historic_games_dir):
    os.makedirs(historic_games_dir)

# Lichess username
lichess_username = None  # replace with your actual Lichess username

# Lichess API URL
lichess_api_url = f'https://lichess.org/api/games/user/{lichess_username}?max=2000&format=pgn'

# Request to get games in PGN format
response = requests.get(lichess_api_url, headers={'Accept': 'application/x-chess-pgn'})

if response.status_code == 200:
    pgn_data = response.text
    pgn_file_path = os.path.join(historic_games_dir, 'lichess_games.pgn')
    with open(pgn_file_path, 'w', encoding='utf-8') as f:  # Specify 'utf-8' encoding here
        f.write(pgn_data)
    print(f'Lichess games downloaded successfully and saved to {pgn_file_path}')
else:
    print(f'Failed to download Lichess games: {response.status_code}')
