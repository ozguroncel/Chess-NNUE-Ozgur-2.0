import chess.pgn
import chess.engine
import os

# Set up the paths
base_dir = None # define your preferred base directory
historic_games_dir = os.path.join(base_dir, 'historic_games')
output_dir = os.path.join(base_dir, 'evaluated_positions')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Paths to the PGN files
lichess_pgn_path = os.path.join(historic_games_dir, 'lichess_games.pgn')
chesscom_pgn_path = os.path.join(historic_games_dir, 'chesscom_games.pgn')

# Path to the to be used NNUE file, ideally, check what is the current Stockfish version, this may change. Mine was from August 2024.
nnue_path = os.path.join(base_dir, 'nn-b1a57edbea57.nnue')  # Replace with the actual file name

# Initialize the Stockfish engine with NNUE support
engine = chess.engine.SimpleEngine.popen_uci("stockfish-windows-x86-64-avx2")
engine.configure({"EvalFile": nnue_path})

# Function to evaluate a single position using NNUE
def evaluate_position(board):
    # Analyze the position using NNUE with a short time limit for quick evaluations
    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    return info["score"].relative.score(mate_score=10000)

# Processes the PGN file and save evaluations
def process_pgn_file(pgn_path, output_file):
    with open(pgn_path) as pgn_file, open(output_file, 'w', encoding='utf-8') as out_file:
        game_number = 1
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # End of file
            board = game.board()
            out_file.write(f"Game {game_number}\n")
            for move in game.mainline_moves():
                board.push(move)
                evaluation = evaluate_position(board)
                fen = board.fen()
                out_file.write(f"{fen} | Evaluation: {evaluation}\n")
            game_number += 1
            out_file.write("\n")
        print(f"Processed {game_number - 1} games from {pgn_path}")

# Process both PGN files
process_pgn_file(lichess_pgn_path, os.path.join(output_dir, 'lichess_evaluations.txt'))
process_pgn_file(chesscom_pgn_path, os.path.join(output_dir, 'chesscom_evaluations.txt'))

# Cleanup
engine.quit()
