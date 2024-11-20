import chess
import pygame
import torch
import numpy as np
from collections import defaultdict
import time
import torch.nn as nn
import feature_encoding_1144


# Part 1 - Load and Prepare the model
model = feature_encoding_1144.CustomNNUE(input_size=1144)
model.load_state_dict(torch.load("3rd_model.pth")) # 3rd model refers to the 1144 version. Adjust accordingly, if you train your own NNUE.
print("Model loaded successfully")
model.eval()
print("Model set to eval/infer mode")


# Part 2: Evaluation Function using NNUE
def evaluate_position_with_model(board):
    fen = board.fen()
    feature_vector = feature_encoding_1144.fen_to_detailed_feature_vector(fen)
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        evaluation = model(feature_tensor).item()
    return evaluation


# Part 3: NegaMax with Alpha-Beta Pruning
def negamax(board, depth, alpha, beta, color, time_limit=60):
    start_time = time.time()

    def timed_out():
        return time.time() - start_time > time_limit

    if depth == 0 or board.is_game_over() or timed_out():
        return color * evaluate_position_with_model(board)

    max_eval = float('-inf')
    legal_moves = sorted(board.legal_moves, key=lambda move: move_ordering_heuristic(board, move), reverse=True)

    for move in legal_moves:
        board.push(move)
        eval = -negamax(board, depth - 1, -beta, -alpha, -color, time_limit)
        board.pop()

        max_eval = max(max_eval, eval)
        alpha = max(alpha, eval)
        if alpha >= beta:
            break

        if timed_out():
            break  # Stop searching if time limit is exceeded

    return max_eval

def move_ordering_heuristic(board, move):
    if board.is_capture(move):
        return 10  # Prioritize captures
    if board.gives_check(move):
        return 5  # Prioritize checks
    return 0

# Select best move with time limit
def select_best_move(board, depth=3, time_limit=60):
    best_move = None
    best_value = float('-inf')

    start_time = time.time()

    for move in board.legal_moves:
        if time.time() - start_time > time_limit:
            break
        board.push(move)
        move_value = -negamax(board, depth - 1, float('-inf'), float('inf'), -1, time_limit)
        board.pop()

        if move_value > best_value:
            best_value = move_value
            best_move = move

    return best_move

# Part 4: GUI for playing with drag-and-drop
# Initialize Pygame
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
FPS = 30

# Load chess pieces
PIECE_IMAGES = {}
piece_files = {
    'P': 'wP.png',
    'R': 'wR.png',
    'N': 'wN.png',
    'B': 'wB.png',
    'Q': 'wQ.png',
    'K': 'wK.png',
    'p': 'bP.png',
    'r': 'bR.png',
    'n': 'bN.png',
    'b': 'bB.png',
    'q': 'bQ.png',
    'k': 'bK.png'
}

for piece, filename in piece_files.items():
    PIECE_IMAGES[piece] = pygame.transform.scale(pygame.image.load(f'images/{filename}'), (SQUARE_SIZE, SQUARE_SIZE))

# Draw the actual chessboard
def render_board(board, screen):
    
    for row in range(8):
        for col in range(8):
            color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            screen.blit(PIECE_IMAGES[piece.symbol()], (col * SQUARE_SIZE, row * SQUARE_SIZE))

    
# Handles human moves from the GUI      
def get_human_move_from_gui(board, screen):
    selected_piece = None
    drag_start_square = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drag_start_square = get_square_under_mouse()
                if board.piece_at(drag_start_square):
                    selected_piece = board.piece_at(drag_start_square)
            elif event.type == pygame.MOUSEBUTTONUP:
                if selected_piece:
                    drag_end_square = get_square_under_mouse()
                    move = chess.Move(drag_start_square, drag_end_square)
                    if move in board.legal_moves:
                        return move

        # Render the board and handle dragging
        render_board(board, screen)
        if selected_piece:
            piece_img = PIECE_IMAGES[selected_piece.symbol()]
            mouse_x, mouse_y = pygame.mouse.get_pos()
            screen.blit(piece_img, (mouse_x - SQUARE_SIZE // 2, mouse_y - SQUARE_SIZE // 2))

        pygame.display.flip()

def get_square_under_mouse():
    """Return the chess square index under the mouse pointer."""
    mouse_x, mouse_y = pygame.mouse.get_pos()
    col = mouse_x // SQUARE_SIZE
    row = mouse_y // SQUARE_SIZE
    square = row * 8 + col
    return square
