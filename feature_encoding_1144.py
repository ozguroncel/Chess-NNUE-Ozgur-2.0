import chess
import pygame
import torch
import numpy as np
from collections import defaultdict
import time
import torch.nn as nn

# Data Preparation - Slight More Advanced Feature Extraction (1144 Edition)
def fen_to_detailed_feature_vector(fen):
    board = chess.Board(fen)
    feature_vector = np.zeros(1144)  
    idx = 0

    # Fine-grained Piece-Square Interactions
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        feature_vector[idx] = 1 if piece.color == chess.WHITE else -1
        idx += 1

    # King Safety Zones
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    for square in chess.SquareSet(chess.BB_KING_ATTACKS[white_king_square]):
        feature_vector[idx] = board.is_attacked_by(chess.BLACK, square)
        idx += 1
    for square in chess.SquareSet(chess.BB_KING_ATTACKS[black_king_square]):
        feature_vector[idx] = board.is_attacked_by(chess.WHITE, square)
        idx += 1

    # Legal Move Categorization
    for piece_type in chess.PIECE_TYPES:
        white_moves = list(board.legal_moves)
        black_moves = list(board.legal_moves)
        feature_vector[idx] = count_attack_moves(board, white_moves, piece_type, chess.WHITE)
        idx += 1
        feature_vector[idx] = count_defense_moves(board, white_moves, piece_type, chess.WHITE)
        idx += 1
        feature_vector[idx] = count_attack_moves(board, black_moves, piece_type, chess.BLACK)
        idx += 1
        feature_vector[idx] = count_defense_moves(board, black_moves, piece_type, chess.BLACK)
        idx += 1

    # 4. Advanced Pawn Structure 
    for square, piece in piece_map.items():
        if piece.piece_type == chess.PAWN:
            feature_vector[idx] = calculate_pawn_tension(board, square)
            idx += 1
            feature_vector[idx] = calculate_pawn_push_potential(board, square)
            idx += 1

    # 5. Center Control and Flank Control
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5, chess.C4, chess.C5, chess.F4, chess.F5]
    for square in center_squares:
        feature_vector[idx] = board.is_attacked_by(chess.WHITE, square)
        idx += 1
        feature_vector[idx] = board.is_attacked_by(chess.BLACK, square)
        idx += 1

    # 6. Rook on open files, queen on open diagonals
    for file_idx in range(8):
        is_open_file = True
        for rank in range(8):
            if board.piece_type_at(chess.square(file_idx, rank)) == chess.PAWN:
                is_open_file = False
                break
        feature_vector[idx] = 1 if is_open_file and board.piece_type_at(chess.square(file_idx, 0)) == chess.ROOK else 0
        idx += 1
        feature_vector[idx] = 1 if is_open_file and board.piece_type_at(chess.square(file_idx, 7)) == chess.ROOK else 0
        idx += 1

    # 7. Knight Outposts
    for square in chess.SQUARES:
        if board.piece_type_at(square) == chess.KNIGHT:
            feature_vector[idx] = is_knight_on_outpost(board, square)
            idx += 1

    # 8. Bishop Pair and Control of Long Diagonals
    feature_vector[idx] = has_bishop_pair(board, chess.WHITE)
    idx += 1
    feature_vector[idx] = has_bishop_pair(board, chess.BLACK)
    idx += 1

    for square in [chess.A1, chess.H8, chess.A8, chess.H1]:  # Main diagonals
        feature_vector[idx] = board.is_attacked_by(chess.WHITE, square)
        idx += 1
        feature_vector[idx] = board.is_attacked_by(chess.BLACK, square)
        idx += 1

    # 9. Material Imbalance 
    material_imbalance_white, material_imbalance_black = calculate_material_imbalance(board)
    feature_vector[idx] = material_imbalance_white
    idx += 1
    feature_vector[idx] = material_imbalance_black
    idx += 1

    # 10. Future Positional Gain Estimates 
    feature_vector[idx] = future_positional_gain(board, chess.WHITE)
    idx += 1
    feature_vector[idx] = future_positional_gain(board, chess.BLACK)
    idx += 1

    # 11. Endgame-Specific Features
    feature_vector[idx] = king_activity_endgame(board, chess.WHITE)
    idx += 1
    feature_vector[idx] = king_activity_endgame(board, chess.BLACK)
    idx += 1
    feature_vector[idx] = calculate_promotion_potential(board, chess.WHITE)
    idx += 1
    feature_vector[idx] = calculate_promotion_potential(board, chess.BLACK)
    idx += 1

    # 12. Tactical Motifs 
    feature_vector[idx] = count_pins(board, chess.WHITE)
    idx += 1
    feature_vector[idx] = count_pins(board, chess.BLACK)
    idx += 1
    feature_vector[idx] = count_skewers(board, chess.WHITE)
    idx += 1
    feature_vector[idx] = count_skewers(board, chess.BLACK)
    idx += 1
    feature_vector[idx] = count_discovered_attacks(board, chess.WHITE)
    idx += 1
    feature_vector[idx] = count_discovered_attacks(board, chess.BLACK)
    idx += 1
    feature_vector[idx] = count_forks(board)
    idx += 1

    return feature_vector

# Required helper Functions

def count_attack_moves(board, moves, piece_type, color):
    return sum(1 for move in moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == piece_type and board.is_capture(move))

def count_defense_moves(board, moves, piece_type, color):
    return sum(1 for move in moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == piece_type and board.is_attacked_by(color, move.from_square))

def calculate_pawn_tension(board, square):
    return 1 if board.is_attacked_by(chess.PAWN, square) else 0

def calculate_pawn_push_potential(board, square):
    color = board.color_at(square)
    rank = chess.square_rank(square)
    return 7 - rank if color == chess.WHITE else rank

def is_knight_on_outpost(board, square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    if rank >= 4 and board.piece_type_at(square) == chess.KNIGHT:
        for adj_file in [file - 1, file + 1]:
            if 0 <= adj_file < 8:
                for adj_rank in [rank - 1, rank + 1]:
                    if 0 <= adj_rank < 8 and board.piece_type_at(chess.square(adj_file, adj_rank)) == chess.PAWN:
                        return 1
    return 0

def has_bishop_pair(board, color):
    bishops = [piece for piece in board.piece_map().values() if piece.piece_type == chess.BISHOP and piece.color == color]
    return 1 if len(bishops) == 2 else 0

def calculate_material_imbalance(board):
    material_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    white_material = sum(material_values.get(piece.piece_type, 0) for piece in board.piece_map().values() if piece.color == chess.WHITE)
    black_material = sum(material_values.get(piece.piece_type, 0) for piece in board.piece_map().values() if piece.color == chess.BLACK)
    return white_material - black_material, black_material - white_material

def future_positional_gain(board, color):
    activity_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            activity_score += len(board.attacks(square))  # More attacking squares = more potential
    return activity_score

def king_activity_endgame(board, color):
    """Evaluate king's activity in the endgame by checking proximity to key squares."""
    king_square = board.king(color)
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    distance_to_center = min(chess.square_distance(king_square, square) for square in center_squares)
    return 8 - distance_to_center  # The closer to the center, the higher the score

def calculate_promotion_potential(board, color):
    """Estimate the likelihood of promotion based on pawn positions."""
    promotion_potential = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            rank = chess.square_rank(square)
            promotion_potential += rank if color == chess.WHITE else 7 - rank
    return promotion_potential

def count_pins(board, color):
    return sum(1 for square in chess.SQUARES if board.is_pinned(color, square))

def count_skewers(board, color):
    """Count skewers (like pins but the more valuable piece is in front)."""
    skewers = 0
    for square in chess.SQUARES:
        attacked_piece = board.piece_at(square)
        if attacked_piece and attacked_piece.color == color:
            attackers = board.attackers(not color, square)
            for attacker in attackers:
                attacking_piece = board.piece_at(attacker)
                if attacking_piece and attacking_piece.piece_type > attacked_piece.piece_type:
                    skewers += 1
    return skewers


def count_discovered_attacks(board, color):
    # Placeholder for now, a full implementation would require deeper analysis
    return 0

def count_forks(board):
    """Counts potential forks in the position (knight forks, queen forks, etc.)."""
    forks = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and (piece.piece_type == chess.KNIGHT or piece.piece_type == chess.QUEEN):
            attackers = list(board.attacks(square))
            enemy_pieces = [target for target in attackers if board.piece_at(target) and board.piece_at(target).color != piece.color]
            if len(enemy_pieces) >= 2:
                forks += 1
    return forks

# Defining the model for 1144 features
class CustomNNUE(nn.Module):
    def __init__(self, input_size=1144, hidden_size1=512, hidden_size2=256, hidden_size3=128, output_size=1):
        super(CustomNNUE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.bn1(self.relu1(self.fc1(x)))
        x = self.bn2(self.relu2(self.fc2(x)))
        x = self.bn3(self.relu3(self.fc3(x)))
        x = self.fc4(x)
        return x