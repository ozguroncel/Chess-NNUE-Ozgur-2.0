from chess_ai_base_1144 import select_best_move, render_board, get_human_move_from_gui
import chess
import pygame

# Function to start a game against the AI. In this setup, we start as whites.
def play_with_gui():
    pygame.init()
    print("Pygame initialized successfully.")

    screen = pygame.display.set_mode((640, 640))  
    pygame.display.set_caption("Chess AI")
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # You as White using drag-and-drop
            move = get_human_move_from_gui(board, screen)
            board.push(move)
        else:
            # AI plays as Black
            print("AI is thinking...")
            ai_move = select_best_move(board, depth=5, time_limit=60) # Depth and time for AI configurable. Higher, the better.
            board.push(ai_move)

        render_board(board, screen)
        pygame.display.update()

    print("Game over!")
    print(board.result())

if __name__ == "__main__":
    play_with_gui()
