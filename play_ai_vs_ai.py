from chess_ai_base_1144 import select_best_move, render_board
import chess
import pygame

# Self play function to watch AI play against itself
def ai_vs_ai():
    pygame.init()
    screen = pygame.display.set_mode((640, 640))  
    pygame.display.set_caption("Chess AI: Self-Play")
    board = chess.Board()

    while not board.is_game_over():
        render_board(board, screen)
        pygame.display.update()

        if board.turn == chess.WHITE:
            print("White AI is thinking...")
            move = select_best_move(board, depth=5, time_limit=60) # depth and time are configurable. Higher, the better
        else:
            print("Black AI is thinking...")
            move = select_best_move(board, depth=5, time_limit=60) # Same for black AI

        board.push(move)

    print("Game over!")
    print(board.result())

if __name__ == "__main__":
    ai_vs_ai()
