import pygame
import chess
import chess.svg

# Initialize Pygame
pygame.init()

# Constants
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

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)

# Initialize screen & the chess board
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption('Chess')
board = chess.Board()

# Helper functions
def draw_board():
    for row in range(8):
        for col in range(8):
            color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            screen.blit(PIECE_IMAGES[piece.symbol()], (col * SQUARE_SIZE, row * SQUARE_SIZE))

def get_square_under_mouse():
    mouse_x, mouse_y = pygame.mouse.get_pos()
    col = mouse_x // SQUARE_SIZE
    row = mouse_y // SQUARE_SIZE
    square = row * 8 + col
    return square

def main():
    """Main game loop."""
    clock = pygame.time.Clock()
    selected_piece = None
    drag_start_square = None

    running = True
    while running:
        draw_board()
        draw_pieces()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drag_start_square = get_square_under_mouse()
                if board.piece_at(drag_start_square):
                    selected_piece = board.piece_at(drag_start_square)
            elif event.type == pygame.MOUSEBUTTONUP:
                if selected_piece:
                    drag_end_square = get_square_under_mouse()
                    move = chess.Move(drag_start_square, drag_end_square)
                    if move in board.legal_moves:
                        board.push(move)
                    selected_piece = None

        if selected_piece:
            draw_board()
            draw_pieces()
            piece_img = PIECE_IMAGES[selected_piece.symbol()]
            mouse_x, mouse_y = pygame.mouse.get_pos()
            screen.blit(piece_img, (mouse_x - SQUARE_SIZE // 2, mouse_y - SQUARE_SIZE // 2))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
