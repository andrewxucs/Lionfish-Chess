from __future__ import print_function
import re, time
from itertools import count
from collections import OrderedDict, namedtuple

# This program is only compatible to Python 3
# Limitations include only white pieces for user, no underpromotions, and no 50 move draw rule

# relative value of each piece according to AlphaZero
piece_relative_value = { 'P': 100, 'N': 305, 'B': 333, 'R': 563, 'Q': 950, 'K': 60000 }

# PST assigns a value for each square to incentivise the correct placement of each piece
# The tables below are compiled by Thomas Ahle when approximating Stockfish's model
piece_squared_tables = {
     'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}

# Pad the PST tables to detect illegal moves that don't stay within the board

for piece, table in piece_squared_tables.items():
    #pad 1 zero each side of each row and add piece value to every entry
    padrow = lambda row: (0,) + tuple(x + piece_relative_value[piece] for x in row) + (0,)
    piece_squared_tables[piece] = sum((padrow(table[i*8 : i*8+8]) for i in range(8)), ())

    #add 20 zeros in front of and after each PST
    piece_squared_tables[piece] = (0,)*20 + piece_squared_tables[piece] + (0,)*20

# The board is represented with a length 120 string
# The following multi-line string with parentheses has 12 rows, each containing 10 characters
starting_position = (
    '         \n'
    '         \n'
    ' rnbqkbnr\n'
    ' pppppppp\n'
    ' ........\n'
    ' ........\n' 
    ' ........\n'
    ' ........\n'
    ' PPPPPPPP\n'
    ' RNBQKBNR\n'
    '         \n'
    '         \n'
)

# The positions of the corner coordinates in the string
a1 = 91
h1 = 98
a8 = 21
h8 = 28

# Defining legal directions for each piece
# U = Up, L = Left, D = Down, R = Right
U, R, D, L = -10, 1, 10, -1
legal_directions = {
    'P': (U, U + U, U + L, U + R),
    'N': (U + U + R, R + U + R, R + D + R, D + D + R, D + D + L, L + D + L, L + U + L, U + U + L),
    'B': (U + R, D + R, D + L, U + L),
    'R': (U, R, D, L),
    'Q': (U, R, D, L, U + R, D + R, D + L, U + L),
    'K': (U, R, D, L, U + R, D + R, D + L, U + L)
}

# MATE_LOWER is used to detect when a position is a checkmate, regardless of material imbalance
# MATE_UPPER sets a high score limit for checkmate positions to prioritize ending the game by checkmate over other moves
# When mate is detected, we shall set MATE_UPPER - plies
# E.g. Mate in 1 will be MATE_UPPER - 2
MATE_LOWER = piece_relative_value['K'] - 10 * piece_relative_value['Q']
MATE_UPPER = piece_relative_value['K'] + 10 * piece_relative_value['Q']

# Transposition table upper limit on the number of elements
MAX_TABLE_SIZE = 1e8

# Tuning search constants
# QS Limit imposed to force the engine to make a move
SEARCH_LIMIT = 150
EVALULATION_ROUGHNESS = 20

Range = namedtuple('Range', 'lower upper')

class Position(namedtuple('Position', 'board score castling_rights opponent_castling_rights en_passant_square king_square')):

    # board: 120 character string
    # score: evaluation
    # castling_rights: [queen side, king side]
    # opponent_castling_rights:[king side, queen side]
    # en_passant_square
    # king_square

    # Generator that produces a list of legal moves
    def generate_moves(self):

        # Check every square
        for index, square in enumerate(self.board):

            # Check if the piece belongs to the user
            if not square.isupper():
                continue
            
            # Check each direction for this individual piece
            for direction in legal_directions[square]:

                # For each direction, yield legal moves
                for target_index in count(index + direction, direction):
                    
                    target_square = self.board[target_index]
                    
                    # Pieces cannot go outside the board
                    # Pieces cannot capture friendly pieces
                    if target_square.isspace() or target_square.isupper():
                        break
                    # Pawns usually move up one square
                    # Unmoved pawns can move up two squares
                    if square == 'P':
                        if direction in (U, U+U) and target_square != '.':
                            break
                        if direction == U+U and (index < a1+U or self.board[index+U] != '.'):
                            break
                        # Pawn capture
                        if direction in (U+L, U+R) and target_square == '.' and target_index not in (self.en_passant_square, self.king_square):
                            break
                    
                    # Add legal move
                    yield (index, target_index)

                    # Ensures that no piece would move more than once in a turn
                    if square in 'PNK' or target_square.islower():
                        break

                    # Castling
                    if index == a1 and self.board[target_index+R] == 'K' and self.castling_rights[0]:
                        yield (target_index + R, target_index + L)
                    if index == h1 and self.board[target_index + L] == 'K' and self.castling_rights[1]:
                        yield (target_index + L, target_index + R)

    # Switch turn to opposing side
    def switch(self):
        # Reverse score evaluation and letter cases to fit the chess engine
        # ::-1 reverses the String
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.opponent_castling_rights,
            self.castling_rights,
            119-self.en_passant_square if self.en_passant_square else 0,
            119-self.king_square if self.king_square else 0
        )

    # Similar to switch(), but clears en_passant_square and king_square
    def voidmove(self):
        return Position(
            self.board[::-1].swapcase(),
            -self.score,
            self.opponent_castling_rights,
            self.castling_rights,
            0,
            0
        )

    # Changes the board after a move
    def move(self, move):
        starting_index, ending_index = move
        starting_square, ending_square = self.board[starting_index], self.board[ending_index]
        add_piece = lambda board, i, piece: board[:i] + piece + board[i+1:]
        board = self.board
        castling_rights, opponent_castling_rights, en_passant_square, king_square = self.castling_rights, self.opponent_castling_rights, 0, 0
        # New score is the sum of the initial score and the added score of the move
        score = self.score + self.value(move)
        # Remove piece and add piece to adjust board for the move
        board = add_piece(board, ending_index, board[starting_index])
        board = add_piece(board, starting_index, '.')
        # Check if castling rights should be disabled
        if starting_index == a1:
            castling_rights = (False, castling_rights[1])
        if starting_index == h1:
            castling_rights = (castling_rights[0], False)
        if ending_index == a8:
            opponent_castling_rights = (opponent_castling_rights[0], False)
        if ending_index == h8:
            opponent_castling_rights = (False, opponent_castling_rights[1])
        # Castling
        if starting_square == 'K':
            # Disable future castling rights
            castling_rights = (False, False)
            if abs(ending_index-starting_index) == 2:
                king_square = (starting_index+ending_index)//2
                board = add_piece(board, a1 if ending_index < starting_index else h1, '.')
                board = add_piece(board, king_square, 'R')
        # Pawn promotion, double move and en passant capture
        if starting_square == 'P':
            if a8 <= ending_index <= h8:
                board = add_piece(board, ending_index, 'Q')
            if ending_index - starting_index == 2*U:
                en_passant_square = starting_index + U
            if ending_index - starting_index in (U+L, U+R) and ending_square == '.':
                board = add_piece(board, ending_index+D, '.')
        # When a move is made, switch to opposing side
        return Position(board, score, castling_rights, opponent_castling_rights, en_passant_square, king_square).switch()

    # Calculates the score of a position
    def value(self, move):
        starting_index, ending_index = move
        starting_square, ending_square = self.board[starting_index], self.board[ending_index]
        # See score change from beginning to end of a move
        score = piece_squared_tables[starting_square][ending_index] - piece_squared_tables[starting_square][starting_index]
        # Capturing opponent material increases
        if ending_square.islower():
            score += piece_squared_tables[ending_square.upper()][119-ending_index]
        # If the king is castled, score increases due to increased safety
        if abs(ending_index-self.king_square) < 2:
            score += piece_squared_tables['K'][119-ending_index]
        # If castling is an option, score increases
        if starting_square == 'K' and abs(starting_index-ending_index) == 2:
            score += piece_squared_tables['R'][(starting_index+ending_index)//2]
            score -= piece_squared_tables['R'][a1 if ending_index < starting_index else h1]
        # Factor in how close a pawn is to promotion
        # Factor in En Passant
        if starting_square == 'P':
            if a8 <= ending_index <= h8:
                score += piece_squared_tables['Q'][ending_index] - piece_squared_tables['P'][ending_index]
            if ending_index == self.en_passant_square:
                score += piece_squared_tables['P'][119-(ending_index+D)]
        return score

# Least Recently Used (LRU) datastructure that removes the least used items when cache reaches capacity
class LRUCache:
    def __init__(self, size):
        self.od = OrderedDict()
        self.size = size

    def get(self, key, default=None):
        try:
            self.od.move_to_end(key)
        except KeyError:
            return default
        return self.od[key]

    def __setitem__(self, key, value):
        try:
            del self.od[key]
        except KeyError:
            if len(self.od) == self.size:
                self.od.popitem(last = False)
        self.od[key] = value

class Searcher:
    def __init__(self):
        # Transposition table to avoid re-calculating moves
        self.transposition_table_scores = LRUCache(MAX_TABLE_SIZE)
        self.transposition_table_moves = LRUCache(MAX_TABLE_SIZE)

        self.num_of_positions = 0

    # returns the score
    def bound(self, position, temp, depth, check = True):
        """ returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
        self.num_of_positions += 1

        # Check if depth is negative and fix to 0 if it is
        depth = max(depth, 0)

        # Since Lionfish is a king-capture engine, we must check if the king is still present
        if position.score <= -MATE_LOWER:
            return -MATE_UPPER

        # Check transposition table
        range = self.transposition_table_scores.get((position, depth, check), Range(-MATE_UPPER, MATE_UPPER))
        if range.lower >= temp and (not check or self.transposition_table_moves.get(position) is not None):
            return range.lower
        if range.upper < temp:
            return range.upper

        # Generator to find moves to search
        def moves_generator():
            # No move check
            if depth > 0 and not check and any(c in position.board for c in 'RBNQ'):
                yield None, -self.bound(position.voidmove(), 1-temp, depth-3, check = False)

            if depth == 0:
                yield None, position.score
            
            target_move = self.transposition_table_moves.get(position)
            if target_move and (depth > 0 or position.value(target_move) >= SEARCH_LIMIT):
                yield target_move, -self.bound(position.move(target_move), 1-temp, depth-1, check = False)
            
            # Check all other moves
            for move in sorted(position.generate_moves(), key=position.value, reverse=True):
                if depth > 0 or position.value(move) >= SEARCH_LIMIT:
                    yield move, -self.bound(position.move(move), 1-temp, depth-1, check=False)

        optimal = -MATE_UPPER
        for move, score in moves_generator():
            optimal = max(optimal, score)
            if optimal >= temp:
                self.transposition_table_moves[position] = move
                break

        # Check for stalemate
        if optimal < temp and optimal < 0 and depth > 0:
            is_stalemate = lambda position: any(position.value(m) >= MATE_LOWER for m in position.generate_moves())
            if all(is_stalemate(position.move(m)) for m in position.generate_moves()):
                check = is_stalemate(position.voidmove())
                optimal = -MATE_UPPER if check else 0

        if optimal >= temp:
            self.transposition_table_scores[(position, depth, check)] = Range(optimal, range.upper)
        else:
            self.transposition_table_scores[(position, depth, check)] = Range(range.lower, optimal)

        return optimal

    # The following algorithm is an iterative deepening MTD-bi search
    def _search(self, position):
        self.num_of_positions = 0

        # greater depth means more positions examined
        for depth in range(1, 1000):
            lower, upper, self.depth = -MATE_UPPER, MATE_UPPER, depth
            while lower < upper - EVALULATION_ROUGHNESS:
                temp = (lower+upper+1) // 2
                score = self.bound(position, temp, depth)
                if score >= temp:
                    lower = score
                else:
                    upper = score
            score = self.bound(position, lower, depth)
            yield

    def search(self, position, time_limit):
        starting_time = time.time()
        for i in self._search(position):
            if time.time() - starting_time > time_limit:
                break
        # moves can be retrieved from the transposition table
        return self.transposition_table_moves.get(position), self.transposition_table_scores.get((position, self.depth, True)).lower

# Analyze the coordinates of the player move
def parse(i: str) -> int:
    file, rank = ord(i[0]) - ord('a'), int(i[1]) - 1
    return a1 + file - 10 * rank

# Find the coordinates of the engine move
def render(i: int) -> str:
    rank, file = divmod(i - a1, 10)
    return chr(file + ord('a')) + str(1 - rank)

# Displays board
def print_position(position):
    print()
    # The reason why we use this dictionary is to make the . (blank squares) more centered
    format_blank_squares = {'R':'R', 'N':'N', 'B':'B', 'Q':'Q', 'K':'K', 'P':'P',
                  'r':'r', 'n':'n', 'b':'b', 'q':'q', 'k':'k', 'p':'p', '.':'·'}
    for row in position.board.split():
        print(' '.join(format_blank_squares.get(p) for p in row))
    print()

def main():
    position = Position(starting_position, 0, (True,True), (True,True), 0, 0)
    searcher = Searcher()
    while True:
        print_position(position)

        # Check if the engine captured player's king
        if position.score <= -MATE_LOWER:
            print("Lionfish Victory")
            break

        # Input request continues until player enters legal move
        move = None
        while move not in position.generate_moves():
            # RegEx finds a match of coordinates from user input
            # [a-h] is an example of metacharacter
            match = re.match('([a-h][1-8])'*2, input('Your move: '))
            if match:
                # move is a tuple of int
                # the group function does not follow indices. group(0) refers to the entire match string
                move = parse(match.group(1)), parse(match.group(2))
            else:
                print("Please enter a starting and ending square as a move such as e2e4")

        # Change the position based on the user input
        position = position.move(move)

        # Switch turn to engine and display most recent move
        print_position(position.switch())

        # Check if player captured the engine's king
        if position.score <= -MATE_LOWER:
            print("You Defeated Lionfish! Congratulations!")
            break

        # Find engine move
        move, score = searcher.search(position, time_limit=2)

        if score == MATE_UPPER:
            print("Lionfish move:", render(119 - move[0]) + render(119 - move[1]))
            position = position.move(move)
            print_position(position)
            print("Checkmate! Game Over")
            break

        # Display Lionfish move, then print board
        # move is a tuple of int
        print("Lionfish move:", render(119 - move[0]) + render(119 - move[1]))
        position = position.move(move)

main()