"""
A library to play the Duffer board game (<http://www.di.fc.ul.pt/~jpn/gv/duffer.htm>).
"""

from __future__ import annotations

__author__ = "Alberto Arias"
__email__ = "alberto.ariasdrake@gmail.com"
__version__ = "0.0.0"

from typing import Callable, Dict, Generic, Iterable, Iterator, List, Optional, SupportsInt, Tuple, TypeVar, Union
import copy
import logging

logging.basicConfig(format='%(asctime)s %(message)s')
LOGGER = logging.getLogger('DufferLogger')
LOGGER.setLevel('NOTSET')

SIDE = 4
DIRECTIONS = [STAY, NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST] = range(9)
DIRECTIONS = [STAY, NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST] = range(9)
DIRECTIONS_FLIP_VERTICAL = [STAY, SOUTH, SOUTH_EAST, EAST, NORTH_EAST, NORTH, NORTH_WEST, WEST, SOUTH_WEST]
DIRECTIONS_FLIP_HORIZONTAL = [STAY, NORTH, NORTH_WEST, WEST, SOUTH_WEST, SOUTH, SOUTH_EAST, EAST, NORTH_EAST]
DIRECTIONS_FLIP_DIAGONAL_A1D4 = [STAY, EAST, NORTH_EAST, NORTH, NORTH_WEST, WEST, SOUTH_WEST, SOUTH, SOUTH_EAST]
DIRECTIONS_FLIP_DIAGONAL_A4D1 = [STAY, WEST, SOUTH_WEST, SOUTH, SOUTH_EAST, EAST, NORTH_EAST, NORTH, NORTH_WEST]
DIRECTIONS_ROTATE_90 = [STAY, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST, NORTH, NORTH_EAST]
DIRECTIONS_ROTATE_180 = [STAY, SOUTH, SOUTH_WEST, WEST, NORTH_WEST, NORTH, NORTH_EAST, EAST, SOUTH_EAST]
DIRECTIONS_ROTATE_270 = [STAY, WEST, NORTH_WEST, NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST]

RANK_JUMPS = [
    0,  # [0]: stay
    +2, # [1]: north
    +2, # [2]: north-east
    0,  # [3]: east
    -2, # [4]: south-east
    -2, # [5]: south
    -2, # [6]: south-west
    0,  # [7]: west
    +2, # [8]: north-west
]
FILE_JUMPS = [
    0,  # [0]: stay
    0,  # [1]: north
    +2, # [2]: north-east
    +2, # [3]: east
    +2, # [4]: south-east
    0,  # [5]: south
    -2, # [6]: south-west
    -2, # [7]: west
    -2, # [8]: north-west
]
JUMPS = {
    -2: {-2: SOUTH_WEST, 0: SOUTH, +2: SOUTH_EAST},
    0: { -2: WEST, 0: STAY, +2: EAST},
    +2: {-2: NORTH_WEST, 0: NORTH, +2: NORTH_EAST}
}
INITIAL_POSITION = 51 # Stones at a1, a2, b1 and b2

[PLAYER1, PLAYER2] = [False, True]

FILE_NAMES = ["a", "b", "c", "d"]
RANK_NAMES = ["1", "2", "3", "4"]

Square = int
SQUARES = [A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3, A4, B4, C4, D4] = range(16)
SQUARES_FLIP_VERTICAL = [A4, B4, C4, D4, A3, B3, C3, D3, A2, B2, C2, D2, A1, B1, C1, D1]
SQUARES_FLIP_HORIZONTAL = [D1, C1, B1, A1, D2, C2, B2, A2, D3, C3, B3, A3, D4, C4, B4, A4]
SQUARES_FLIP_DIAGONAL_A1D4 = [A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, C3, C4, D1, D2, D3, D4]
SQUARES_FLIP_DIAGONAL_A4D1 = [D4, D3, D2, D1, C4, C3, C2, C1, B4, B3, B2, B1, A4, A3, A2, A1]
SQUARES_ROTATE_90 = [A4, A3, A2, A1, B4, B3, B2, B1, C4, C3, C2, C1, D4, D3, D2, D1]
SQUARES_ROTATE_180 = [D4, C4, B4, A4, D3, C3, B3, A3, D2, C2, B2, A2, D1, C1, B1, A1]
SQUARES_ROTATE_270 = [D1, D2, D3, D4, C1, C2, C3, C4, B1, B2, B3, B4, A1, A2, A3, A4]
SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]

def is_square(square: Square) -> bool:
    return square in SQUARES

def parse_square(name: str) -> Square:
    """
    Gets the square index for the given square *name*
    (e.g., ``a1`` returns ``3``).
    :raises: :exc:`ValueError` if the square name is invalid.
    """
    return SQUARE_NAMES.index(name)

def square_name(square: Square) -> str:
    """Gets the name of the square, like ``a3``, given the square index."""
    return SQUARE_NAMES[square]

def square(file: int, rank: int) -> Square:
    """Gets a square number by file and rank index."""
    if (rank < 0 or rank >= 4 or file < 0 or file >= 4):
        raise ValueError("Invalid square: out of bounds.")
    return rank * SIDE + file

def square_file(square: Square) -> int:
    """
    Gets the file index of the square where ``0`` is the d-file,
    and ``3`` is the a-file.
    """
    return square % SIDE

def square_rank(square: Square) -> int:
    """
    Gets the rank index of the square where ``0`` is the first rank,
    and ``3`` is the fourth rank.
    """
    return square // SIDE

def squared_distance(a: Square, b: Square) -> int:
    return abs(square_file(a) - square_file(b))**2 + abs(square_rank(a) - square_rank(b))**2

Bitboard = int

BB_EMPTY = 0
BB_ALL = 0xffff

BB_SQUARES = [
    BB_A1, BB_B1, BB_C1, BB_D1,
    BB_A2, BB_B2, BB_C2, BB_D2,
    BB_A3, BB_B3, BB_C3, BB_D3,
    BB_A4, BB_B4, BB_C4, BB_D4,
] = [1 << sq for sq in SQUARES]

BB_FILES = [
    BB_FILE_A,
    BB_FILE_B,
    BB_FILE_C,
    BB_FILE_D,
] = [0x1111 << (1 + i) for i in range(4)]

BB_RANKS = [
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
] = [0xf << (4 * i) for i in range(4)]

def scan_forward(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r

class Move:
    """
    Represents a move between squares.
    """

    def __init__(self, path):
        self._path = path
        
    def __str__(self):
        return '->'.join([square_name(sq) for sq in self.path])

    def __repr__(self):
        return '"' + self.__str__() + '"'

    def __eq__(self, other):
        return all([sp == op for sp, op in zip(self.path, other.path)])
        # return self.path == other.path
    
    def __ne__(self, other):
        return any([sp != op for sp, op in zip(self.path, other.path)])
        # return self.path != other.path

    def __hash__(self):
        return hash(self.__str__)

    @property
    def source(self) -> Square:
        return self.path[0]

    @property
    def destinations(self) -> List[Square]:
        return self.path[1:]

    @property
    def final_destination(self) -> Square:
        return self.path[-1]

    @property
    def jumps(self) -> List[Tuple[Square]]:
        return zip(self.path, self.path[1:])

    def flip_vertical(self) -> Move:
        jumps = [0] * 3
        source, jumps[0], jumps[1], jumps[2] = self.to_id()
        source = SQUARES_FLIP_VERTICAL[source]
        jumps = [DIRECTIONS_FLIP_VERTICAL[j] for j in jumps]
        return Move.from_id((source, jumps[0], jumps[1], jumps[2]))

    def flip_horizontal(self) -> Move:
        jumps = [0] * 3
        source, jumps[0], jumps[1], jumps[2] = self.to_id()
        source = SQUARES_FLIP_HORIZONTAL[source]
        jumps = [DIRECTIONS_FLIP_HORIZONTAL[j] for j in jumps]
        return Move.from_id((source, jumps[0], jumps[1], jumps[2]))

    def flip_diagA1D4(self) -> Move:
        jumps = [0] * 3
        source, jumps[0], jumps[1], jumps[2] = self.to_id()
        source = SQUARES_FLIP_DIAGONAL_A1D4[source]
        jumps = [DIRECTIONS_FLIP_DIAGONAL_A1D4[j] for j in jumps]
        return Move.from_id((source, jumps[0], jumps[1], jumps[2]))

    def flip_diagA4D1(self) -> Move:
        jumps = [0] * 3
        source, jumps[0], jumps[1], jumps[2] = self.to_id()
        source = SQUARES_FLIP_DIAGONAL_A4D1[source]
        jumps = [DIRECTIONS_FLIP_DIAGONAL_A4D1[j] for j in jumps]
        return Move.from_id((source, jumps[0], jumps[1], jumps[2]))

    def rotate90(self) -> Move:
        jumps = [0] * 3
        source, jumps[0], jumps[1], jumps[2] = self.to_id()
        source = SQUARES_ROTATE_90[source]
        jumps = [DIRECTIONS_ROTATE_90[j] for j in jumps]
        return Move.from_id((source, jumps[0], jumps[1], jumps[2]))

    def rotate180(self) -> Move:
        jumps = [0] * 3
        source, jumps[0], jumps[1], jumps[2] = self.to_id()
        source = SQUARES_ROTATE_180[source]
        jumps = [DIRECTIONS_ROTATE_180[j] for j in jumps]
        return Move.from_id((source, jumps[0], jumps[1], jumps[2]))

    def rotate270(self) -> Move:
        jumps = [0] * 3
        source, jumps[0], jumps[1], jumps[2] = self.to_id()
        source = SQUARES_ROTATE_270[source]
        jumps = [DIRECTIONS_ROTATE_270[j] for j in jumps]
        return Move.from_id((source, jumps[0], jumps[1], jumps[2]))

    def contains(self, square:Square) -> bool:
        return (square in self.path)

    def to_id(self) -> Tuple:
        source = self.path[0]
        previous_rank, previous_file = square_rank(source), square_file(source)
        jumps = [0] * 3
        i = 0
        for sq in self.path[1:]:
            current_rank, current_file = square_rank(sq), square_file(sq)
            vd, hd = current_rank - previous_rank, current_file - previous_file
            previous_rank, previous_file = current_rank, current_file
            jumps[i] = JUMPS[vd][hd]
            i += 1
        return (source, jumps[0], jumps[1], jumps[2])

    @property
    def path(self):
        return self._path

    def has_legal_jumps(self) -> bool:
        return all([self._is_legal_jump(src, dst) for src, dst in self.jumps])

    @classmethod
    def _is_legal_jump(cls, source, destination) -> bool:
        sd = squared_distance(source, destination)
        return sd == 4 or sd == 8
    
    @classmethod
    def from_id(cls, id: Tuple) -> Optional[Move]:
        jumps = [STAY] * 3
        source, jumps[0], jumps[1], jumps[2] = id
        if (source not in SQUARES):
            raise ValueError("Illegal move: initial square out of range")
        rank = square_rank(source)
        file = square_file(source)
        path = [source]
        for j in range(0, 3):
            if (jumps[j] > 0):
                rank += RANK_JUMPS[jumps[j]]
                file += FILE_JUMPS[jumps[j]]
                try:
                    path.append(square(file, rank))
                except:
                    raise ValueError("Illegal move: invalid jump.")
        return cls.from_path(path)

    @classmethod
    def from_path(cls, path) -> Optional[Move]:
        if (len(path) < 2):
            raise ValueError("Move needs at least two squares.")
        for sq in path:
            if (not is_square(sq)):
                raise TypeError("Move is not made of squares.")
        return cls(path)

    @classmethod
    def from_string(cls, name) -> Optional[Move]:
        try:
            path = [parse_square(token) for token in name.split("->")]
        except:
            raise ValueError("Illegal move: invalid squares.")
        return cls(path)

class BaseBoard:
    """
    A board representing the position of the stones. See
    :class:`~duffer.Board` for a full board with move generation.

    The board is initialized with stones at a1, a2, b1 and b2,
    unless otherwise specified in the optional *position* argument.
    """

    def __init__(self, position: Bitboard = INITIAL_POSITION) -> None:
        self.position = position

    def __repr__(self) -> str:
        return "BaseBoard(position=%04x)" % (self.position)

    def __str__(self) -> str:
        ret = ''
        for r in range(SIDE):
            rank = self.position & BB_RANKS[r]
            rank = rank >> SIDE * r
            ret = '\n' + ret
            rank_str = ''
            for f in range(SIDE):
                rank_str += 'O ' if (rank % 2) else '· '
                rank = rank >> 1
            ret = str(r + 1) + ' | ' + rank_str + ret
        ret += '    -------\n'
        ret += '    ' + ' '.join([chr(100 - f) for f in range(SIDE - 1, -1, -1)])
        return ret

    @classmethod
    def from_array(cls, board) -> Optional[BaseBoard]:
        # Assuming ``board`` a SIDExSIDE array, as a List of Lists
        # TODO: verify this assumption with a sanity check
        position = [int(board[rank][file])*2**(SIDE*rank + file) for file in range(SIDE) for rank in range(SIDE)]
        return cls(sum(position))

    @classmethod
    def from_string_representation(cls, s) -> Optional[BaseBoard]:
        return cls(int(s, base=16))

    def string_representation(self) -> str:
        return ("%04x" % self.position)

    def array_representation(self) -> List[List[int]]:
        b = [[0] * SIDE for i in range(SIDE)]
        for st in self.stones():
            b[square_rank(st)][square_file(st)] = 1
        return b

    def reset(self) -> None:
        self.position = INITIAL_POSITION

    def clear(self) -> None:
        self.position = 0

    def set_stone_at(self, square: Square) -> bool:
        """
        Places a stone at the given square. Returns ``True`` if the square was empty,
        or ``False`` if it was already occupied. 
        """
        mask = BB_SQUARES[square]
        placed = self.is_empty_at(square)
        self.position |= mask
        return placed

    def remove_stone_at(self, square: Square) -> bool:
        """
        Removes a stone from the given square. Returns ``True`` if the square was occupied,
        or ``False`` is it was already empty.
        """
        mask = BB_ALL ^ BB_SQUARES[square]
        removed = not self.is_empty_at(square)
        self.position &= mask
        return removed

    def flip_vertical(self) -> BaseBoard:
        b = self.position
        b = (
            ((b >> 12)            ) |
            ((b >> 4)  & BB_RANK_2) |
            ((b << 4)  & BB_RANK_3) |
            ((b << 12) & BB_RANK_4)
        )
        return BaseBoard(b)

    def flip_horizontal(self) -> BaseBoard:
        b = self.position
        k1 = 0x5555
        k2 = 0x3333
        b = ((b >> 1) & k1) | (((b & k1) << 1) % 2**16)
        b = ((b >> 2) & k2) | (((b & k2) << 2) % 2**16)
        return BaseBoard(b)

    def flip_diagA1D4(self) -> BaseBoard:
        b = self.position
        k1 = 0x5050
        k2 = 0x3300
        c =  k2 & (b ^ ((b << 6) % 2**16))
        b ^=       c ^  (c >> 6)
        c =  k1 & (b ^ ((b << 3) % 2**16))
        b ^=       c ^  (c >> 3)
        return BaseBoard(b)

    def flip_diagA4D1(self) -> BaseBoard:
        b = self.position
        k1 = 0xa0a0
        k2 = 0xcc00
        c =  k2 & (b ^ ((b << 10) % 2**16))
        b ^=       c ^  (c >> 10)
        c =  k1 & (b ^ ((b << 5) % 2**16))
        b ^=       c ^  (c >> 5)
        return BaseBoard(b)

    def rotate90(self) -> BaseBoard:
        return self.flip_diagA1D4().flip_vertical()

    def rotate180(self) -> BaseBoard:
        return self.flip_vertical().flip_horizontal()

    def rotate270(self) -> BaseBoard:
        return self.flip_vertical().flip_diagA1D4()

    def is_empty_at(self, square: Square) -> bool:
        mask = BB_SQUARES[square]
        return not bool(self.position & mask)

    def is_empty_between(self, src: Square, dst: Square) -> bool:
        sd = squared_distance(src, dst)
        if (sd != 4 and sd != 8):
            raise ValueError("Invalid jump from source to destination.")
        between_rank = (square_rank(dst) + square_rank(src)) // 2
        between_file = (square_file(dst) + square_file(src)) // 2
        return self.is_empty_at(square(between_file, between_rank))

    def stones(self) -> List[Square]:
        return list(scan_forward(self.position))
            
BoardT = TypeVar("BoardT", bound="Board")

class _BoardState(Generic[BoardT]):

    def __init__(self, board: BoardT) -> None:
        self.position = board.position
        self.turn = board.turn
        self.fullmove_number = board.fullmove_number

    def restore(self, board: BoardT) -> None:
        board.position = self.position
        board.turn = self.turn
        board.fullmove_number = board.fullmove_number

class Board(BaseBoard):
    """
    A :class:`~duffer.BaseBoard` and a :data:`move stack <duffer.Board.move_stack>`.
    """

    def __init__(self: BoardT, position: Bitboard = INITIAL_POSITION, ply: int = 0) -> None:
        self.position = position
        self.move_stack: List[Move] = []
        self._stack: List[_BoardState[BoardT]] = []
        self.root_fullmove_number = ply // 2
        self.fullmove_number = self.root_fullmove_number
        self.root_turn = bool(ply % 2)
        self.turn = self.root_turn

    def reset(self) -> None:
        """Restores the starting position."""
        self.root_fullmove_number = 0
        self.fullmove_number = 0
        self.root_turn = PLAYER1
        self.turn = PLAYER1
        self.reset_board()

    def reset_board(self) -> None:
        """
        Resets only stones to the starting position. Use
        :func:`~duffer.Board.reset()` to fully restore the starting position
        (including turn and move number).
        """
        super().reset()
        self.clear_stack()

    def clear_stack(self) -> None:
        """Clears the move stack."""
        self.move_stack.clear()
        self._stack.clear()

    def _board_state(self: BoardT) -> _BoardState[BoardT]:
        return _BoardState(self)

    def is_legal(self, move: Move) -> bool:
        if not move:
            return False
        if self.is_empty_at(move.source):
            return False
        if any([not self.is_empty_at(sq) for sq in move.destinations]):
            return False
        if not move.has_legal_jumps():
            return False
        return all([not self.is_empty_between(i, j) for i, j in move.jumps])

    def is_over(self) -> bool:
        return len(list(self.legal_moves())) == 0

    def winner(self) -> Optional[bool]:
        return not self.turn if self.is_over() else None

    def generate_legal_moves(self) -> Move:
        LOGGER.debug("BEGIN - Getting legal moves")
        for source in self.stones():
            LOGGER.debug("* with stone %s", square_name(source))
            path = [source]
            index = [NORTH] * 3
            jump = 1
            while (jump > 0):
                while (index[jump-1] < len(DIRECTIONS)):
                    while (jump < 4):
                        LOGGER.debug("[i] jump=%d, index=%s, path=%s", jump, index, path)
                        dst_rank = square_rank(path[jump-1]) + RANK_JUMPS[index[jump-1]]
                        dst_file = square_file(path[jump-1]) + FILE_JUMPS[index[jump-1]]
                        try:
                            sq = square(dst_file, dst_rank)
                            LOGGER.debug("... considering %s -> %s", '->'.join([square_name(p) for p in path]), square_name(sq))
                            if (sq not in path):
                                move = Move.from_path(path + [sq])
                                if (self.is_legal(move)):
                                    path.append(sq)
                                    LOGGER.debug("+ appending %s", move)
                                    yield move
                                    jump += 1
                                else:
                                    break
                            else:
                                break
                        except:
                            break
                    if jump == 4:
                        jump -= 1
                        del path[-1]
                    index[jump-1] += 1
                del path[-1]
                index[jump-1:] = [NORTH] * (4 - jump)
                jump -= 1
                if (jump > 0):
                    index[jump-1] += 1
        LOGGER.debug("END - Getting legal moves")

    def legal_moves(self) -> LegalMoveGenerator:
        return LegalMoveGenerator(self)

    def push(self, move: Move) -> bool:
        if not self.is_legal(move):
            return False
        board_state = self._board_state()
        self.move_stack.append(move)
        self._stack.append(board_state)

        for sq in move.destinations:
            self.set_stone_at(sq)
        self.turn = not self.turn
        self.fullmove_number += 1 if not self.turn else 0
        return True

    def pop(self) -> Optional[Move]:
        move = self.move_stack.pop()
        self._stack.pop().restore(self)
        return move

    def peek(self) -> Optional[Move]:
        return self.move_stack[-1] if len(self.move_stack) > 0 else None

    def history(self) -> str:
        moves = [] if not self.root_turn else ["..."]
        moves += [str(m) for m in self.move_stack]
        moves += [""] if self.turn else []
        move_numbers = [self.root_fullmove_number + 1 + n for n in range((len(self.move_stack) + int(self.root_turn) + 1) // 2)]
        ret = ["%2d. %s\n    %s" % (n, moves[2*i], moves[2*i + 1]) for i, n in enumerate(move_numbers)]
        return '\n'.join(ret)

class LegalMoveGenerator:

    def __init__(self, board: Board) -> None:
        self.board = board

    def __bool__(self) -> bool:
        return any(self.board.generate_legal_moves())

    def count(self) -> int:
        # List conversion is faster than iterating.
        return len(list(self))

    def __iter__(self) -> Iterator[Move]:
        return self.board.generate_legal_moves()

    def __contains__(self, move: Move) -> bool:
        return self.board.is_legal(move)

    def __repr__(self) -> str:
        sans = ", ".join(str(move) for move in self)
        return f"<LegalMoveGenerator at {id(self):#x} ({sans})>"