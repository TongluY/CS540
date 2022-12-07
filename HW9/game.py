import copy
import random
import time
import numpy as np


class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase = True  # TODO: detect drop phase
        b, r = 0, 0
        for i in state:
            b += i.count('b')
            r += i.count('r')
        if b >= 4 and r >= 4:
            drop_phase = False

        if not drop_phase:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            move = []
            _, bstate = self.max_value(state, 0)
            arr1 = np.array(state) == np.array(bstate)
            # check difference between succ and curr state
            arr2 = np.where(arr1 == False)
            if state[arr2[0][0]][arr2[1][0]] == ' ':  # find original to define move
                (nr, nc) = (arr2[0][1], arr2[1][1])
                (row, col) = (arr2[0][0], arr2[1][0])
            else:
                (nr, nc) = (arr2[0][0], arr2[1][0])
                (row, col) = (arr2[0][1], arr2[1][1])
            move.insert(0, (int(row), int(col)))
            move.insert(1, (int(nr), int(nc)))  # move for after drop phase
            # print(move)
            # print(len(move))
            # print(isinstance(move[0],tuple))
            # print(len(move[0]))
            # print(isinstance(move[0][0], int))
            return move

        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better
        move = []
        _, mstate = self.max_value(state, 0)
        # print(state)
        # print(mstate)
        arr1 = np.array(state) == np.array(mstate)
        # check difference between succ and curr state
        arr2 = np.where(arr1 == False)
        # print(arr1)
        # print(arr2)
        (row, col) = (arr2[0][0], arr2[1][0])
        # (row, col) = (random.randint(0, 4), random.randint(0, 4))
        while not state[row][col] == ' ':
            # (row, col) = (random.randint(0, 4), random.randint(0, 4))
            (row, col) = (arr2[0][0], arr2[1][0])

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (int(row), int(col)))
        # print(move)
        # print(len(move))
        # print(isinstance(move[0],tuple))
        # print(len(move[0]))
        # print(isinstance(move[0][0], int))
        return move

    def succ(self, state, piece):
        """ Takes in a board state and returns a list of the legal successors.
        """
        result = []

        drop_phase = True  # TODO: detect drop phase
        b, r = 0, 0
        for i in state:
            b += i.count('b')
            r += i.count('r')
        if b >= 4 and r >= 4:
            drop_phase = False

        # adding a new piece of the current player's type to the board
        if drop_phase:
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ':
                        succ = copy.deepcopy(state)
                        succ[row][col] = piece
                        result.append(succ)
            return result
        # moving any one of the current player's pieces to an unoccupied location on the board, adjacent to that piece
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def val(x, y):
            return x >= 0 and x <= 4 and y >= 0 and y <= 4
        for row in range(5):
            for col in range(5):
                if state[row][col] == piece:
                    for x, y in directions:
                        x1, y1 = row + x, col + y
                        if val(x1, y1) and state[x1][y1] == ' ':
                            succ = copy.deepcopy(state)
                            succ[x1][y1] = succ[row][col]
                            succ[row][col] = ' '
                            result.append(succ)
        return result

    def heuristic_game_value(self, state, piece):
        """ Evaluates non-terminal states and returns some floating-point value between 1 and -1.
        """
        if self.game_value(state) != 0:
            return self.game_value(state)
        if piece == 'b':
            my, op = 'b', 'r'
        elif piece == 'r':
            my, op = 'r', 'b'
        mymax, opmax = 0, 0
        # check horizontal points
        for row in state:
            mypt, oppt = 0, 0
            for i in range(5):
                if row[i] == my:
                    mypt += 1
                if row[i] == op:
                    oppt += 1
            if mypt > mymax:
                mymax = mypt
            if oppt > opmax:
                opmax = oppt

        # check vertical points
        for col in range(5):
            mypt, oppt = 0, 0
            for i in range(5):
                if state[i][col] == my:
                    mypt += 1
                if state[i][col] == op:
                    oppt += 1
            if mypt > mymax:
                mymax = mypt
            if oppt > opmax:
                opmax = oppt

        # check \ diagonal points
        for row in range(2):
            for col in range(2):
                mypt, oppt = 0, 0
                if state[row][col] == my:
                    mypt += 1
                if state[row + 1][col + 1] == my:
                    mypt += 1
                if state[row + 2][col + 2] == my:
                    mypt += 1
                if state[row + 3][col + 3] == my:
                    mypt += 1
                if state[row][col] == op:
                    oppt += 1
                if state[row + 1][col + 1] == op:
                    oppt += 1
                if state[row + 2][col + 2] == op:
                    oppt += 1
                if state[row + 3][col + 3] == op:
                    oppt += 1
                if mypt > mymax:
                    mymax = mypt
                if oppt > opmax:
                    opmax = oppt

        # check / diagonal points
        for row in range(3, 5):
            for col in range(2):
                mypt, oppt = 0, 0
                if state[row][col] == my:
                    mypt += 1
                if state[row - 1][col + 1] == my:
                    mypt += 1
                if state[row - 2][col + 2] == my:
                    mypt += 1
                if state[row - 3][col + 3] == my:
                    mypt += 1
                if state[row][col] == op:
                    oppt += 1
                if state[row - 1][col + 1] == op:
                    oppt += 1
                if state[row - 2][col + 2] == op:
                    oppt += 1
                if state[row - 3][col + 3] == op:
                    oppt += 1
                if mypt > mymax:
                    mymax = mypt
                if oppt > opmax:
                    opmax = oppt

        # check box points
        for row in range(4):
            for col in range(4):
                mypt, oppt = 0, 0
                if state[row][col] == my:
                    mypt += 1
                if state[row + 1][col] == my:
                    mypt += 1
                if state[row][col + 1] == my:
                    mypt += 1
                if state[row + 1][col + 1] == my:
                    mypt += 1
                if state[row][col] == op:
                    oppt += 1
                if state[row + 1][col] == op:
                    oppt += 1
                if state[row][col + 1] == op:
                    oppt += 1
                if state[row + 1][col + 1] == op:
                    oppt += 1
                if mypt > mymax:
                    mymax = mypt
                if oppt > opmax:
                    opmax = oppt

        if mymax == opmax:
            return 0.0, state
        elif mymax > opmax:
            return mymax/5, state
        else:
            return opmax/(-5), state

    def max_value(self, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state), state
        if depth >= 3:
            return self.heuristic_game_value(state, self.opp)
        else:
            r = state
            a = float('-Inf')
            for succ in self.succ(state, self.my_piece):
                # print(type(succ),succ)
                # print(type(depth),depth)
                v, s = self.min_value(succ, depth + 1)
                if v > a:
                    a, r = v, succ
            return a, r

    def min_value(self, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state), state
        if depth >= 3:
            return self.heuristic_game_value(state, self.opp)
        else:
            r = state
            b = float('Inf')
            for succ in self.succ(state, self.my_piece):
                v, s = self.max_value(succ, depth + 1)
                if v < b:
                    b, r = v, succ
            return b, r

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception(
                    'Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[row + 2][col + 2] == state[row + 3][col + 3]:
                    return 1 if state[i][col] == self.my_piece else -1

        # TODO: check / diagonal wins
        for row in range(3, 5):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row - 1][col + 1] == state[row - 2][col + 2] == state[row - 3][col + 3]:
                    return 1 if state[i][col] == self.my_piece else -1

        # TODO: check box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col] == state[row][col + 1] == state[row + 1][col + 1]:
                    return 1 if state[i][col] == self.my_piece else -1

        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            # s = time.time()
            move = ai.make_move(ai.board)
            # print(time.time()-s)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " +
                  chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move(
                        [(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            # s = time.time()
            move = ai.make_move(ai.board)
            # print(time.time() - s)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " +
                  chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
