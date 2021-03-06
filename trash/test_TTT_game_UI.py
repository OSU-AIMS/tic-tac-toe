# User interface for a Tic Tac Toe AI
# This program provides a user interface, so that a human 
# can play against a Tic Tac Toe AI. The AI is not included; 
# it is assumed to exist in a separate file "tictactoe.py". 
# To play, place this program in the same folder as 
# tictactoe.py, and then run this program. By default, the 
# human moves first. To let the AI move first, change the 
# last line of the program to call the computerVsHuman 
# function instead of the humanVsComputer function.

import tictactoe as ttt

# Asks the user to enter an integer. Asks repeatedly, until the user complies.
# Input: String to be used as a prompt.
# Output: Integer that the user entered.
def userInteger(prompt):
    while True:
        try:
            return int(raw_input(prompt))
        except ValueError:
            pass

# Requires the user to enter a valid move.
# Do not use this function with a full board. There are then no valid moves.
# Input: Board.
# Output: A move (a pair of numbers between 0 and 2).
def userMove(board):
    while True:
        row = -1
        while row < 0 or row > 2:
            row = userInteger("Which row: 0, 1, or 2? ")
        col = -1
        while col < 0 or col > 2:
            col = userInteger("Which column: 0, 1, or 2? ")
        if board[row][col] == " ":
            return [row, col]
        else:
            print("Player", board[row][col], "has already moved there.")

# Plays the user against the computer, with the user going first.
# Input: None.
# Output: None.
def humanVsComputer():
    # Initialize the game.
    board = [[" ", " " ," "], [" ", " " ," "], [" ", " " ," "]]
    turns = 0
    ttt.printBoard(board)
    # Let the user and computer alternate turns.
    while turns < 5 and not ttt.hasWon(board, "X") and not ttt.hasWon(board, "O"):
        m = userMove(board)
        board[m[0]][m[1]] = "O"
        print
        ttt.printBoard(board)
        turns = turns + 1
        if turns < 5 and not ttt.hasWon(board, "X") and not ttt.hasWon(board, "O"):
            m = ttt.move(board, "X")
            board[m[0]][m[1]] = "X"
            print
            ttt.printBoard(board)
    # Finish the game.
    if ttt.hasWon(board, "O"):
        print("O wins.")
    elif ttt.hasWon(board, "X"):
        print("X wins.")
    else:
        print("Draw.")

# Plays the user against the computer, with the computer going first.
# Input: None.
# Output: None.
def computerVsHuman():
    # Initialize the game.
    board = [[" ", " " ," "], [" ", " " ," "], [" ", " " ," "]]
    turns = 0
    ttt.printBoard(board)
    # Let the computer and user alternate turns.
    while turns < 5 and not ttt.hasWon(board, "X") and not ttt.hasWon(board, "O"):
        m = ttt.move(board, "O")
        board[m[0]][m[1]] = "O"
        print
        ttt.printBoard(board)
        turns = turns + 1
        if turns < 5 and not ttt.hasWon(board, "X") and not ttt.hasWon(board, "O"):
            m = userMove(board)
            board[m[0]][m[1]] = "X"
            print
            ttt.printBoard(board)
    # Finish the game.
    if ttt.hasWon(board, "O"):
        print("O wins.")
    elif ttt.hasWon(board, "X"):
        print("X wins.")
    else:
        print("Draw.")

# If the user ran this file directly, then this code will be executed.
# If the user imported this file, then this code will not be executed.
if __name__ == "__main__":
    #humanVsComputer()
    computerVsHuman()


