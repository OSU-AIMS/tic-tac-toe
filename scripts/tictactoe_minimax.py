#Script from https://github.com/aschmid1/TicTacToeAI/blob/master/tictactoe.py
import random
random.seed()

# Prints the board in a pretty format.
# Input: Board.
# Output: None.
def printBoard(b):
    for r in range(3):
        print("[", b[r][0], b[r][1], b[r][2], "]")

# Creates a copy of the given board.
# Input: Board.
# Output: A new board with the same values as board.
def cloneBoard(board):
    newboard = []
    for row in board:
        newboard.append(list(row))
    return newboard

# Returns the other player.
# Input: Player "O" or "X".
# Output: Other player.
def otherPlayer(p):
    if p == "O":
        return "X"
    else:
        return "O"

# Detects whether a given player has won the game.
# Input: Board. Player "O" or "X".
# Output: Boolean.
def hasWon(b, p):
    for line in getWinningLines():
        cell0 = b[ line[0][0] ][ line[0][1] ]
        cell1 = b[ line[1][0] ][ line[1][1] ]
        cell2 = b[ line[2][0] ][ line[2][1] ]
        
        if cell0 == p and cell1 == p and cell2 == p:
            return True
    # no win conditions found for player
    return False

# Returns a list of all 8 possible winning lines for standard tictactoe.
# Output: a list of 8 lines; a line is a list of cell positions
def getWinningLines():
    lines = []
    # build list of horizontal lines
    for row in range(3):
        line = []
        for col in range(3):
            line.append([row, col])
        lines.append(line)
    # build list of vertical lines
    for col in range(3):
        line = []
        for row in range(3):
            line.append([row, col])
        lines.append(line)
    # add diagonal lines
    lines.append([[0, 0], [1, 1], [2, 2]])
    lines.append([[0, 2], [1, 1], [2, 0]])
    
    return lines

# Evaluates a score based on the layout of a line
# Input: Board. Player "O" or "X".
#        Line represented by list of cell positions
# Output: Score
def getLineScore(b, p, line):
    """Scoring scheme for a board layout.
    
    Player has 3 cells in line:           100 points
    Player has 2 unblocked cells in line:  10 points
    Player has 1 unblocked cell in line:    1 point
    All other cases:                        0 points"""
    cell0 = b[ line[0][0] ][ line[0][1] ]
    cell1 = b[ line[1][0] ][ line[1][1] ]
    cell2 = b[ line[2][0] ][ line[2][1] ]
    
    # default score if line is empty or blocked by opponent
    lineScore = 0
    
    # Player has 3 cells in line
    if cell0 == p and cell1 == p and cell2 == p:
        lineScore = 100
    # Player has 2 unblocked cells in line
    elif (cell0 == p and cell1 == p and cell2 == " ") or (
        cell0 == p and cell1 == " " and cell2 == p) or (
        cell0 == " " and cell1 == p and cell2 == p):
        lineScore = 10
    # Player has 1 unblocked cell in line
    elif (cell0 == p and cell1 == " " and cell2 == " ") or (
        cell0 == " " and cell1 == p and cell2 == " ") or (
        cell0 == " " and cell1 == " " and cell2 == p):
        lineScore = 1
    
    return lineScore

# Computes the player's score from the board layout.
# Input: Board. Player "O" or "X".
# Output: Score.
def getBoardScore(b, p):
    score = 0
    
    for line in getWinningLines():
        score += getLineScore(b, p, line)
    
    # Old simple scoring scheme
    #if hasWon(p):
    #    score = 1
    
    return score

# Returns a list of valid moves for the given board.
# It is possible to improve this by removing symmetric moves
# Input: Board.
# Output: List of [row, column] pairs of 0, 1, or 2, indicating possible
#         spaces in which to move, or an empty list if the game is over.
def getPossibleMoves(b):
    if hasWon(b, "O") or hasWon(b, "X"):
        return []
    
    moves = []
    for row in range(3):
        for col in range(3):
            if b[row][col] == " ":
                moves.append([row, col])
    
    return moves

# Recursive search function to determine best possible move
# This function should not be called with a full board.
# Input: Board. Player "O" or "X". Maximum search depth (optional)
# Output: (move, score) as a tuple
def negamax_move(b, p, alpha, beta, depth = 8):
    """Move function using Negamax search with alpha beta pruning.
    
    alpha is the best maximizing score; beta is the best minimizing score
    A node's score is the max of it's children scores after negation.
    [The outcome of the scoring scheme is similar to scoring with 1 and 0]"""
    # Handle call where board is full
    if not getPossibleMoves(b):
        print("Invalid call to negamax_move with full board")
        return -1, 0 # originally None, 0
    
    best_move = []
    
    for move in getPossibleMoves(b):
        if best_move == []:
            best_move = move
        # create next board with move applied to it
        mr, mc = move[0], move[1]
        next_b = cloneBoard(b)
        next_b[mr][mc] = p
        
        # Handle final move of the game; leaf node
        if not getPossibleMoves(next_b) or depth == 0:
            score = getBoardScore(next_b, p) - getBoardScore(next_b, otherPlayer(p))
            return move, score
        
        # Choose move with lowest score for opponent 
        # by negating their highest scores and taking maximum
        # m is a placeholder for unpacking the returned tuple
        m, score = negamax_move(next_b, otherPlayer(p), -beta, -alpha, depth-1)
        score = -score
        if score > alpha:
            alpha = score
            best_move = move
            if beta < alpha:
                break;
        
    return best_move, alpha

# Selects the next move for the given player.
# Input: Board. Player "O" or "X".
# Output: Pair of 0, 1, or 2, indicating space in which to move, or None.
def competent_move(b, p):
    """Placeholder move function that only handles basic competence."""
    moves = getPossibleMoves(b)
    for move in moves:
        # determine next state
        next_board = cloneBoard(b)
        next_board[move[0]][move[1]] = p
        # Competence Condition 1: select a winning move if available
        if hasWon(next_board, p):
            return move
        # Competence Condition 2: select a non-losing move if available
        else:
            # test if opponent selecting move would win
            next_board[move[0]][move[1]] = otherPlayer(p)
            if hasWon(next_board, otherPlayer(p)):
                return move
    # No obvious moves exist
    # Select a random move if board isn't full
    if moves:
        rand = random.randint(0, len(moves)-1)
        return moves[rand]
    else:
        return -1 # originally None

# Selects the next move for the given player.
# Input: Board. Player "O" or "X".
# Output: Pair of 0, 1, or 2, indicating space in which to move, or None.
def move(b, p):
    # Basic competence
    #return competent_move(b, p)
    
    # If AI goes first then the first two moves will be by lookup table
    # The AI doesn't handle this part very well
    if b == [[" ", " " ," "], [" ", " " ," "], [" ", " " ," "]]:
        return [0, 0]
    elif b == [[p, " " ," "], [" ", otherPlayer(p) ," "], [" ", " " ," "]]:
        return [2, 2]
    
    # Negamax search with alpha beta pruning
    # alpha must be less than lowest possible score
    # beta must be greater than highest possible score
    # s holds the score of the move (not read)
    m, s = negamax_move(b, p, -1000, +1000)
    return m

# Plays the computer against itself.
# Input: None.
# Output: None.
def computerVsComputer():
    # Initialize the game.
    board = [[" ", " " ," "], [" ", " " ," "], [" ", " " ," "]]
    player = "O"
    turns = 0
    printBoard(board)
    # Run the game.
    while turns < 9 and not hasWon(board, "X") and not hasWon(board, "O"):
        m = move(board, player)
        if m == None or board[m[0]][m[1]] != " ":
            print("Invalid move", m, "by player", player, "on this board:")
            printBoard(board)
            return
        board[m[0]][m[1]] = player
        print
        printBoard(board)
        player = otherPlayer(player)
        turns = turns + 1
    # Finish the game.
    if hasWon(board, "O"):
        print("O wins.")
    elif hasWon(board, "X"):
        print("X wins.")
    else:
        print("Draw.")

# If the user ran this file directly, then this code will be executed.
# If the user imported this file, then this code will not be executed.
if __name__ == "__main__":
    computerVsComputer()


