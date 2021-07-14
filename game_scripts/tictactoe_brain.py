import random
import sys
 
board=[i for i in range(0,9)]
player, computer = '',''
 
 
moves=((1,7,3,9),(5,),(2,4,6,8))
 
winners=((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
# Table
tab=range(1,10)
 
 
# Function to print the board
def print_board():
    x=1
    counter = 0
    for i in range(0,9):
        if board[i] == 'X' or board[i] == 'O' :
            print(board[i],end ='')
        else :
            print(" ",end = ' ')
        counter+=1
        if counter%3 ==0 :
            print("",end = '\n----------\n')
        else :
            print("| ",end='')
 
# This function will decide player will play with X or O
def select_char():
    if random.randint(0,1) == 0:
        return ('X','O')
    return ('O','X')
 
# Function to decide a valid move.
def can_move(brd, player, move):
    if move in tab and brd[move-1] == move-1:
        return True
    return False
 
# Function to check if win condition is satisfied.
def can_win(brd, player, move):
    places=[]
    x=0
    for i in brd:
        if i == player: places.append(x);
        x+=1
    win=True
    for tup in winners:
        win=True
        for ix in tup:
            if brd[ix] != player:
                win=False
                break
        if win == True:
            break
    return win
 
def make_move(brd, player, move, undo=False):
    if can_move(brd, player, move):
        brd[move-1] = player
        win=can_win(brd, player, move)
        if undo:
            brd[move-1] = move-1
        return (True, win)
    return (False, False)
 
def random_move():
    move=-1
    empty_place = []
    # If I can win, others don't matter.
    for i in range(1,10):
        if board[i-1] == i-1 :
            empty_place.append(i)
    if len(empty_place) > 0 :
        idx = random.randint(0,len(empty_place)-1)
        move = empty_place[idx]
        return make_move(board,computer,move)
 
    return (False,False)
 
# AI goes here
def computer_move():
    move=-1
    # If the computer can win in this turn, it will take the turn.
    for i in range(1,10):
        if make_move(board, computer, i, True)[1]:
            move=i
            break
    if move == -1:
        # If player can win next turn then we have to block that.
        for i in range(1,10):
            if make_move(board, player, i, True)[1]:
                move=i
                break
    if move == -1:
        # Otherwise, try to take one of the desired places.
        for tup in moves:
            for mv in tup:
                if move == -1 and can_move(board, computer, mv):
                    move=mv
                    break
    return make_move(board, computer, move)
 
def is_full():
    for i in range(0,9):
        if board[i] == i:
            return True
    return False
 
player, computer = select_char()
print('Player is [%s] and computer is [%s]' % (player, computer))
 
turn = random.randint(0,1)
 
if turn == 0 :
    print("Player will play first.")
else :
    print("Computer will play first.")
    print_board()
    computer_move()
    print()
 
 
result="It's a tie !!"
 
while is_full():
    print_board()
    print('Make your move  [1-9] : ', end='')
    move = int(input())
    moved, won = make_move(board, player, move)
    if not moved:
        print(' >> Invalid number ! Try again !')
        continue
 
    if won:
        result=' You won !!'
        break
    elif computer_move()[1]:
        result=' You lose !!'
        break;
 
print_board()
print(result)