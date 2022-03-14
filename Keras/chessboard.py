import numpy as np
chessboard = np.zeros(shape = (7,8), dtype = 'int')
chessboard = np.insert(chessboard, 1,[1], axis = 0) + np.insert(chessboard, 6,[1], axis = 0)

"""
    8 = Castle
    7 = Knight
    6 = Bishop
    5 = King
    4 = Queen
    1 = Pawn
    0 = Empty
"""
#White
chessboard[0][0] = 8
chessboard[0][7] = 8
chessboard[0][1] = 7
chessboard[0][6] = 7
chessboard[0][2] = 6
chessboard[0][5] = 6
chessboard[0][3] = 5
chessboard[0][4] = 4

#Black
chessboard[7][0] = 8
chessboard[7][7] = 8
chessboard[7][1] = 7
chessboard[7][6] = 7
chessboard[7][2] = 6
chessboard[7][5] = 6
chessboard[7][3] = 5
chessboard[7][4] = 4

print(chessboard)

x1,y1 = 800,800
x2,y2 = 700,700
x3,y3 = 600,600
x4,y4 = 500,500
x5,y5 = 400,400
x6,y6 = 300,300
x7,y7 = 200,200
x8,y8 = 100,100

column = np.zeros(shape = (8,8), dtype = 'int')

if x1 == 800 and y1 == 800:
    column[0][0] = result

if x2 == 700 and y2 == 700:
    column[1][1] = result

if x3 == 600 and y3 == 600:
    column[2][2] = result

if x4 == 500 and y4 == 500:
    column[3][3] = result

if x5 == 400 and y5 == 400:
    column[4][4] = result

if x6 == 300 and y6 == 300:
    column[5][5] = result

if x7 == 200 and y7 == 200:
    column[6][6] = result

if x8 == 100 and y8 == 100:
    column[7][7] = result

print(column)

coordinates = []
