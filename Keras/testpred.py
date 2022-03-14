from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io
import chess
import chess.engine

def board_to_fen(board):
    # Use StringIO to build string more efficiently than concatenating
    with io.StringIO() as s:
        for row in board:
            empty = 0
            for cell in row:
                c = cell[0]
                if c in ('r', 'g'):
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(cell[1].upper() if c == 'g' else cell[1].lower())
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        # Move one position back to overwrite last '/'
        s.seek(s.tell() - 1)
        # If you do not have the additional information choose what to put
        s.write(' w KQkq - 0 1')
        return s.getvalue()

img_width, img_height = 70,70
model = load_model('/Users/christiangrinling/Desktop/Keras/model9.h5') # Load the model


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
global result1
result1 = [] #save the result in an array


for i in range (0,64):
    img_path = '/Users/christiangrinling/Desktop/Keras/extracted/img' + (str(i)) + '.jpg'

    test_image= image.load_img(img_path, target_size = (img_width, img_height, 3))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    images = np.vstack([test_image])
    result = model.predict_classes(images)
    result1 = np.append(result1, result)
    result1 = result1.astype(int)
    #result2 = model.predict_proba(images)
    #print(np.around(result2,3))
print(result1) # prints array of the predictions - result1.shape should be 64
board = np.zeros(shape = (8,8), dtype='object') # makes an 8x8 empty array for the chessboard

#'train_empty_dir': 0,
#'train_green_bishop_dir': 1,
#'train_green_castle_dir': 2,
#'train_green_king_dir': 3,
#'train_green_knight_dir': 4,
#'train_green_pawn_dir': 5,
#'train_green_queen_dir': 6,

#'train_red_bishop_dir': 7,
#'train_red_castle_dir': 8,
#'train_red_king_dir ': 9,
#'train_red_knight_dir': 10,
#'train_red_pawn_dir': 11,
#'train_red_queen_dir': 12

for i in range(0,8):
    for j in range(0,8):
            board[i][j] = result1[i*8+j]
            if board[i][j] == 0:
                board[i][j] = 'em' #empty
            elif board[i][j] == 1:
                board[i][j] = 'gb' #bishop
            elif board[i][j] == 2:
                board[i][j] = 'gr' #castle
            elif board[i][j] == 3:
                board[i][j] = 'gk' #king
            elif board[i][j] == 4:
                board[i][j] = 'gn' #knight
            elif board[i][j] == 5:
                board[i][j] = 'gp' #pawn
            elif board[i][j] == 6:
                board[i][j] = 'gq' #queen
            elif board[i][j] == 7:
                board[i][j] = 'rb' #bishop
            elif board[i][j] == 8:
                board[i][j] = 'rr' #castle
            elif board[i][j] == 9:
                board[i][j] = 'rk' #king
            elif board[i][j] == 10:
                board[i][j] = 'rn' #knight
            elif board[i][j] == 11:
                board[i][j] = 'rp' #pawn
            elif board[i][j] == 12:
                board[i][j] = 'rq' #queen


print(board_to_fen(board)) #prints the chessboard with values of predicitions in array
print(board)
y = board_to_fen(board)

#e2e4

#need to write code to move piece from array position , replace that position with empty, then replace
#the end position with the piece if the piece =! empty ot just set end position as piece if empty

#also need code to make sure it knows whos turn it is from python_chess module

board1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
board = chess.Board(y)
engine = chess.engine.SimpleEngine.popen_uci("/Users/christiangrinling/Desktop/Keras/stockfish-10-mac/Mac/stockfish-10-64")
print(engine.id.get("name"))
limit = chess.engine.Limit(time=2.0,depth = 5)
result = engine.play(board, limit)
print("Best move is", result.move)
print("I think you're going to do",result.ponder)
#board.push(result.move)
move = (result.move)
print(move)

engine.quit()

print(board)


#train reinforcement learning neural net to pick stuff up
