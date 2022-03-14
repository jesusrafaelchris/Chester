import chess
import chess.engine
import chess.svg


print(chess.svg.piece(chess.Piece.from_symbol("R")))
engine = chess.engine.SimpleEngine.popen_uci("/Users/christiangrinling/Desktop/Keras/stockfish-10-mac/Mac/stockfish-10-64")
print(engine.id.get("name"))


board = chess.Board("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1")
print(board)
limit = chess.engine.Limit(time=2.0)
print(engine.play(board, limit))

engine.quit()
