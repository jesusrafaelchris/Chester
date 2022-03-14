import io

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
                    s.write(cell[1].upper() if c == 'r' else cell[1].lower())
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

print(board_to_fen(board))
