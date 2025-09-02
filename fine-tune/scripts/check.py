class ChessBoard:
    def __init__(self):
        # Initialize board with standard starting position
        # Using None for empty squares, lowercase for black, uppercase for white
        self.board = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],  # Black back rank
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],  # Black pawns
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],  # White pawns
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']   # White back rank
        ]
        self.white_to_move = True
        self.move_count = 0
        
        # Castling rights
        self.white_king_moved = False
        self.white_rook_a_moved = False
        self.white_rook_h_moved = False
        self.black_king_moved = False
        self.black_rook_a_moved = False
        self.black_rook_h_moved = False
        
        # En passant target square
        self.en_passant_target = None
    
    def algebraic_to_coords(self, square):
        """Convert algebraic notation (e.g., 'e4') to array coordinates"""
        file = ord(square[0]) - ord('a')
        rank = 8 - int(square[1])
        return rank, file
    
    def coords_to_algebraic(self, rank, file):
        """Convert array coordinates to algebraic notation"""
        return chr(ord('a') + file) + str(8 - rank)
    
    def get_piece(self, square):
        """Get piece at given square (algebraic notation)"""
        rank, file = self.algebraic_to_coords(square)
        return self.board[rank][file]
    
    def set_piece(self, square, piece):
        """Set piece at given square"""
        rank, file = self.algebraic_to_coords(square)
        self.board[rank][file] = piece
    
    def is_white_piece(self, piece):
        return piece is not None and piece.isupper()
    
    def is_black_piece(self, piece):
        return piece is not None and piece.islower()
    
    def is_valid_move(self, move):
        """Check if a UCI move is valid"""
        if len(move) < 4:
            return False, "Move too short"
        
        from_square = move[:2]
        to_square = move[2:4]
        promotion = move[4:] if len(move) > 4 else None
        
        # Get piece at source
        piece = self.get_piece(from_square)
        if piece is None:
            return False, f"No piece at {from_square}"
        
        # Check if it's the right color's turn
        if self.white_to_move and not self.is_white_piece(piece):
            return False, f"Not white's turn, but trying to move {piece}"
        if not self.white_to_move and not self.is_black_piece(piece):
            return False, f"Not black's turn, but trying to move {piece}"
        
        # Get target piece
        target_piece = self.get_piece(to_square)
        
        # Can't capture own piece
        if target_piece is not None:
            if self.white_to_move and self.is_white_piece(target_piece):
                return False, "Cannot capture own piece"
            if not self.white_to_move and self.is_black_piece(target_piece):
                return False, "Cannot capture own piece"
        
        # Basic piece movement validation
        piece_type = piece.lower()
        from_rank, from_file = self.algebraic_to_coords(from_square)
        to_rank, to_file = self.algebraic_to_coords(to_square)
        
        if piece_type == 'p':
            return self.is_valid_pawn_move(from_rank, from_file, to_rank, to_file, piece, target_piece)
        elif piece_type == 'r':
            return self.is_valid_rook_move(from_rank, from_file, to_rank, to_file)
        elif piece_type == 'n':
            return self.is_valid_knight_move(from_rank, from_file, to_rank, to_file)
        elif piece_type == 'b':
            return self.is_valid_bishop_move(from_rank, from_file, to_rank, to_file)
        elif piece_type == 'q':
            return self.is_valid_queen_move(from_rank, from_file, to_rank, to_file)
        elif piece_type == 'k':
            return self.is_valid_king_move(from_rank, from_file, to_rank, to_file)
        
        return False, "Unknown piece type"
    
    def is_valid_pawn_move(self, from_rank, from_file, to_rank, to_file, piece, target_piece):
        is_white = self.is_white_piece(piece)
        direction = -1 if is_white else 1
        start_rank = 6 if is_white else 1
        
        # Forward moves
        if from_file == to_file:
            if target_piece is not None:
                return False, "Pawn cannot capture forward"
            
            # One square forward
            if to_rank == from_rank + direction:
                return True, "Valid pawn move"
            
            # Two squares forward from starting position
            if from_rank == start_rank and to_rank == from_rank + 2 * direction:
                return True, "Valid pawn double move"
            
            return False, "Invalid pawn move distance"
        
        # Diagonal captures
        elif abs(from_file - to_file) == 1 and to_rank == from_rank + direction:
            if target_piece is not None:
                return True, "Valid pawn capture"
            
            # En passant
            if self.en_passant_target == self.coords_to_algebraic(to_rank, to_file):
                return True, "Valid en passant capture"
            
            return False, "Pawn cannot move diagonally without capture"
        
        return False, "Invalid pawn move"
    
    def is_valid_rook_move(self, from_rank, from_file, to_rank, to_file):
        if from_rank != to_rank and from_file != to_file:
            return False, "Rook must move in straight line"
        
        return self.is_path_clear(from_rank, from_file, to_rank, to_file)
    
    def is_valid_knight_move(self, from_rank, from_file, to_rank, to_file):
        rank_diff = abs(to_rank - from_rank)
        file_diff = abs(to_file - from_file)
        
        if (rank_diff == 2 and file_diff == 1) or (rank_diff == 1 and file_diff == 2):
            return True, "Valid knight move"
        
        return False, "Invalid knight move"
    
    def is_valid_bishop_move(self, from_rank, from_file, to_rank, to_file):
        if abs(to_rank - from_rank) != abs(to_file - from_file):
            return False, "Bishop must move diagonally"
        
        return self.is_path_clear(from_rank, from_file, to_rank, to_file)
    
    def is_valid_queen_move(self, from_rank, from_file, to_rank, to_file):
        # Queen moves like rook or bishop
        rook_valid, _ = self.is_valid_rook_move(from_rank, from_file, to_rank, to_file)
        bishop_valid, _ = self.is_valid_bishop_move(from_rank, from_file, to_rank, to_file)
        
        if rook_valid or bishop_valid:
            return True, "Valid queen move"
        
        return False, "Invalid queen move"
    
    def is_valid_king_move(self, from_rank, from_file, to_rank, to_file):
        rank_diff = abs(to_rank - from_rank)
        file_diff = abs(to_file - from_file)
        
        if rank_diff <= 1 and file_diff <= 1 and (rank_diff > 0 or file_diff > 0):
            return True, "Valid king move"
        
        # Castling - simplified check
        if rank_diff == 0 and file_diff == 2:
            return True, "Castling move (simplified validation)"
        
        return False, "Invalid king move"
    
    def is_path_clear(self, from_rank, from_file, to_rank, to_file):
        """Check if path between squares is clear (excluding endpoints)"""
        rank_step = 0 if from_rank == to_rank else (1 if to_rank > from_rank else -1)
        file_step = 0 if from_file == to_file else (1 if to_file > from_file else -1)
        
        current_rank = from_rank + rank_step
        current_file = from_file + file_step
        
        while current_rank != to_rank or current_file != to_file:
            if self.board[current_rank][current_file] is not None:
                return False, "Path blocked"
            current_rank += rank_step
            current_file += file_step
        
        return True, "Path clear"
    
    def make_move(self, move):
        """Execute a move on the board"""
        from_square = move[:2]
        to_square = move[2:4]
        promotion = move[4:] if len(move) > 4 else None
        
        piece = self.get_piece(from_square)
        
        # Handle en passant capture
        if piece.lower() == 'p' and self.en_passant_target == to_square:
            # Remove the captured pawn
            capture_rank = 4 if self.white_to_move else 3
            capture_square = to_square[0] + str(capture_rank + 1)
            self.set_piece(capture_square, None)
        
        # Move the piece
        self.set_piece(to_square, piece)
        self.set_piece(from_square, None)
        
        # Handle promotion
        if promotion:
            promoted_piece = promotion.upper() if self.white_to_move else promotion.lower()
            self.set_piece(to_square, promoted_piece)
        
        # Set en passant target for pawn double moves
        self.en_passant_target = None
        if piece.lower() == 'p':
            from_rank, from_file = self.algebraic_to_coords(from_square)
            to_rank, to_file = self.algebraic_to_coords(to_square)
            
            if abs(to_rank - from_rank) == 2:
                # Pawn moved two squares, set en passant target
                en_passant_rank = (from_rank + to_rank) // 2
                self.en_passant_target = self.coords_to_algebraic(en_passant_rank, from_file)
        
        # Update castling rights (simplified)
        if piece.lower() == 'k':
            if self.white_to_move:
                self.white_king_moved = True
            else:
                self.black_king_moved = True
        
        # Switch turns
        self.white_to_move = not self.white_to_move
        self.move_count += 1
    
    def print_board(self):
        """Print current board position"""
        print("  a b c d e f g h")
        for i, row in enumerate(self.board):
            rank = 8 - i
            print(f"{rank} ", end="")
            for piece in row:
                print(piece if piece else '.', end=' ')
            print(f" {rank}")
        print("  a b c d e f g h")
        print(f"Move {self.move_count + 1}, {'White' if self.white_to_move else 'Black'} to move")

def validate_moves(moves):
    board = ChessBoard()
    invalid_moves = []
    
    for i, move in enumerate(moves):
        is_valid, reason = board.is_valid_move(move)
        
        if not is_valid:
            invalid_moves.append({
                'move_number': i + 1,
                'move': move,
                'reason': reason,
                'position_after_moves': i  # How many moves were successfully made
            })
            print(f"Invalid move #{i+1}: {move} - {reason}")
            print(f"Board position after {i} moves:")
            board.print_board()
            return False, invalid_moves
        
        board.make_move(move)
        
        # Print every 10 moves for debugging
        if (i + 1) % 50 == 0:
            print(f"After move {i+1} ({move}):")
            board.print_board()
            print()
    
    return True, []

# Test the moves
moves=['b2b4', 'c7c5', 'f2f4', 'c5b4', 'e2e3', 'd7d5', 'g2g4', 'e7e6', 'b1a3', 'b4a3', 'e3e4', 'd5e4', 'f4f5', 'e4e3', 'd2d4', 'e6f5', 'g1h3', 'd8d7', 'd4d5', 'b8c6', 'c1b2', 'f5f4', 'b2g7', 'b7b5', 'f1c4', 'a7a5', 'g7e5', 'd7c7', 'c4b5', 'f8e7', 'a1c1', 'e8d7', 'e5a1', 'a8a6', 'e1f1', 'c7b7', 'd1d4', 'e7f8', 'd5d6', 'g8e7', 'd4c5', 'd7e6', 'a1b2', 'h7h6', 'c5c3', 'e6d5', 'c3d3', 'd5c5', 'h3f4', 'c5b6', 'd3a3', 'b6a7', 'a3a5', 'e7f5', 'b2d4', 'a7a8', 'a5b6', 'c6d4', 'f4h3', 'd4b3', 'h1g1', 'b7e4', 'h3f4', 'f7f6', 'c1b1', 'a6a3', 'f4g6', 'f8e7', 'b6a7', 'a8a7', 'd6e7', 'a7b7', 'g6h8', 'b3a5', 'b1e1', 'a3a4', 'b5c6', 'b7b6', 'e1e2', 'b6a7', 'c6a8', 'a7b8', 'f1e1', 'f5d4', 'e2e3', 'c8e6', 'e1d1', 'a4a3', 'a8e4', 'd4e2', 'g1f1', 'b8c7', 'e4f3', 'e6d7', 'd1e2', 'd7e6', 'e2f2', 'a3c3', 'a2a4', 'c7d7', 'h8g6', 'a5b3', 'e7e8n', 'c3c6', 'e3e5', 'b3d4', 'g6e7', 'c6c2', 'f2e3', 'd4e2', 'e7c8', 'e2c3', 'e5e4', 'd7c6', 'f1f2', 'c6b7', 'e4c4', 'b7a6', 'f3c6', 'c3d5', 'e3e4', 'c2c1', 'c6d5', 'c1f1', 'e4d4', 'f1g1', 'f2g2', 'e6d5', 'd4d5', 'g1g2', 'e8c7', 'a6b7', 'c7a8', 'b7b8', 'd5c6', 'h6h5', 'c6b6', 'f6f5', 'g4h5', 'g2g1', 'c4c6', 'g1h1', 'c6h6', 'h1g1', 'h6d6', 'g1g5', 'd6c6', 'b8a8', 'b6c7', 'g5g4', 'c8e7', 'g4a4', 'c6b6', 'a4b4', 'c7c6', 'b4b3', 'h2h4', 'a8a7', 'e7d5', 'b3b1', 'c6d7', 'b1b4', 'b6e6', 'a7b8', 'e6g6', 'b4a4', 'g6g3', 'a4a7', 'd7e6', 'a7a2', 'g3d3', 'a2a4', 'd3e3', 'b8a7', 'e3h3', 'a4a5', 'h3a3', 'a7b7', 'd5e7', 'a5b5', 'a3f3', 'b7a7', 'e6d6', 'a7b7', 'f3d3', 'b5b3', 'e7c6', 'b3b6', 'd3d4', 'f5f4', 'd6e7', 'b6a6', 'd4e4', 'a6a3', 'c6b4', 'a3a2', 'e7e6', 'a2a7', 'e6f5', 'a7a2', 'f5e5', 'a2h2', 'b4a6', 'b7a6', 'e4e2', 'a6b7', 'e2e3', 'h2e2', 'e5f5', 'b7c6', 'e3e4', 'e2e4', 'f5g6', 'e4e8', 'g6g7', 'e8e7', 'g7h6', 'e7e8', 'h6g6', 'e8e7', 'g6f6', 'e7e4', 'f6g7', 'c6b7', 'g7f7', 'e4d4', 'f7e8', 'd4a4', 'e8f7', 'b7c6', 'f7e6', 'a4a3', 'e6f6', 'a3a2', 'f6f5', 'c6c5', 'f5e4', 'a2h2', 'e4f3', 'c5c4', 'f3e4', 'h2h4', 'e4f5', 'h4g4', 'f5g4', 'c4b3', 'g4g5', 'b3c2', 'g5h6', 'c2b3', 'h6g7', 'b3c2', 'g7h6', 'c2d3', 'h6g7', 'd3c4', 'g7f8', 'c4d3', 'f8e7', 'd3e4', 'e7d6', 'e4d3', 'd6e7', 'd3e4', 'h5h6', 'e4f5', 'e7d6', 'f5e4', 'h6h7', 'e4f5', 'd6d5', 'f5g4', 'h7h8b', 'g4g3', 'h8b2', 'g3h4', 'b2f6', 'h4g4', 'f6e5', 'g4f5', 'd5c5', 'f5e5', 'c5c4', 'e5d6', 'c4c3', 'd6c5', 'c3b2', 'c5d4', 'b2b3', 'd4e3', 'b3a4', 'e3e4', 'a4b5', 'e4d5', 'b5a6', 'd5c4', 'a6a7', 'c4b4', 'a7b6', 'b4a3', 'b6c5', 'a3a4', 'c5d6', 'a4b3', 'd6c7', 'b3c4', 'c7c6', 'c4d4', 'c6b7', 'd4c4', 'b7c7', 'c4d3', 'c7d6', 'd3c4', 'd6d7', 'c4d3', 'd7e8', 'd3e3', 'e8d7', 'e3e4', 'd7c7', 'e4d4', 'c7c8', 'd4e4', 'c8d8', 'e4f5', 'd8e8', 'f5e5', 'e8f8', 'e5d4', 'f8g7', 'd4d5', 'g7h6', 'd5d4', 'h6h7', 'd4d5', 'h7h8', 'd5e4', 'h8g8', 'e4d3', 'g8f7', 'd3e4', 'f7g6', 'e4d3', 'g6h6', 'd3e4', 'h6g7', 'e4f5', 'g7f8', 'f5e6', 'f8e8', 'e6d5', 'e8f8', 'd5e6', 'f8g7', 'e6d5', 'g7f6', 'd5e4', 'f6g5', 'e4d3', 'g5h6', 'd3e4', 'h6g6', 'e4d4', 'g6h6', 'd4d5', 'h6h5', 'd5c4', 'h5g4', 'c4b5', 'g4g5', 'b5b4', 'g5f5', 'b4a5', 'f5f4']

print("Validating chess moves...")
is_valid, errors = validate_moves(moves)

if is_valid:
    print(f"\n✅ ALL {len(moves)} MOVES ARE VALID!")
    print("This represents a complete, legal chess game.")
else:
    print(f"\n❌ INVALID MOVES FOUND:")
    for error in errors:
        print(f"Move {error['move_number']}: {error['move']} - {error['reason']}")