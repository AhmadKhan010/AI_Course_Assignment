import tkinter as tk
from tkinter import messagebox, font
import copy
import time
import random

class UltimateTicTacToe:
    """Main game class for Ultimate Tic-Tac-Toe with variant rule: active board persists until won or full."""
    def __init__(self):
        # Initialize 3x3 grid of 3x3 small boards
        self.board = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
        # Track status of small boards (None: not won, 'X': X won, 'O': O won, 'D': draw)
        self.small_board_status = [[None for _ in range(3)] for _ in range(3)]
        # Current active small board (None: any board, else (row, col))
        self.active_board = None
        # Current player ('X' or 'O')
        self.current_player = 'X'
        # Game over flag
        self.game_over = False
        # Game winner
        self.winner = None

    def make_move(self, big_row, big_col, small_row, small_col):
        """Attempt to make a move at the specified position."""
        if self.game_over:
            return False

        # Validate move against active board
        if self.active_board is not None and (big_row, big_col) != self.active_board:
            return False

        # Check if small board is won or full
        if self.small_board_status[big_row][big_col] is not None:
            return False

        # Check if cell is occupied
        if self.board[big_row][big_col][small_row][small_col] is not None:
            return False

        # Make the move
        self.board[big_row][big_col][small_row][small_col] = self.current_player

        # Check for small board winner
        small_winner = self.check_small_board_winner(big_row, big_col)
        if small_winner:
            self.small_board_status[big_row][big_col] = small_winner
            # Check for game winner
            game_winner = self.check_game_winner()
            if game_winner:
                self.game_over = True
                self.winner = game_winner

        # Update active board: keep same board until won or full
        if self.small_board_status[big_row][big_col] is not None:
            # Board is completed; allow choosing any non-completed board next
            self.active_board = None
        else:
            # Continue in the same board
            self.active_board = (big_row, big_col)

        # Switch player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def check_small_board_winner(self, big_row, big_col):
        """Check if a small board has a winner or is a draw."""
        board = self.board[big_row][big_col]
        # Check rows, columns, diagonals
        for i in range(3):
            if board[i][0] and board[i][0] == board[i][1] == board[i][2]:
                return board[i][0]
            if board[0][i] and board[0][i] == board[1][i] == board[2][i]:
                return board[0][i]
        if board[0][0] and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]
        # Check for draw
        if all(board[i][j] is not None for i in range(3) for j in range(3)):
            return 'D'
        return None

    def check_game_winner(self):
        """Check if the game has a winner or is a draw."""
        for i in range(3):
            if self.small_board_status[i][0] in ['X', 'O'] and self.small_board_status[i][0] == self.small_board_status[i][1] == self.small_board_status[i][2]:
                return self.small_board_status[i][0]
            if self.small_board_status[0][i] in ['X', 'O'] and self.small_board_status[0][i] == self.small_board_status[1][i] == self.small_board_status[2][i]:
                return self.small_board_status[0][i]
        if self.small_board_status[0][0] in ['X', 'O'] and self.small_board_status[0][0] == self.small_board_status[1][1] == self.small_board_status[2][2]:
            return self.small_board_status[0][0]
        if self.small_board_status[0][2] in ['X', 'O'] and self.small_board_status[0][2] == self.small_board_status[1][1] == self.small_board_status[2][0]:
            return self.small_board_status[0][2]
        # Check for draw
        if all(status is not None for row in self.small_board_status for status in row):
            return 'D'
        return None

    def get_legal_moves(self):
        """Return list of legal moves as (big_row, big_col, small_row, small_col)."""
        moves = []
        if self.active_board:
            br, bc = self.active_board
            if self.small_board_status[br][bc] is None:
                moves.extend((br, bc, sr, sc) for sr in range(3) for sc in range(3) if self.board[br][bc][sr][sc] is None)
        if not moves:
            for br in range(3):
                for bc in range(3):
                    if self.small_board_status[br][bc] is None:
                        moves.extend((br, bc, sr, sc) for sr in range(3) for sc in range(3) if self.board[br][bc][sr][sc] is None)
        return moves

class CSPSolver:
    """Constraint Satisfaction Problem solver for Ultimate Tic-Tac-Toe."""
    def __init__(self, game):
        self.game = game

    def get_move(self):
        """Select optimal move using CSP with Forward Checking and AC-3."""
        legal_moves = self.game.get_legal_moves()
        if not legal_moves:
            return None

        # Evaluate moves with MRV heuristic
        moves_scores = [(move, self.evaluate_move(move)) for move in legal_moves]
        moves_scores = self.forward_checking(moves_scores)
        moves_scores = self.arc_consistency(moves_scores)
        
        # Return highest-scoring move
        return max(moves_scores, key=lambda x: x[1])[0] if moves_scores else legal_moves[0]

    def evaluate_move(self, move):
        """Evaluate move based on strategic heuristics."""
        br, bc, sr, sc = move
        game_copy = copy.deepcopy(self.game)
        game_copy.make_move(br, bc, sr, sc)
        score = 0
        # Winning small board
        if game_copy.small_board_status[br][bc] == self.game.current_player:
            score += 100
        # Center moves in small board
        if (sr, sc) == (1, 1):
            score += 10
        # MRV: Fewer opponent moves
        score -= len(game_copy.get_legal_moves()) * 0.5
        return score

    def forward_checking(self, moves_scores):
        """Filter moves that don't lead to immediate opponent wins."""
        filtered = []
        for move, score in moves_scores:
            game_copy = copy.deepcopy(self.game)
            game_copy.make_move(*move)
            opponent_moves = game_copy.get_legal_moves()
            can_win = any(self.check_opponent_win(game_copy, opp_move) for opp_move in opponent_moves)
            if not can_win:
                filtered.append((move, score))
        return filtered or moves_scores

    def check_opponent_win(self, game, move):
        """Check if opponent's move wins the game."""
        game_copy = copy.deepcopy(game)
        game_copy.make_move(*move)
        return game_copy.winner == game.current_player

    def arc_consistency(self, moves_scores):
        """Enforce arc consistency to prune inconsistent moves."""
        refined = []
        for move, score in moves_scores:
            game_copy = copy.deepcopy(self.game)
            game_copy.make_move(*move)
            opponent_moves = game_copy.get_legal_moves()
            if not opponent_moves:
                refined.append((move, score + 1000))
                continue
            worst_outcome = float('inf')
            for opp_move in opponent_moves[:3]:  # Limit for performance
                game_copy2 = copy.deepcopy(game_copy)
                game_copy2.make_move(*opp_move)
                if game_copy2.winner == game_copy.current_player:
                    worst_outcome = float('-inf')
                    break
                our_moves = game_copy2.get_legal_moves()
                outcome = max((self.evaluate_move(m) for m in our_moves[:3]), default=-500) if our_moves else -500
                worst_outcome = min(worst_outcome, outcome)
            if worst_outcome > -100:
                refined.append((move, score + worst_outcome * 0.1))
        return refined or moves_scores

class MinimaxSolver:
    """Minimax solver with alpha-beta pruning for Ultimate Tic-Tac-Toe."""
    def __init__(self, game, max_depth=3):
        self.game = game
        self.max_depth = max_depth

    def get_move(self):
        """Select optimal move using minimax with alpha-beta pruning."""
        legal_moves = self.game.get_legal_moves()
        if not legal_moves:
            return None
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        for move in legal_moves:
            game_copy = copy.deepcopy(self.game)
            game_copy.make_move(*move)
            score = self.minimax(game_copy, 0, False, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        return best_move

    def minimax(self, game, depth, is_maximizing, alpha, beta):
        """Minimax algorithm with alpha-beta pruning."""
        if game.game_over or depth == self.max_depth:
            return self.evaluate_board(game)
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return self.evaluate_board(game)
        if is_maximizing:
            value = float('-inf')
            for move in legal_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(*move)
                value = max(value, self.minimax(game_copy, depth + 1, False, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(*move)
                value = min(value, self.minimax(game_copy, depth + 1, True, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def evaluate_board(self, game):
        """Evaluate board state for minimax."""
        if game.winner == 'X':
            return 1000
        if game.winner == 'O':
            return -1000
        if game.winner == 'D':
            return 0
        score = sum(100 for row in game.small_board_status for status in row if status == 'X') - \
                sum(100 for row in game.small_board_status for status in row if status == 'O')
        for i in range(3):
            for player in ['X', 'O']:
                s = -10 if player == 'O' else 10
                if all(game.small_board_status[i][j] in [player, None] for j in range(3)):
                    score += s + ([game.small_board_status[i][j] for j in range(3)].count(player) == 2) * 40
                if all(game.small_board_status[j][i] in [player, None] for j in range(3)):
                    score += s + ([game.small_board_status[j][i] for j in range(3)].count(player) == 2) * 40
        for player in ['X', 'O']:
            s = -10 if player == 'O' else 10
            if all(game.small_board_status[i][i] in [player, None] for i in range(3)):
                score += s + ([game.small_board_status[i][i] for i in range(3)].count(player) == 2) * 40
            if all(game.small_board_status[i][2-i] in [player, None] for i in range(3)):
                score += s + ([game.small_board_status[i][2-i] for i in range(3)].count(player) == 2) * 40
        return -score if game.current_player == 'O' else score

class HybridSolver:
    """Hybrid solver combining CSP and minimax for optimal moves."""
    def __init__(self, game):
        self.game = game
        self.csp_solver = CSPSolver(game)
        self.minimax_solver = MinimaxSolver(game, max_depth=2)

    def get_move(self):
        """Select move using CSP for early game, minimax for later stages."""
        moves_made = sum(1 for br in range(3) for bc in range(3) for sr in range(3) for sc in range(3)
                         if self.game.board[br][bc][sr][sc] is not None)
        return self.csp_solver.get_move() if moves_made < 10 else self.minimax_solver.get_move()

class GameGUI:
    """GUI for Ultimate Tic-Tac-Toe using Tkinter with experiment mode."""
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate Tic-Tac-Toe")
        self.game = UltimateTicTacToe()
        self.solver = HybridSolver(self.game)
        self.buttons = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.font = font.Font(family="Arial", size=12, weight="bold")
        self.experiment_results = {"CSP": {"wins": 0, "time": 0, "moves": []}, 
                                 "Minimax": {"wins": 0, "time": 0, "moves": []}, 
                                 "Hybrid": {"wins": 0, "time": 0, "moves": []}}
        self.experiment_in_progress = False
        self.current_game = 0
        self.current_solver = None
        self.game_logs = []
        self.create_gui()

    def create_gui(self):
        """Create the 3x3 grid of 3x3 boards with experiment controls."""
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=10)
        for br in range(3):
            for bc in range(3):
                board_frame = tk.Frame(main_frame, borderwidth=2, relief="groove")
                board_frame.grid(row=br, column=bc, padx=5, pady=5)
                for sr in range(3):
                    for sc in range(3):
                        btn = tk.Button(board_frame, text="", width=3, height=1, font=self.font,
                                        command=lambda br=br, bc=bc, sr=sr, sc=sc: self.player_move(br, bc, sr, sc))
                        btn.grid(row=sr, column=sc)
                        self.buttons[br][bc][sr][sc] = btn
        self.status_label = tk.Label(self.root, text="Player X's turn", font=("Arial", 14))
        self.status_label.pack(pady=5)
        self.experiment_button = tk.Button(self.root, text="Run Experiments", font=("Arial", 12),
                                          command=self.start_experiments)
        self.experiment_button.pack(pady=5)
        self.experiment_status = tk.Label(self.root, text="", font=("Arial", 12))
        self.experiment_status.pack(pady=5)
        self.update_board()

    def player_move(self, br, bc, sr, sc):
        """Handle player's move in manual mode."""
        if self.experiment_in_progress:
            messagebox.showwarning("Experiment in Progress", "Cannot play manually during experiments.")
            return
        if self.game.current_player == 'X' and not self.game.game_over:
            if self.game.make_move(br, bc, sr, sc):
                self.update_board()
                if not self.game.game_over:
                    self.root.after(500, self.ai_move)
            else:
                messagebox.showwarning("Invalid Move", "Please select a valid cell in the active board.")

    def ai_move(self):
        """Handle AI's move in manual mode."""
        if self.game.current_player == 'O' and not self.game.game_over:
            move = self.solver.get_move()
            if move:
                self.game.make_move(*move)
                self.update_board()

    def start_experiments(self):
        """Start experiments for CSP, Minimax, and Hybrid solvers."""
        if self.experiment_in_progress:
            messagebox.showwarning("Experiment in Progress", "Please wait for current experiments to complete.")
            return
        self.experiment_in_progress = True
        self.experiment_button.config(state="disabled")
        self.experiment_results = {"CSP": {"wins": 0, "time": 0, "moves": []}, 
                                 "Minimax": {"wins": 0, "time": 0, "moves": []}, 
                                 "Hybrid": {"wins": 0, "time": 0, "moves": []}}
        self.game_logs = []
        self.current_game = 0
        self.solvers = [("CSP", CSPSolver), ("Minimax", MinimaxSolver), ("Hybrid", HybridSolver)]
        self.run_next_experiment()

    def run_next_experiment(self):
        """Run the next game in the experiment sequence."""
        num_games_per_solver = 10
        if self.current_game >= num_games_per_solver * len(self.solvers):
            self.finish_experiments()
            return
        
        solver_idx = self.current_game // num_games_per_solver
        solver_name, solver_class = self.solvers[solver_idx]
        game_num = self.current_game % num_games_per_solver + 1
        self.experiment_status.config(text=f"Running {solver_name} Game {game_num}/{num_games_per_solver}")
        
        # Reset game
        self.game = UltimateTicTacToe()
        self.solver = solver_class(self.game)
        self.current_solver = solver_name
        self.current_game_moves = []
        self.current_game_start = time.time()
        self.update_board()
        self.root.after(500, self.experiment_move)

    def experiment_move(self):
        """Execute a move in the experiment with delay."""
        if self.game.game_over:
            self.log_game_result()
            self.current_game += 1
            self.root.after(1000, self.run_next_experiment)  # Pause before next game
            return
        
        # AI move for current player
        move = self.solver.get_move()
        if move:
            self.current_game_moves.append((self.game.current_player, move))
            self.game.make_move(*move)
            self.update_board()
            self.root.after(500, self.experiment_move)
        else:
            self.log_game_result()
            self.current_game += 1
            self.root.after(1000, self.run_next_experiment)

    def log_game_result(self):
        """Log the result of the current experiment game."""
        game_time = time.time() - self.current_game_start
        self.experiment_results[self.current_solver]["time"] += game_time
        if self.game.winner == 'O':  # AI is 'O'
            self.experiment_results[self.current_solver]["wins"] += 1
        self.experiment_results[self.current_solver]["moves"].append(self.current_game_moves)
        self.game_logs.append({
            "solver": self.current_solver,
            "winner": self.game.winner,
            "moves": self.current_game_moves,
            "time": game_time
        })

    def finish_experiments(self):
        """Display experiment results and reset state."""
        self.experiment_in_progress = False
        self.experiment_button.config(state="normal")
        self.experiment_status.config(text="Experiments Complete")
        
        # Summarize results
        num_games = 10
        result_text = "Experiment Results:\n"
        for solver_name in self.experiment_results:
            data = self.experiment_results[solver_name]
            avg_time = data["time"] / num_games
            win_rate = data["wins"] / num_games
            result_text += f"{solver_name}: Wins={data['wins']}/{num_games}, Avg Time={avg_time:.4f}s, Win Rate={win_rate:.2%}\n"
        
        messagebox.showinfo("Experiment Results", result_text)
        self.experiment_status.config(text="")
        
        # Reset game for manual play
        self.game = UltimateTicTacToe()
        self.solver = HybridSolver(self.game)
        self.update_board()

    def update_board(self):
        """Update GUI to reflect current game state."""
        for br in range(3):
            for bc in range(3):
                bg = "lightgreen" if self.game.active_board == (br, bc) else "white"
                for sr in range(3):
                    for sc in range(3):
                        cell = self.game.board[br][bc][sr][sc]
                        text = cell if cell else ""
                        state = "disabled" if cell or self.game.small_board_status[br][bc] or self.experiment_in_progress else "normal"
                        self.buttons[br][bc][sr][sc].config(text=text, state=state, bg=bg)
                if self.game.small_board_status[br][bc] in ['X', 'O', 'D']:
                    for sr in range(3):
                        for sc in range(3):
                            self.buttons[br][bc][sr][sc].config(state="disabled", bg="lightgray")
        self.status_label.config(text=f"Player {self.game.current_player}'s turn")
        if self.game.game_over:
            result = f"Player {self.game.winner} wins!" if self.game.winner in ['X', 'O'] else "Draw!"
            if not self.experiment_in_progress:
                messagebox.showinfo("Game Over", result)
            self.status_label.config(text=result)

if __name__ == "__main__":
    root = tk.Tk()
    gui = GameGUI(root)
    root.mainloop()