import numpy as np
import random
import time
import pickle
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os
import json
from datetime import datetime

# Game constants
GRID_WIDTH = 10
GRID_HEIGHT = 20

# Tetromino shapes in matrix format for efficient processing
SHAPES = {
    'I': np.array([[1, 1, 1, 1]]),
    'O': np.array([[1, 1], [1, 1]]),
    'T': np.array([[1, 1, 1], [0, 1, 0]]),
    'S': np.array([[0, 1, 1], [1, 1, 0]]),
    'Z': np.array([[1, 1, 0], [0, 1, 1]]),
    'J': np.array([[1, 0, 0], [1, 1, 1]]),
    'L': np.array([[0, 0, 1], [1, 1, 1]])
}

@dataclass
class GameState:
    """Represents the current state of the game for efficient processing"""
    board: np.ndarray
    current_piece: str
    current_rotation: int
    current_pos: Tuple[int, int]
    score: int
    lines_cleared: int
    game_over: bool = False

class TetrisMatrixGame:
    """Efficient matrix-based Tetris game implementation"""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset game state"""
        self.board = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int8)
        self.current_piece = random.choice(list(SHAPES.keys()))
        self.next_piece = random.choice(list(SHAPES.keys()))
        self.current_rotation = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self._spawn_piece()

    def _rotate_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Rotate a piece matrix clockwise"""
        return np.rot90(matrix, k=-1)

    def _spawn_piece(self) -> bool:
        """Spawn a new piece and check if game is over"""
        self.current_piece = self.next_piece
        self.next_piece = random.choice(list(SHAPES.keys()))
        self.current_rotation = 0
        piece_matrix = SHAPES[self.current_piece]

        # Calculate spawn position (center top)
        self.current_pos = (0, GRID_WIDTH // 2 - piece_matrix.shape[1] // 2)

        return self._is_valid_move(piece_matrix, self.current_pos)

    def _is_valid_move(self, piece_matrix: np.ndarray, pos: Tuple[int, int]) -> bool:
        """Check if a move is valid using matrix operations"""
        row, col = pos
        height, width = piece_matrix.shape

        # Check boundaries
        if (row < 0 or row + height > GRID_HEIGHT or
                col < 0 or col + width > GRID_WIDTH):
            return False

        # Check collision with existing pieces
        try:
            board_section = self.board[row:row + height, col:col + width]
            return not np.any(np.logical_and(board_section, piece_matrix))
        except Exception as e:
            print(f"Error checking move validity: {e}")
            return False

    def _clear_lines(self) -> int:
        """Clear completed lines and return number of lines cleared"""
        lines_to_clear = []
        for i in range(GRID_HEIGHT):
            if np.all(self.board[i]):
                lines_to_clear.append(i)

        if lines_to_clear:
            # Remove completed lines
            self.board = np.delete(self.board, lines_to_clear, axis=0)
            # Add new empty lines at top
            new_rows = np.zeros((len(lines_to_clear), GRID_WIDTH), dtype=np.int8)
            self.board = np.vstack((new_rows, self.board))

        return len(lines_to_clear)

    def get_state(self) -> GameState:
        """Return current game state"""
        return GameState(
            board=self.board.copy(),
            current_piece=self.current_piece,
            current_rotation=self.current_rotation,
            current_pos=self.current_pos,
            score=self.score,
            lines_cleared=self.lines_cleared,
            game_over=self.game_over
        )

    def get_next_states(self) -> List[Tuple[GameState, Tuple[int, int, int]]]:
        """Generate all possible next states from current position"""
        states = []
        piece = SHAPES[self.current_piece]

        # Try all rotations and positions
        for rotation in range(4):
            rotated_piece = np.rot90(piece, k=-rotation)
            height, width = rotated_piece.shape

            # Try all horizontal positions
            for col in range(-width + 1, GRID_WIDTH):
                # Find lowest valid position
                for row in range(GRID_HEIGHT - 1, -1, -1):
                    if self._is_valid_move(rotated_piece, (row, col)):
                        try:
                            # Create new state
                            new_board = self.board.copy()
                            new_board[row:row + height,
                            col:col + width] |= rotated_piece

                            new_state = GameState(
                                board=new_board,
                                current_piece=self.current_piece,
                                current_rotation=rotation,
                                current_pos=(row, col),
                                score=self.score,
                                lines_cleared=self.lines_cleared
                            )
                            states.append((new_state, (rotation, row, col)))
                            break
                        except Exception as e:
                            print(f"Error generating next state: {e}")
                            continue

        return states

class TetrisAIWeights:
    """Manages the AI's decision-making weights"""
    def __init__(self, load_file: Optional[str] = None):
        if load_file and os.path.exists(load_file):
            self.load_weights(load_file)
        else:
            self.initialize_random_weights()

        self.fitness = 0
        self.total_games = 0
        self.total_lines = 0
        self.max_score = 0

    def initialize_random_weights(self):
        """Initialize weights with carefully chosen ranges"""
        self.weights = {
            'holes': random.uniform(-0.8, -0.5),
            'bumpiness': random.uniform(-0.4, -0.2),
            'total_height': random.uniform(-0.3, -0.1),
            'cleared_lines': random.uniform(0.5, 0.8),
            'max_height': random.uniform(-0.4, -0.2),
            'avg_height': random.uniform(-0.3, -0.1),
            'hole_depth': random.uniform(-0.4, -0.2),
            'well_score': random.uniform(0.1, 0.3),
            'side_touch': random.uniform(0.1, 0.2),
            'flat_factor': random.uniform(0.1, 0.3)
        }

    def save_weights(self, filename: str):
        """Save weights with additional metadata"""
        data = {
            'weights': self.weights,
            'fitness': self.fitness,
            'total_games': self.total_games,
            'total_lines': self.total_lines,
            'max_score': self.max_score,
            'timestamp': datetime.now().isoformat()
        }
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"Successfully saved weights to {filename}")
        except Exception as e:
            print(f"Error saving weights to {filename}: {e}")

    def load_weights(self, filename: str):
        """Load weights and metadata with validation"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

                # Validate the loaded data
                expected_keys = {'holes', 'bumpiness', 'total_height', 'cleared_lines',
                                 'max_height', 'avg_height', 'hole_depth', 'well_score',
                                 'side_touch', 'flat_factor'}

                if not isinstance(data, dict) or 'weights' not in data:
                    print(f"Invalid checkpoint format in {filename}")
                    self.initialize_random_weights()
                    return

                if not all(key in data['weights'] for key in expected_keys):
                    print(f"Missing weights in checkpoint {filename}")
                    self.initialize_random_weights()
                    return

                self.weights = data['weights']
                self.fitness = data.get('fitness', 0)
                self.total_games = data.get('total_games', 0)
                self.total_lines = data.get('total_lines', 0)
                self.max_score = data.get('max_score', 0)

                print(f"Successfully loaded weights from {filename}")

        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading checkpoint {filename}: {e}")
            self.initialize_random_weights()
        except Exception as e:
            print(f"Unexpected error loading checkpoint {filename}: {e}")
            self.initialize_random_weights()

    def mutate(self, mutation_rate: float = 0.1, generation: int = 0):
        """Adaptive mutation with generation-based adjustments"""
        adapted_rate = mutation_rate * (1 - generation / 1000)

        for key in self.weights:
            if random.random() < adapted_rate:
                mutation_range = 0.05 if self.fitness > 5000 else 0.1
                self.weights[key] += random.uniform(-mutation_range, mutation_range)
                self.weights[key] = max(-1.0, min(1.0, self.weights[key]))

class TetrisAI:
    """AI player using matrix-based evaluation"""
    def __init__(self, weights: TetrisAIWeights):
        self.weights = weights

    def evaluate_state(self, state: GameState) -> float:
        """Evaluate a game state using vectorized operations"""
        try:
            board = np.asarray(state.board)

            # Get column heights
            cols = np.any(board != 0, axis=0)
            heights = np.zeros(GRID_WIDTH)
            for i in range(GRID_WIDTH):
                if cols[i]:
                    heights[i] = GRID_HEIGHT - np.argmax(board[:, i] != 0)

            # Calculate holes and depths
            holes = 0
            hole_depth = 0
            for col in range(GRID_WIDTH):
                found_block = False
                col_height = int(heights[col])
                for row in range(GRID_HEIGHT):
                    if board[row, col] != 0:
                        found_block = True
                    elif found_block:
                        holes += 1
                        hole_depth += col_height - row

            # Calculate bumpiness
            bumpiness = np.sum(np.abs(np.diff(heights)))

            # Calculate well scores
            well_score = 0
            for col in range(1, GRID_WIDTH-1):
                for row in range(GRID_HEIGHT):
                    if (board[row, col] == 0 and
                            board[row, col-1] != 0 and
                            board[row, col+1] != 0):
                        well_score += 1

            # Calculate aggregate features
            total_height = np.sum(heights)
            max_height = np.max(heights)
            avg_height = np.mean(heights)
            flat_factor = np.sum(heights == avg_height)
            side_touch = np.sum(board[:, [0, -1]] != 0)

            # Combine all features
            score = (
                    self.weights.weights['holes'] * holes +
                    self.weights.weights['bumpiness'] * bumpiness +
                    self.weights.weights['total_height'] * total_height +
                    self.weights.weights['max_height'] * max_height +
                    self.weights.weights['avg_height'] * avg_height +
                    self.weights.weights['hole_depth'] * hole_depth +
                    self.weights.weights['well_score'] * well_score +
                    self.weights.weights['side_touch'] * side_touch +
                    self.weights.weights['flat_factor'] * flat_factor
            )

            return float(score)

        except Exception as e:
            print(f"Error in evaluate_state: {e}")
            return float('-inf')

    def get_best_move(self, game: TetrisMatrixGame) -> Optional[Tuple[int, int, int]]:
        """Find the best move for the current game state"""
        try:
            best_score = float('-inf')
            best_move = None

            # Get all possible next states
            next_states = game.get_next_states()

            for state, move in next_states:
                score = self.evaluate_state(state)
                if score > best_score:
                    best_score = score
                    best_move = move

            return best_move

        except Exception as e:
            print(f"Error in get_best_move: {e}")
            return None

def train_generation(population: List[TetrisAIWeights],
                     games_per_genome: int = 5,
                     generation: int = 0) -> List[TetrisAIWeights]:
    """Train a generation of AI weights"""
    for i, weights in enumerate(population):
        total_score = 0
        total_lines = 0

        # Print progress
        if i % 10 == 0:
            print(f"Training genome {i}/{len(population)}")

        # Play multiple games
        for game_num in range(games_per_genome):
            game = TetrisMatrixGame()
            ai = TetrisAI(weights)
            moves = 0
            max_moves = 500  # Limit moves per game

            while not game.game_over and moves < max_moves:
                try:
                    best_move = ai.get_best_move(game)
                    if best_move is None:
                        break

                    rotation, row, col = best_move

                    # Get the correct piece shape after rotation
                    piece = SHAPES[game.current_piece]
                    for _ in range(rotation):
                        piece = np.rot90(piece, k=-1)

                    height, width = piece.shape

                    # Check if the piece can be placed at the position
                    if (row + height <= GRID_HEIGHT and
                            col + width <= GRID_WIDTH and
                            row >= 0 and col >= 0):

                        # Update position and lock piece
                        game.current_pos = (row, col)
                        game.board[row:row + height,
                        col:col + width] |= piece

                        # Clear lines and update score
                        lines = game._clear_lines()
                        total_lines += lines
                        total_score += (lines * 100 + 1)  # Score for lines plus move bonus

                        # Spawn new piece
                        if not game._spawn_piece():
                            game.game_over = True
                    else:
                        game.game_over = True

                    moves += 1

                except Exception as e:
                    print(f"Error during move: {e}")
                    break

        # Update weights statistics
        weights.fitness = total_score / games_per_genome
        weights.total_lines += total_lines
        weights.total_games += games_per_genome
        weights.max_score = max(weights.max_score, total_score)

    # Sort population by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population

def create_next_generation(population: List[TetrisAIWeights],
                           population_size: int,
                           generation: int) -> List[TetrisAIWeights]:
    """Create the next generation using selection and mutation"""
    # Keep top performers
    elite_count = max(population_size // 10,
                      int(population_size * (generation / 1000)))
    next_gen = population[:elite_count]

    # Tournament selection
    def tournament_select():
        tournament = random.sample(population[:len(population)//2], 5)
        return max(tournament, key=lambda x: x.fitness)

    # Create new individuals
    while len(next_gen) < population_size:
        parent1 = tournament_select()
        parent2 = tournament_select()

        # Create child with mixed weights
        child = TetrisAIWeights()
        for key in child.weights:
            # Bias towards better parent
            if parent1.fitness > parent2.fitness:
                child.weights[key] = parent1.weights[key] if random.random() < 0.7 else parent2.weights[key]
            else:
                child.weights[key] = parent2.weights[key] if random.random() < 0.7 else parent1.weights[key]

        # Apply mutation
        child.mutate(mutation_rate=0.1, generation=generation)
        next_gen.append(child)

    return next_gen

def main():
    """Main training loop with checkpointing"""
    # Training parameters
    POPULATION_SIZE = 100
    GENERATIONS = 500
    GAMES_PER_GENOME = 5
    CHECKPOINT_INTERVAL = 10

    # Initialize population
    population = [TetrisAIWeights() for _ in range(POPULATION_SIZE)]

    # Load previous checkpoint if exists
    checkpoint_dir = "checkpoints"
    try:
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir)
                                  if f.startswith("best_weights_gen_")])
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                print(f"Found checkpoint: {latest_checkpoint}")
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

                # Try to load checkpoint
                try:
                    population[0] = TetrisAIWeights(checkpoint_path)
                    print("Successfully initialized from checkpoint")
                except Exception as e:
                    print(f"Failed to load checkpoint: {e}")
                    print("Starting with fresh population")
    except Exception as e:
        print(f"Error handling checkpoints: {e}")
        print("Starting with fresh population")

    best_fitness = 0
    stagnant_generations = 0

    # Training loop
    for generation in range(GENERATIONS):
        start_time = time.time()

        # Train generation
        population = train_generation(population, GAMES_PER_GENOME, generation)

        # Check for improvement
        if population[0].fitness > best_fitness:
            best_fitness = population[0].fitness
            stagnant_generations = 0

            # Save checkpoint for significant improvements
            if generation % CHECKPOINT_INTERVAL == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                population[0].save_weights(
                    os.path.join(checkpoint_dir, f"best_weights_gen_{generation}.pkl")
                )
        else:
            stagnant_generations += 1

        # Print generation statistics
        elapsed_time = time.time() - start_time
        print(f"\nGeneration {generation} completed in {elapsed_time:.2f} seconds")
        print(f"Best Fitness: {population[0].fitness}")
        print(f"Best Lines Cleared: {population[0].total_lines / population[0].total_games:.2f}")
        print(f"Best Weights: {population[0].weights}")

        # Handle stagnation
        if stagnant_generations > 20:
            print("\nTraining stagnant. Introducing diversity...")
            # Replace bottom 30% with new random weights
            new_count = POPULATION_SIZE // 3
            population = population[:-new_count] + [TetrisAIWeights() for _ in range(new_count)]
            stagnant_generations = 0

        # Create next generation
        population = create_next_generation(population, POPULATION_SIZE, generation)

    # Save final results
    print("\nTraining complete!")
    final_dir = os.path.join(checkpoint_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    population[0].save_weights(os.path.join(final_dir, "best_weights_final.pkl"))

    # Print final statistics
    print(f"\nFinal Best Fitness: {population[0].fitness}")
    print(f"Final Average Lines per Game: {population[0].total_lines / population[0].total_games:.2f}")
    print(f"Final Max Score: {population[0].max_score}")
    print("\nFinal Best Weights:")
    for key, value in population[0].weights.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()