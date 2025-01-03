import pyglet
import random
import numpy as np
from pyglet import shapes
from pyglet.window import key
from pyglet.window import Window

# Game constants from original implementation
SCREEN_WIDTH = 1280  # Default to common resolution
SCREEN_HEIGHT = 720
BLOCK_SIZE = int(SCREEN_HEIGHT * 0.04)
GRID_WIDTH = 10
GRID_HEIGHT = 20
GAME_WIDTH = BLOCK_SIZE * (GRID_WIDTH + 8)
GAME_HEIGHT = BLOCK_SIZE * GRID_HEIGHT
SPACING = BLOCK_SIZE * 4
GAME1_X = (SCREEN_WIDTH - (2 * GAME_WIDTH + SPACING)) // 2
GAME2_X = GAME1_X + GAME_WIDTH + SPACING
GAME_Y = (SCREEN_HEIGHT - GAME_HEIGHT) // 2

# Background colors (Added missing color definitions)
BG_COLOR = (20, 20, 35)
GRID_COLOR = (40, 40, 60)
BORDER_COLOR = (60, 60, 80)

# Colors and shapes from original implementation
COLORS = {
    'I': (0, 240, 240),    # Cyan
    'O': (240, 240, 0),    # Yellow
    'T': (160, 0, 240),    # Purple
    'S': (0, 240, 0),      # Green
    'Z': (240, 0, 0),      # Red
    'J': (0, 0, 240),      # Blue
    'L': (240, 160, 0),    # Orange
    'B': (80, 80, 80),     # Gray (for blocked cells)
}

SHAPES = {
    'I': [(0, 0), (0, 1), (0, 2), (0, 3)],
    'O': [(0, 0), (0, 1), (1, 0), (1, 1)],
    'T': [(0, 0), (0, 1), (0, 2), (1, 1)],
    'S': [(0, 0), (0, 1), (1, 1), (1, 2)],
    'Z': [(0, 1), (0, 2), (1, 0), (1, 1)],
    'J': [(0, 0), (1, 0), (1, 1), (1, 2)],
    'L': [(0, 2), (1, 0), (1, 1), (1, 2)],
}

# AI Matrix shapes for efficient processing
MATRIX_SHAPES = {
    'I': np.array([[1, 1, 1, 1]]),
    'O': np.array([[1, 1], [1, 1]]),
    'T': np.array([[1, 1, 1], [0, 1, 0]]),
    'S': np.array([[0, 1, 1], [1, 1, 0]]),
    'Z': np.array([[1, 1, 0], [0, 1, 1]]),
    'J': np.array([[1, 0, 0], [1, 1, 1]]),
    'L': np.array([[0, 0, 1], [1, 1, 1]])
}

class TetrisAIWeights:
    def __init__(self, weights_file=None):
        if weights_file:
            self.load_weights(weights_file)
        else:
            self.weights = {
                'height': -0.1073970056360456,
                'holes': 0.04098181069888697,
                'bumpiness': -0.0049203169309888495,
                'lines': 0.5135583793719365,
                'clear_bonus': 1.0,
                'well_depth': -0.03538849731434979,
                'covered_holes': -1.0,
                'edge_touch': 0.027422090700334953,
                'flat_factor': -0.8348604491911062
            }

    def load_weights(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                self.weights = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':')
                        key = key.strip()
                        if key in ['Fitness', 'Average lines cleared']:
                            continue
                        try:
                            value = float(value.strip())
                            self.weights[key] = value
                        except ValueError:
                            print(f"Skipping invalid weight value for {key}")
                            continue
        except Exception as e:
            print(f"Error loading weights: {e}")
            self.initialize_default_weights()

    def initialize_default_weights(self):
        self.weights = {
            'height': -0.1073970056360456,
            'holes': 0.04098181069888697,
            'bumpiness': -0.0049203169309888495,
            'lines': 0.5135583793719365,
            'clear_bonus': 1.0,
            'well_depth': -0.03538849731434979,
            'covered_holes': -1.0,
            'edge_touch': 0.027422090700334953,
            'flat_factor': -0.8348604491911062
        }

from typing import List, Tuple

class TetrisAI:
    def __init__(self, weights: TetrisAIWeights):
        self.genome = weights
        self.move_history = []
        self.last_clear_height = GRID_HEIGHT

    def evaluate_position(self, player_state, test_pos: List[int], test_shape: List[Tuple[int, int]] | List[List[int]]) -> float:
        # Create a copy of the current grid
        test_grid = [row[:] for row in player_state.grid]

        # Place the piece
        for block in test_shape:
            x = test_pos[0] + block[0]
            y = test_pos[1] + block[1]
            if 0 <= y < GRID_HEIGHT:
                test_grid[y][x] = player_state.current_piece

        # Calculate enhanced features
        height = self._get_height(test_grid)
        holes = self._count_holes(test_grid)
        covered_holes = self._count_covered_holes(test_grid)
        bumpiness = self._get_bumpiness(test_grid)
        lines = self._count_complete_lines(test_grid)
        well_score = self._evaluate_well(test_grid)
        edge_score = self._evaluate_edge_touching(test_shape, test_pos)
        top_penalty = self._calculate_top_penalty(height)

        # Dynamic line clear bonus
        clear_bonus = lines * lines * self.genome.weights.get('clear_bonus', 1.0)

        # Calculate comprehensive score
        score = (
                self.genome.weights.get('height', -0.1) * height +
                self.genome.weights.get('holes', 0.04) * holes +
                self.genome.weights.get('covered_holes', -1.0) * covered_holes +
                self.genome.weights.get('bumpiness', -0.005) * bumpiness +
                self.genome.weights.get('lines', 0.5) * lines +
                clear_bonus +
                self.genome.weights.get('well_depth', -0.04) * well_score +
                self.genome.weights.get('edge_touch', 0.03) * edge_score +
                self.genome.weights.get('top_penalty', -0.8) * top_penalty
        )

        # Punish moves that create isolated holes
        if covered_holes > holes:
            score *= 0.8

        return score

    def _get_height(self, grid):
        heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if grid[y][x] is not None:
                    heights.append(GRID_HEIGHT - y)
                    break
            else:
                heights.append(0)
        return sum(heights) / len(heights)

    def _count_holes(self, grid):
        holes = 0
        for x in range(GRID_WIDTH):
            found_block = False
            for y in range(GRID_HEIGHT):
                if grid[y][x] is not None:
                    found_block = True
                elif found_block and grid[y][x] is None:
                    holes += 1
        return holes

    def _count_covered_holes(self, grid):
        covered_holes = 0
        for x in range(GRID_WIDTH):
            blocks_above = 0
            for y in range(GRID_HEIGHT-1, -1, -1):
                if grid[y][x] is not None:
                    blocks_above += 1
                elif blocks_above > 0:
                    covered_holes += blocks_above
        return covered_holes

    def _get_bumpiness(self, grid):
        heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if grid[y][x] is not None:
                    heights.append(GRID_HEIGHT - y)
                    break
            else:
                heights.append(0)

        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def _count_complete_lines(self, grid):
        return sum(1 for y in range(GRID_HEIGHT) if all(cell is not None for cell in grid[y]))

    def _evaluate_well(self, grid):
        well_score = 0
        for x in range(GRID_WIDTH):
            empty_count = 0
            for y in range(GRID_HEIGHT):
                if grid[y][x] is None:
                    if (x > 0 and grid[y][x-1] is not None) and (x < GRID_WIDTH-1 and grid[y][x+1] is not None):
                        empty_count += 1
                else:
                    break
            if 2 <= empty_count <= 4:  # Ideal well depth
                well_score += 1
        return well_score

    def _evaluate_edge_touching(self, shape, pos):
        edge_count = 0
        for block in shape:
            x = pos[0] + block[0]
            if x == 0 or x == GRID_WIDTH - 1:
                edge_count += 1
        return edge_count / len(shape)

    def _calculate_top_penalty(self, height):
        threshold = GRID_HEIGHT * 0.7
        if height > threshold:
            return (height - threshold) / threshold
        return 0

    def get_best_move(self, player_state) -> tuple:
        try:
            best_score = float('-inf')
            current_pos = player_state.current_pos
            best_move = (0, current_pos[0], current_pos[1] - 1)

            # Try all rotations
            shape = player_state.current_shape
            for rotation in range(4):
                # Try all horizontal positions
                for x in range(-2, GRID_WIDTH + 2):
                    # Start from current y position
                    y = current_pos[1]

                    # Drop piece to bottom
                    while y >= 0 and player_state.is_valid_move(shape, [x, y - 1]):
                        y -= 1

                    # If move is valid, evaluate it
                    if player_state.is_valid_move(shape, [x, y]):
                        score = self.evaluate_position(player_state, [x, y], shape)
                        if score > best_score:
                            best_score = score
                            best_move = (rotation, x, y)

                # Rotate shape for next iteration
                shape = [(y, -x) for x, y in shape]
                # Normalize shape
                min_x = min(x for x, y in shape)
                min_y = min(y for x, y in shape)
                shape = [(x - min_x, y - min_y) for x, y in shape]

            return best_move

        except Exception as e:
            print(f"Error in get_best_move: {e}")
            return (0, current_pos[0], current_pos[1] - 1)


class PlayerState:
    """Enhanced player state with AI support"""
    def __init__(self, game_x, is_player_one=True, game_window=None, is_ai=False):
        print(f"Initializing PlayerState: {'AI' if is_ai else 'Human'} player at x={game_x}")
        self.game_x = game_x
        self.is_player_one = is_player_one
        self.game_window = game_window
        self.is_ai = is_ai
        self.current_rotation = 0  # Track current rotation
        self.move_delay = 0.0  # For controlling AI move timing

        # Initialize AI without loading weights file for now
        self.ai = TetrisAI(TetrisAIWeights()) if is_ai else None
        print("AI initialized" if is_ai else "Human player initialized")

        self.reset()
        self.create_labels()

    def reset(self):
        """Reset game state"""
        self.grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.game_over = False
        self.current_piece = None
        self.current_shape = None
        self.current_pos = None
        self.next_piece = random.choice(list(SHAPES.keys()))
        self.spawn_new_piece()

    def spawn_new_piece(self):
        """Spawn a new piece at the top of the grid"""
        self.current_piece = self.next_piece
        self.next_piece = random.choice(list(SHAPES.keys()))
        self.current_shape = SHAPES[self.current_piece]
        self.current_pos = [GRID_WIDTH // 2 - 2, GRID_HEIGHT - 1]

        if not self.is_valid_move(self.current_shape, self.current_pos):
            self.game_over = True

    def is_valid_move(self, shape, pos):
        """Check if a move is valid"""
        for block in shape:
            x = pos[0] + block[0]
            y = pos[1] + block[1]

            if x < 0 or x >= GRID_WIDTH or y < 0:
                return False

            if y >= GRID_HEIGHT:
                continue

            if self.grid[y][x] is not None:
                return False
        return True

    def rotate_shape(self):
        """Rotate the current piece"""
        if self.current_piece == 'O':  # Square doesn't need rotation
            return

        new_shape = [(y, -x) for x, y in self.current_shape]
        min_x = min(x for x, y in new_shape)
        min_y = min(y for x, y in new_shape)
        new_shape = [(x - min_x, y - min_y) for x, y in new_shape]

        if self.is_valid_move(new_shape, self.current_pos):
            self.current_shape = new_shape

    def lock_piece(self):
        """Lock the current piece in place"""
        for block in self.current_shape:
            x = self.current_pos[0] + block[0]
            y = self.current_pos[1] + block[1]
            if 0 <= y < GRID_HEIGHT:
                self.grid[y][x] = self.current_piece

        self.clear_lines()
        self.spawn_new_piece()

    def clear_lines(self):
        """Clear completed lines and update score"""
        lines_cleared = 0
        rows_to_clear = []

        for y in range(GRID_HEIGHT):
            if all(cell is not None for cell in self.grid[y]) and not all(cell == 'B' for cell in self.grid[y]):
                rows_to_clear.append(y)
                lines_cleared += 1

        if lines_cleared > 0:
            for row in sorted(rows_to_clear, reverse=True):
                self.grid.pop(row)

            for _ in range(lines_cleared):
                self.grid.append([None for _ in range(GRID_WIDTH)])

            self.lines += lines_cleared
            self.score += lines_cleared * 100 * self.level

            opponent = self.game_window.player2 if self.is_player_one else self.game_window.player1
            if opponent:  # Add blocked rows to opponent
                opponent.add_blocked_rows(lines_cleared * 2)

            new_level = (self.lines // 10) + 1
            if new_level != self.level:
                self.level = new_level

            self.score_label.text = f'Score: {self.score}'
            self.lines_label.text = f'Lines: {self.lines}'

    def add_blocked_rows(self, num_rows):
        """Add blocked rows at the bottom pushing existing rows up"""
        for y in range(GRID_HEIGHT - 1, num_rows - 1, -1):
            for x in range(GRID_WIDTH):
                self.grid[y][x] = self.grid[y - num_rows][x]

        for y in range(num_rows):
            for x in range(GRID_WIDTH):
                self.grid[y][x] = 'B'

    def create_labels(self):
        """Create UI labels"""
        try:
            player_name = "AI PLAYER" if self.is_ai else ("PLAYER 1" if self.is_player_one else "PLAYER 2")
            self.title_label = pyglet.text.Label(player_name,
                                                 x=self.game_x + GAME_WIDTH - BLOCK_SIZE * 4,
                                                 y=SCREEN_HEIGHT - BLOCK_SIZE * 2,
                                                 anchor_x='center',
                                                 font_size=BLOCK_SIZE,
                                                 bold=True,
                                                 color=(255, 255, 255, 255))

            self.score_label = pyglet.text.Label('Score: 0',
                                                 x=self.game_x + GAME_WIDTH - BLOCK_SIZE * 4,
                                                 y=SCREEN_HEIGHT - BLOCK_SIZE * 4,
                                                 anchor_x='center',
                                                 font_size=int(BLOCK_SIZE * 0.6),
                                                 color=(200, 200, 200, 255))

            self.lines_label = pyglet.text.Label('Lines: 0',
                                                 x=self.game_x + GAME_WIDTH - BLOCK_SIZE * 4,
                                                 y=SCREEN_HEIGHT - BLOCK_SIZE * 5,
                                                 anchor_x='center',
                                                 font_size=int(BLOCK_SIZE * 0.6),
                                                 color=(200, 200, 200, 255))

            self.next_label = pyglet.text.Label('Next',
                                                x=self.game_x + GAME_WIDTH - BLOCK_SIZE * 4,
                                                y=SCREEN_HEIGHT - BLOCK_SIZE * 8,
                                                anchor_x='center',
                                                font_size=int(BLOCK_SIZE * 0.6),
                                                color=(200, 200, 200, 255))
            print(f"Labels created for {player_name}")
        except Exception as e:
            print(f"Error creating labels: {e}")
        # ... [rest of label creation code remains the same]

    def update(self, dt):
        """Update game state"""
        if self.game_over:
            return

        if self.is_ai:
            # Add move delay
            self.move_delay += dt
            if self.move_delay < 0.1:  # Slightly slower for better visibility
                return
            self.move_delay = 0

            # Get AI move
            best_move = self.ai.get_best_move(self)
            if best_move:
                rotation, target_x, target_y = best_move

                # First, try rotation if needed
                while rotation > 0:
                    self.rotate_shape()
                    rotation -= 1

                # Then, move horizontally if needed
                dx = target_x - self.current_pos[0]
                if dx != 0:
                    new_x = self.current_pos[0] + (1 if dx > 0 else -1)
                    new_pos = [new_x, self.current_pos[1]]
                    if self.is_valid_move(self.current_shape, new_pos):
                        self.current_pos = new_pos

                # Always try to move down
                self.move_down()
        else:
            # Regular game update
            self.move_down()

    def move_down(self):
        """Move the current piece down one row"""
        if self.game_over:
            return False

        new_pos = [self.current_pos[0], self.current_pos[1] - 1]
        if self.is_valid_move(self.current_shape, new_pos):
            self.current_pos = new_pos
            return True
        else:
            self.lock_piece()
            return False

    def draw(self):
        """Draw the game state"""
        try:
            # Draw game area background
            background = shapes.Rectangle(self.game_x, GAME_Y,
                                          BLOCK_SIZE * GRID_WIDTH,
                                          BLOCK_SIZE * GRID_HEIGHT,
                                          color=GRID_COLOR)
            background.draw()

            # Draw grid lines
            for i in range(GRID_WIDTH + 1):
                line = shapes.Line(self.game_x + i * BLOCK_SIZE, GAME_Y,
                                   self.game_x + i * BLOCK_SIZE, GAME_Y + GAME_HEIGHT,
                                   color=BORDER_COLOR + (100,))
                line.draw()

            for i in range(GRID_HEIGHT + 1):
                line = shapes.Line(self.game_x, GAME_Y + i * BLOCK_SIZE,
                                   self.game_x + GRID_WIDTH * BLOCK_SIZE, GAME_Y + i * BLOCK_SIZE,
                                   color=BORDER_COLOR + (100,))
                line.draw()

            # Draw grid blocks
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if self.grid[y][x] is not None:
                        self.draw_block(self.game_x + x * BLOCK_SIZE,
                                        GAME_Y + y * BLOCK_SIZE,
                                        COLORS[self.grid[y][x]])

            # Draw current piece
            if not self.game_over and self.current_piece and self.current_shape:
                color = COLORS[self.current_piece]
                for block in self.current_shape:
                    x = self.game_x + (self.current_pos[0] + block[0]) * BLOCK_SIZE
                    y = GAME_Y + (self.current_pos[1] + block[1]) * BLOCK_SIZE
                    if 0 <= self.current_pos[1] + block[1] < GRID_HEIGHT:
                        self.draw_block(x, y, color)

            # Draw next piece preview
            preview_x = self.game_x + GAME_WIDTH - BLOCK_SIZE * 6
            preview_y = SCREEN_HEIGHT - BLOCK_SIZE * 10

            # Draw preview box
            preview_box = shapes.Rectangle(preview_x - BLOCK_SIZE,
                                           preview_y - BLOCK_SIZE,
                                           BLOCK_SIZE * 4,
                                           BLOCK_SIZE * 4,
                                           color=GRID_COLOR)
            preview_box.draw()

            # Draw next piece in preview
            next_color = COLORS[self.next_piece]
            next_shape = SHAPES[self.next_piece]
            for block in next_shape:
                x = preview_x + block[0] * BLOCK_SIZE
                y = preview_y + block[1] * BLOCK_SIZE
                self.draw_block(x, y, next_color)

            # Draw UI elements
            self.title_label.draw()
            self.score_label.draw()
            self.lines_label.draw()
            self.next_label.draw()

            # Draw game over text if needed
            if self.game_over:
                game_over_label = pyglet.text.Label(f'Game Over!',
                                                    x=self.game_x + GAME_WIDTH//2,
                                                    y=SCREEN_HEIGHT//2,
                                                    anchor_x='center',
                                                    anchor_y='center',
                                                    font_size=BLOCK_SIZE,
                                                    bold=True,
                                                    color=(255, 0, 0, 255))
                game_over_label.draw()

        except Exception as e:
            print(f"Error in PlayerState.draw: {e}")
            import traceback
            traceback.print_exc()

        # Draw grid blocks
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] is not None:
                    self.draw_block(self.game_x + x * BLOCK_SIZE,
                                    GAME_Y + y * BLOCK_SIZE,
                                    COLORS[self.grid[y][x]])

        # Draw current piece
        if not self.game_over:
            color = COLORS[self.current_piece]
            for block in self.current_shape:
                x = self.game_x + (self.current_pos[0] + block[0]) * BLOCK_SIZE
                y = GAME_Y + (self.current_pos[1] + block[1]) * BLOCK_SIZE
                if 0 <= self.current_pos[1] + block[1] < GRID_HEIGHT:
                    self.draw_block(x, y, color)

        # Draw next piece preview
        preview_x = self.game_x + GAME_WIDTH - BLOCK_SIZE * 6
        preview_y = SCREEN_HEIGHT - BLOCK_SIZE * 10

        # Draw preview box
        shapes.Rectangle(preview_x - BLOCK_SIZE,
                         preview_y - BLOCK_SIZE,
                         BLOCK_SIZE * 4,
                         BLOCK_SIZE * 4,
                         color=GRID_COLOR).draw()

        next_color = COLORS[self.next_piece]
        next_shape = SHAPES[self.next_piece]
        for block in next_shape:
            x = preview_x + block[0] * BLOCK_SIZE
            y = preview_y + block[1] * BLOCK_SIZE
            self.draw_block(x, y, next_color)

        # Draw UI elements
        self.title_label.draw()
        self.score_label.draw()
        self.lines_label.draw()
        self.next_label.draw()

        if self.game_over:
            loser_label = pyglet.text.Label(f'Game Over!',
                                            x=self.game_x + GAME_WIDTH//2,
                                            y=SCREEN_HEIGHT//2,
                                            anchor_x='center',
                                            anchor_y='center',
                                            font_size=BLOCK_SIZE,
                                            bold=True,
                                            color=(255, 0, 0, 255))
            loser_label.draw()

    def draw_block(self, x, y, color):
        """Draw a single block"""
        # Draw block shadow
        shapes.Rectangle(x + 2, y - 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4,
                         color=(0, 0, 0, 50)).draw()

        # Draw main block
        shapes.Rectangle(x, y, BLOCK_SIZE - 4, BLOCK_SIZE - 4,
                         color=color).draw()

    def _get_heights(self):
        """Get the height of each column"""
        heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] is not None:
                    heights.append(GRID_HEIGHT - y)
                    break
            else:
                heights.append(0)
        return heights

    def lock_piece(self):
        """Lock the current piece in place safely"""
        if self.current_piece is None or self.current_shape is None:
            return

        try:
            for block in self.current_shape:
                x = self.current_pos[0] + block[0]
                y = self.current_pos[1] + block[1]
                if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                    self.grid[y][x] = self.current_piece

            self.clear_lines()
            self.spawn_new_piece()
        except Exception as e:
            print(f"Error in lock_piece: {e}")
            self.spawn_new_piece()

class TetrisGame(Window):
    def __init__(self, enable_ai=False, ai_type="genetic", ai_weights_file=None, training_mode=False):
        """Initialize game window
        Args:
            enable_ai (bool): Whether to enable AI for player 2
            ai_type (str): Type of AI to use ("genetic" or "dqn")
            ai_weights_file (str): Path to the AI weights file
            training_mode (bool): Whether the game is in training mode (no window)
        """
        if not training_mode:
            super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, caption='AI Battle Tetris', fullscreen=False)
            pyglet.gl.glClearColor(*(c/255.0 for c in BG_COLOR), 1)
            print(f"Window created: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

        # Store game settings
        self.enable_ai = enable_ai
        self.ai_type = ai_type
        self.ai_weights_file = ai_weights_file
        self.training_mode = training_mode

        # Initialize game state
        self.initialize_game()

        if not training_mode:
            self.base_interval = 0.1
            pyglet.clock.schedule_interval(self.update, self.base_interval)

    def initialize_game(self):
        """Initialize or reset game state"""
        # Create players
        self.player1 = PlayerState(GAME1_X, True, self, is_ai=False)
        self.player2 = PlayerState(GAME2_X, False, self, is_ai=self.enable_ai)

        if self.enable_ai:
            if self.ai_type == "genetic" and self.ai_weights_file:
                weights = TetrisAIWeights(self.ai_weights_file)
                self.player2.ai = TetrisAI(weights)
                print(f"Genetic AI weights loaded from {self.ai_weights_file}")
            elif self.ai_type == "dqn":
                self.player2.ai = TetrisAI(TetrisAIWeights())
                print("Initialized for DQN training")

        print(f"Players initialized (Player 2 AI: {self.enable_ai}, Type: {self.ai_type})")

    def update(self, dt):
        """Update game state"""
        try:
            self.player1.update(dt)
            self.player2.update(dt)
        except Exception as e:
            print(f"Error in update: {e}")

    def on_draw(self):
        """Draw the game state"""
        try:
            # Clear the window
            self.clear()

            # Draw background
            background = shapes.Rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT,
                                          color=BG_COLOR)
            background.draw()

            # Draw both players' states
            self.player1.draw()
            self.player2.draw()

            # Draw game over overlay if needed
            if self.player1.game_over and self.player2.game_over:
                self._draw_game_over()

        except Exception as e:
            print(f"Error in on_draw: {e}")

    def _draw_game_over(self):
        """Draw game over screen"""
        try:
            # Draw semi-transparent overlay
            overlay = shapes.Rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT,
                                       color=(0, 0, 0, 128))
            overlay.opacity = 128
            overlay.draw()

            # Determine winner
            if self.player1.game_over and not self.player2.game_over:
                winner = "Player 2"
            elif self.player2.game_over and not self.player1.game_over:
                winner = "Player 1"
            elif self.player1.score > self.player2.score:
                winner = "Player 1"
            elif self.player2.score > self.player1.score:
                winner = "Player 2"
            else:
                winner = "Tie"

            # Draw game over text
            game_over_label = pyglet.text.Label('Game Over!',
                                                x=SCREEN_WIDTH//2,
                                                y=SCREEN_HEIGHT//2 + 40,
                                                anchor_x='center',
                                                anchor_y='center',
                                                font_size=BLOCK_SIZE,
                                                bold=True,
                                                color=(255, 255, 255, 255))
            game_over_label.draw()

            winner_label = pyglet.text.Label(f'Winner: {winner}',
                                             x=SCREEN_WIDTH//2,
                                             y=SCREEN_HEIGHT//2 - 40,
                                             anchor_x='center',
                                             anchor_y='center',
                                             font_size=int(BLOCK_SIZE * 0.6),
                                             color=(200, 200, 200, 255))
            winner_label.draw()

            retry_label = pyglet.text.Label('Press R to Restart',
                                            x=SCREEN_WIDTH//2,
                                            y=SCREEN_HEIGHT//2 - 120,
                                            anchor_x='center',
                                            anchor_y='center',
                                            font_size=int(BLOCK_SIZE * 0.5),
                                            color=(200, 200, 200, 255))
            retry_label.draw()

        except Exception as e:
            print(f"Error in _draw_game_over: {e}")

    def on_key_press(self, symbol, modifiers):
        """Handle keyboard input"""
        try:
            if self.player1.game_over and self.player2.game_over:
                if symbol == key.R:
                    # Reset game state without creating new window
                    self.initialize_game()
                    return

            # Player 1 controls (Arrow keys)
            if not self.player1.game_over:
                if symbol == key.A:
                    new_pos = [self.player1.current_pos[0] - 1, self.player1.current_pos[1]]
                    if self.player1.is_valid_move(self.player1.current_shape, new_pos):
                        self.player1.current_pos = new_pos

                elif symbol == key.D:
                    new_pos = [self.player1.current_pos[0] + 1, self.player1.current_pos[1]]
                    if self.player1.is_valid_move(self.player1.current_shape, new_pos):
                        self.player1.current_pos = new_pos

                elif symbol == key.S:
                    self.player1.move_down()

                elif symbol == key.W:
                    self.player1.rotate_shape()

                elif symbol == key.SPACE:
                    while self.player1.move_down():
                        pass

            # Player 2 controls (WASD + E)
            if not self.player2.game_over and not self.player2.is_ai:
                if symbol == key.A:
                    new_pos = [self.player2.current_pos[0] - 1, self.player2.current_pos[1]]
                    if self.player2.is_valid_move(self.player2.current_shape, new_pos):
                        self.player2.current_pos = new_pos

                elif symbol == key.D:
                    new_pos = [self.player2.current_pos[0] + 1, self.player2.current_pos[1]]
                    if self.player2.is_valid_move(self.player2.current_shape, new_pos):
                        self.player2.current_pos = new_pos

                elif symbol == key.S:
                    self.player2.move_down()

                elif symbol == key.W:
                    self.player2.rotate_shape()

                elif symbol == key.E:
                    while self.player2.move_down():
                        pass

            if symbol == key.ESCAPE:
                self.close()

        except Exception as e:
            print(f"Error in key press handler: {e}")
if __name__ == '__main__':
    try:
        print("Starting game...")
        # Specify the path to your trained weights file
        game = TetrisGame(enable_ai=True, ai_weights_file="best_tetris_weights_final.txt")
        print("Game instance created, starting main loop...")
        pyglet.app.run()
    except Exception as e:
        print(f"Error running game: {e}")
        import traceback
        traceback.print_exc()