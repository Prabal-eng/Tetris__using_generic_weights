Here is a unified README.md file documenting both the game executable (t.py) and the training system (trainer.py).

AI Battle Tetris
This project consists of two main components: a fully playable Tetris Game (t.py) featuring Human vs. AI capabilities, and a Genetic Algorithm Trainer (trainer.py) used to evolve and optimize the AI's strategy.

Part 1: The Game (t.py)
This file contains the visual game engine. It allows you to play Tetris normally, watch an AI play, or battle against an AI opponent. It features a graphical interface built with pyglet.

🎮 Features
Human vs. AI: Play locally against a computer opponent.

AI vs. AI: Watch two AI agents battle (requires code adjustment in main).

Customizable Weights: The AI's behavior is driven by a genetic algorithm, and you can load custom weight files to change its playstyle.

Visuals: Real-time rendering of the grid, score, next pieces, and "Game Over" states.

⌨️ Controls
Player 1 (Human):

Arrow Left/Right: Move Piece

Arrow Up: Rotate Piece

Arrow Down: Soft Drop

Space: Hard Drop (Instant Lock)

Player 2 (Human mode):

A/D: Move Piece

W: Rotate Piece

S: Soft Drop

E: Hard Drop

General:

R: Restart Game

ESC: Quit

🚀 How to Run
To start the game with the AI enabled for Player 2:

Bash

python t.py
Note: By default, the script looks for a weight file named best_tetris_weights_final.txt. If not found, it defaults to hardcoded standard weights.

Part 2: The Trainer (trainer.py)
This file is a "headless" (no graphics) simulation engine designed to train the AI. It uses a Genetic Algorithm to play thousands of games in the background, evolving the best possible weights for the AI to use in t.py.

🧠 How it Works
Population: It creates a population of 100 random AI agents.

Simulation: Each agent plays multiple games using a matrix-based engine (faster than the visual game).

Evolution: The best agents are selected to "breed" and mutate, creating a better next generation.

Checkpointing: The best weights are saved periodically to the checkpoints/ folder.

⚙️ Configuration
You can adjust training parameters inside the main() function of trainer.py:

POPULATION_SIZE: Number of agents per generation (Default: 100).

GENERATIONS: Total generations to run (Default: 500).

GAMES_PER_GENOME: Games played per agent to calculate average score (Default: 5).

🚀 How to Run
Bash

python trainer.py
The script will output progress bars and statistics to the console. When finished, it saves the best model to checkpoints/final/best_weights_final.pkl.

🔗 Connecting the Trainer to the Game
The trainer produces raw data and Pickle files (.pkl), but the game (t.py) reads a simple text file format (.txt). Here is how to use your trained AI in the actual game:

Run trainer.py until completion or until you are satisfied with the fitness score.

Look at the console output from the trainer. It will print the best weights in this format:

Plaintext

holes: -0.7654
bumpiness: -0.3210
total_height: -0.1543
...
Copy that list of weights.

Create a new file named best_tetris_weights_final.txt in the same folder as t.py.

Paste the weights into that file.

Run python t.py. The game will now load your trained "Brain" into the AI player.
