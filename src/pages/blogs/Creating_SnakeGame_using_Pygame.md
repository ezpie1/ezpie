  ---
      layout: ../../layout/PostLayout.astro
      title: Creating an Engaging Snake Game with pygame
      description: Dive into the world of game development as we guide you through building a captivating Snake game using the pygame library. Learn the essentials of game design, graphics, and logic, all while having fun!
      published: September 30, 2023
      author: Tholkappiar M
      permalink: /blogs/creating-snake-game-pygame
      image: /images/thumbnail/snake-game.jpg
      ---

# Installation : 

```bash
pip install pygame
```
#### Pygame is a free and open-source cross-platform library for the development of multimedia applications like video games using Python. It uses the Simple DirectMedia Layer library and several other popular libraries to abstract the most common functions, making writing these programs a more intuitive task. 

# Section 1: Importing Libraries
```python
import pygame
import sys
import time
import random

```
#### Description: This section imports necessary Python libraries for the game:

pygame:  A library for creating 2D games.
sys: Provides functions and variables to interact with the Python runtime environment.
time: Used for time-related operations.
random: Allows you to generate random numbers.
# Section 2: Difficulty Settings
```python
difficulty = 25
```
#### Description: This variable sets the difficulty level of the game. In this case, it's set to medium (25).

# Section 3: Window Size
```python
frame_size_x = 720
frame_size_y = 480
```
#### Description: These variables define the size of the game window with a width of 720 pixels and a height of 480 pixels.
# Section 4: Error Checking
```python
check_errors = pygame.init()
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initializing the game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialized')
```
#### Description: Here, the code checks for any errors during the initialization of the pygame library. If errors are found, it prints an error message and exits the game. Otherwise, it prints a success message.
# Section 5: Initialize Game Window
```python
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))
```
#### Description: This code sets the game window's title to "Snake Eater" and initializes the game window with the specified size.
# Section 6: Define Colors
```python
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)
```
#### Description: These variables define colors that will be used in the game. They are represented as RGB values.
# Section 7: FPS (Frames Per Second) Controller
```python
fps_controller = pygame.time.Clock()
```
Description: This line creates an object to control the frame rate of the game.
# Section 8: Game Variables
```python
# Initialize snake position, body, and food
# Set initial direction, change direction, and initialize score
```
#### Description: This section initializes various game variables, including the snake's position and body, the food's position, the snake's direction, the direction the player intends to change to, and the score.
# Section 9: Game Over Function
```python
def game_over():
    # ...
```
#### Description: This function displays a game over message, waits for a few seconds, and then quits the game.
# Section 10: Score Display Function
```python
def show_score(choice, color, font, size):
    # ...
```
#### Description: This function displays the player's score on the screen.

# Section 11: Main Game Logic (Game Loop)
```python
while True:
    # Handle events, such as key presses or quitting the game
    # Update snake's direction and position
    # Check for collisions with food, boundaries, or itself
    # Draw the game elements
    # Update the display and control the frame rate
```
#### Description: This is the main game loop that continuously runs while the game is active. It handles various aspects of the game, such as user input, updating the snake's position, checking for collisions, drawing game elements, and controlling the frame rate.

# Section 12: Full Program (Game)

```
"""
Snake Eater
Made with PyGame
"""

import pygame, sys, time, random


# Difficulty settings
# Easy      ->  10
# Medium    ->  25
# Hard      ->  40
# Harder    ->  60
# Impossible->  120
difficulty = 25

# Window size
frame_size_x = 720
frame_size_y = 480

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')


# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))


# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()


# Game variables
snake_pos = [100, 50]
snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]

food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
food_spawn = True

direction = 'RIGHT'
change_to = direction

score = 0


# Game Over
def game_over():
    my_font = pygame.font.SysFont('times new roman', 90)
    game_over_surface = my_font.render('YOU DIED', True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_size_x/2, frame_size_y/4)
    game_window.fill(black)
    game_window.blit(game_over_surface, game_over_rect)
    show_score(0, red, 'times', 20)
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()
    sys.exit()


# Score
def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (frame_size_x/10, 15)
    else:
        score_rect.midtop = (frame_size_x/2, frame_size_y/1.25)
    game_window.blit(score_surface, score_rect)
    # pygame.display.flip()


# Main logic
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # Whenever a key is pressed down
        elif event.type == pygame.KEYDOWN:
            # W -> Up; S -> Down; A -> Left; D -> Right
            if event.key == pygame.K_UP or event.key == ord('w'):
                change_to = 'UP'
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                change_to = 'DOWN'
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                change_to = 'LEFT'
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                change_to = 'RIGHT'
            # Esc -> Create event to quit the game
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

    # Making sure the snake cannot move in the opposite direction instantaneously
    if change_to == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP':
        direction = 'DOWN'
    if change_to == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if change_to == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'

    # Moving the snake
    if direction == 'UP':
        snake_pos[1] -= 10
    if direction == 'DOWN':
        snake_pos[1] += 10
    if direction == 'LEFT':
        snake_pos[0] -= 10
    if direction == 'RIGHT':
        snake_pos[0] += 10

    # Snake body growing mechanism
    snake_body.insert(0, list(snake_pos))
    if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
        score += 1
        food_spawn = False
    else:
        snake_body.pop()

    # Spawning food on the screen
    if not food_spawn:
        food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
    food_spawn = True

    # GFX
    game_window.fill(black)
    for pos in snake_body:
        # Snake body
        # .draw.rect(play_surface, color, xy-coordinate)
        # xy-coordinate -> .Rect(x, y, size_x, size_y)
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

    # Snake food
    pygame.draw.rect(game_window, white, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

    # Game Over conditions
    # Getting out of bounds
    if snake_pos[0] < 0 or snake_pos[0] > frame_size_x-10:
        game_over()
    if snake_pos[1] < 0 or snake_pos[1] > frame_size_y-10:
        game_over()
    # Touching the snake body
    for block in snake_body[1:]:
        if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
            game_over()

    show_score(1, white, 'consolas', 20)
    # Refresh game screen
    pygame.display.update()
    # Refresh rate
    fps_controller.tick(difficulty)
```
## Conclusion

In this journey of creating a Snake game with pygame, we've explored the exciting world of game development. We started with importing libraries, defining game variables, and setting up the game window. As we delved deeper, we tackled challenges like handling user input, collision detection, and rendering graphics.

Thank you for joining us on this exciting journey. Stay tuned for more game development adventures and tech tutorials!
