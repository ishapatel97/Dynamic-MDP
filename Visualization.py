from Dynamic_MDP import UAV_state, Bird_real_state

import pygame
import time

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define grid size
WIDTH = 20
HEIGHT = 20
MARGIN = 5

# Initialize Pygame
pygame.init()

# Set the size of the window
WINDOW_SIZE = [(WIDTH + MARGIN) * 20 + MARGIN, (HEIGHT + MARGIN) * 20 + MARGIN]
screen = pygame.display.set_mode(WINDOW_SIZE)

# Set the title of the window
pygame.display.set_caption("UAV and Birds")

# Define the state of the UAV and birds

uv = {}

for t in range(20):
    uv[t] = (UAV_state[t][0], UAV_state[t][1])


bd = {}

for t in range(20):
    lst = []
    for pac in Bird_real_state[t]:
        lst.append((pac[0],pac[1]))
    bd[t] = lst

uav_state = uv
bird_state = bd

# Define the current time step
current_time_step = 0

# Define a function to draw the grid
def draw_grid():
    for row in range(20):
        for column in range(20):
            color = GRAY
            if (row, column) in bird_state[current_time_step]:
                color = RED
            if (row, column) == uav_state[current_time_step]:
                color = GREEN
            pygame.draw.rect(screen, color, [(MARGIN + WIDTH) * column + MARGIN,
                                             (MARGIN + HEIGHT) * row + MARGIN,
                                             WIDTH,
                                             HEIGHT])

# Run the game loop
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Clear the screen
    screen.fill(WHITE)

    # Draw the grid
    draw_grid()

    # Update the current time step
    current_time_step += 1

    # Check if we have reached the end of the simulation
    if current_time_step >= len(uav_state):
        current_time_step = 0

    # Update the screen
    pygame.display.flip()

    # Wait for 1 second
    time.sleep(1)

# Quit Pygame
pygame.quit()