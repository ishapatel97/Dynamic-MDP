#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aptil 28 21:40:50 2023

@author: ishapatel
"""
import gym
from gym import spaces
import numpy as np
import random
import pygame

class BirdEnv(gym.Env):
    def __init__(self, num_birds=10, grid_size=20, goal_state=(19, 19)):
        # define state space and action space
        self.observation_space = spaces.Tuple((spaces.Discrete(grid_size**2), spaces.Box(low=-10, high=10, shape=(num_birds, 4))))
        self.action_space = spaces.Discrete(5)

        # define transition model
        self.transition_probs = {
            0: {"forward": 0.98, "left": 0.01, "right": 0.01, "lift up": 0.00, "lift down": 0.00},
            1: {"forward": 0.95, "left": 0.025, "right": 0.025, "lift up": 0.00, "lift down": 0.00},
            2: {"forward": 0.95, "left": 0.025, "right": 0.025, "lift up": 0.00, "lift down": 0.00},
            3: {"forward": 0.90, "left": 0.05, "right": 0.05, "lift up": 0.025, "lift down": 0.025},
            4: {"forward": 0.90, "left": 0.05, "right": 0.05, "lift up": 0.025, "lift down": 0.025},
            5: {"forward": 0.98, "left": 0.01, "right": 0.01, "lift up": 0.00, "lift down": 0.00},
            6: {"forward": 0.95, "left": 0.025, "right": 0.025, "lift up": 0.00, "lift down": 0.00},
            7: {"forward": 0.95, "left": 0.025, "right": 0.025, "lift up": 0.00, "lift down": 0.00},
            8: {"forward": 0.90, "left": 0.05, "right": 0.05, "lift up": 0.025, "lift down": 0.025},
            9: {"forward": 0.90, "left": 0.05, "right": 0.05, "lift up": 0.025, "lift down": 0.025}
        }
        self.directions = {"forward": 0, "left": -1, "right": 1, "lift up": 1, "lift down": -1}

        # define reward function
        self.X = -0.1
        self.goal_state = goal_state
        
        # initialize UAV and bird locations and velocities
        self.grid_size = grid_size
        self.uav_loc = (0, 0)
        self.bird_locs = np.zeros((num_birds, 2))
        self.bird_vels = np.zeros((num_birds, 2))
        
        # initialize reward
        self.reward = 0
    
    def step(self, state, action):
        direction = list(self.directions.keys())[list(self.directions.values()).index(action)]
        state_key = int(state[0]) if not isinstance(state[0], int) else state[0]
        if state_key not in self.transition_probs:
            raise ValueError(f"Invalid state: {state}")
        prob = self.transition_probs[state_key][direction]
        if direction == "forward":
            next_loc = (state[0] + 1, state[1])
        elif direction == "left":
            next_loc = (state[0] + 1, state[1] - 1)
        elif direction == "right":
            next_loc = (state[0] + 1, state[1] + 1)
        elif direction == "lift up":
            next_loc = (state[0] + 1, state[1])
            next_birds = self.bird_locs + np.array([0, 0, self.directions[direction], 0])
            self.bird_locs = next_birds[:, :2]
        elif direction == "lift down":
            next_loc = (state[0] + 1, state[1])
            next_birds = self.bird_locs + np.array([0, 0, self.directions[direction], 0])
            self.bird_locs = next_birds[:, :2]
        else:
            raise ValueError("Invalid action")
        # check if the agent collides with a bird
        if np.any(np.abs(self.bird_locs[:, :2] - np.array([next_loc[0], next_loc[1]])) < 1e-6):
            reward = -100
            done = True
        else:
            # calculate reward
            if direction == "forward":
                if next_loc == self.goal_state:
                    reward = 100
                    done = True
                else:
                    reward = self.X + self.calc_forward_reward(state, self.bird_locs) # X + FORWARD
                    done = False
            elif direction == "left":
                reward = self.X + self.calc_left_reward(state, self.bird_locs) - self.calc_forward_reward(state, self.bird_locs) # X + LEFT - FORWARD
                done = False
            elif direction == "right":
                reward = self.X + self.calc_right_reward(state, self.bird_locs) - self.calc_forward_reward(state, self.bird_locs) # X + RIGHT - FORWARD
                done = False
            elif direction == "lift up":
                reward = self.X + self.calc_upper_reward(state, self.bird_locs) - self.calc_forward_reward(state, self.bird_locs) # X + UPPER - FORWARD
                done = False
            elif direction == "lift down":
                reward = self.X + self.calc_lower_reward(state, self.bird_locs) - self.calc_forward_reward(state, self.bird_locs) # X + DOWN
                done = False
            else:
                raise ValueError("Invalid action")
        # update state
        if direction == "lift up" or direction == "lift down":
            self.bird_locs = next_birds
        self.uav_loc = next_loc
        self.bird_locs = self.update_birds()
        # check if the episode is done
        if self.uav_loc == self.goal_state or done:
            done = True
        else:
            done = False
        # return next state, reward, and done
        return (self.uav_loc, self.bird_locs, reward, done)

    def update_birds(self, bird_locs, bird_vels):
        for i in range(len(bird_locs)):
            probs = list(self.transition_probs[i].values())
            prob_sum = sum(probs)
            probs_normalized = [p / prob_sum for p in probs]
            directions = list(self.directions.values())
            new_vel = np.array([np.random.choice(directions, p=probs_normalized), np.random.choice(directions, p=probs_normalized)])
            new_loc = bird_locs[i] + new_vel
            if new_loc[0] < 0 or new_loc[0] >= self.grid_size:
                new_loc[0] = bird_locs[i][0]
                new_vel[0] = -new_vel[0]
            if new_loc[1] < 0 or new_loc[1] >= self.grid_size:
                new_loc[1] = bird_locs[i][1]
                new_vel[1] = -new_vel[1]
            bird_locs[i] = new_loc
            bird_vels[i] = new_vel
        return bird_locs, bird_vels


    def reset(self):
        # initialize UAV location
        self.uav_loc = (1, 1)

        # initialize bird locations and velocities
        for i in range(self.bird_locs.shape[0]):
            self.bird_locs[i] = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            self.bird_vels[i] = (random.uniform(-10, 10), random.uniform(-10, 10))

        # return initial state
        return (self.uav_loc[0] * self.grid_size + self.uav_loc[1], np.hstack((self.bird_locs, self.bird_vels)))
    
    def policy_iteration(env, gamma=0.99, max_iterations=1000):
        V = np.zeros(env.observation_space.spaces[0].n) # initialize value function to zeros
        policy = np.zeros((env.observation_space.spaces[0].n, env.action_space.n)) # initialize policy to a random deterministic policy
        for state in range(env.observation_space.spaces[0].n):
            random_action = np.random.randint(env.action_space.n)
            policy[state][random_action] = 1

        for i in range(max_iterations):
            # Policy evaluation
            theta = 0.0001
            while True:
                delta = 0
                for state in range(env.observation_space.spaces[0].n):
                    v = V[state]
                    new_v = 0
                    for action in range(env.action_space.n):
                        new_state, reward, done, _ = env.step(state, action)
                        new_v += policy[state][action] * (reward + gamma * V[new_state])
                    V[state] = new_v
                    delta = max(delta, abs(v - V[state]))
                if delta < theta:
                    break

            # Policy improvement
            policy_stable = True
            for state in range(env.observation_space.spaces[0].n):
                old_action = np.argmax(policy[state])
                action_values = np.zeros(env.action_space.n)
                for action in range(env.action_space.n):
                    new_state, reward, done, _ = env.step(action)
                    action_values[action] = reward + gamma * V[new_state]
                best_action = np.argmax(action_values)
                if old_action != best_action:
                    policy_stable = False
                policy[state] = np.eye(env.action_space.n)[best_action]

            if policy_stable:
                break

        return policy, V
    
    def render(self, mode='human'):
        if mode == 'human':
            # initialize Pygame
            pygame.init()

            # set up the screen
            screen_size = (self.grid_size * 30, self.grid_size * 30)
            screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption("Bird-UAV Environment")

            # set up colors
            white = (255, 255, 255)
            black = (0, 0, 0)
            red = (255, 0, 0)
            blue = (0, 0, 255)
            green = (0, 255, 0)

            # set up font
            font = pygame.font.Font(None, 30)

            # render the environment
            while True:
                # handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # update the birds
                self.update_birds(self.bird_locs, self.bird_vels)

                # clear the screen
                screen.fill(white)

                # draw the grid lines
                for i in range(self.grid_size):
                    pygame.draw.line(screen, black, (0, i * 30), (screen_size[0], i * 30))
                    pygame.draw.line(screen, black, (i * 30, 0), (i * 30, screen_size[1]))

                # draw the birds
                for i in range(self.bird_locs.shape[0]):
                    bird_color = red if np.abs(self.bird_vels[i][0]) > np.abs(self.bird_vels[i][1]) else blue
                    bird_pos = (int(self.bird_locs[i][1] * 30), int(self.bird_locs[i][0] * 30))
                    #pygame.draw.circle(screen, bird_color, bird_pos, 10)
                    
                    if bird_color == red:
                        bird_image = pygame.image.load('redbird.png')  # Load the bird image
                        bird_surface = pygame.Surface((20, 20))     # Create a surface for the bird image
                        bird_surface.set_colorkey((0, 0, 0))        # Set the color key for the surface
                        bird_surface.blit(bird_image, (0, 0))       # Blit the bird image onto the surface
                        #pygame.draw.circle(screen, bird_color, bird_pos, 10)   # Draw the circle
                        screen.blit(bird_surface, (bird_pos[0]-10, bird_pos[1]-10)) 
                    else:
                        bird_image = pygame.image.load('birdO.png')  # Load the bird image
                        bird_surface = pygame.Surface((20, 20))     # Create a surface for the bird image
                        bird_surface.set_colorkey((0, 0, 0))        # Set the color key for the surface
                        bird_surface.blit(bird_image, (0, 0))       # Blit the bird image onto the surface
                        #pygame.draw.circle(screen, bird_color, bird_pos, 10)   # Draw the circle
                        screen.blit(bird_surface, (bird_pos[0]-10, bird_pos[1]-10))  # Blit the bird surface onto the circle
                        
                    bird_loc = self.bird_locs[i]
                    bird_rect = pygame.Rect(bird_loc[1] * 30, bird_loc[0] * 30, 30.0, 30.0)
        
                    # determine color based on z altitude
                    z = bird_loc[1] #index must be 2
                    if z < -5:
                        color = (255, 0, 0)  # red
                    elif z < 0:
                        color = (255, 128, 0)  # orange
                    elif z == 0:
                        color = (255, 255, 255)  # white
                    elif z < 5:
                        color = (128, 128, 128)  # gray
                    else:
                        color = (0, 255, 0)  # green
                    #pygame.draw.rect(screen, color, bird_rect)


                # draw the UAV
                uav_pos = (int(self.uav_loc[1] * 30), int(self.uav_loc[0] * 30))
                #pygame.draw.circle(screen, green, uav_pos, 10)
                
                uav_image = pygame.image.load('uav.png')  # Load the bird image
                uav_surface = pygame.Surface((20, 20))     # Create a surface for the bird image
                uav_surface.set_colorkey((0, 0, 0))        # Set the color key for the surface
                uav_surface.blit(uav_image, (0, 0))       # Blit the bird image onto the surface
                screen.blit(uav_surface, (uav_pos[0]-10, uav_pos[1]-10)) 

                # draw the goal
                goal_pos = (int(self.goal_state[1] * 30), int(self.goal_state[0] * 30))
                #pygame.draw.circle(screen, black, goal_pos, 10)
                goal_image = pygame.image.load('goal.png')  # Load the bird image
                goal_surface = pygame.Surface((20, 20))     # Create a surface for the bird image
                goal_surface.set_colorkey((0, 0, 0))        # Set the color key for the surface
                goal_surface.blit(goal_image, (0, 0))       # Blit the bird image onto the surface
                screen.blit(goal_surface, (goal_pos[0]-10, goal_pos[1]-10)) 

                # render the state and reward
                # state_str = f"UAV location: {self.uav_loc}\nBird locations:\n{self.bird_locs}"
                # state_text = font.render(state_str, True, black)
                # screen.blit(state_text, (10, screen_size[1] - 90))

                reward_str = f"Reward: {self.reward:.2f}"
                reward_text = font.render(reward_str, True, black)
                screen.blit(reward_text, (10, screen_size[1] - 50))

                # update the screen
                pygame.display.flip()

                # wait for a short time
                pygame.time.wait(100)
        elif mode == 'console':
            # print the current state and reward to the console
            print(f"UAV location: {self.uav_loc}")
            print("Bird locations:")
            print(self.bird_locs)
            print(f"Reward: {self.reward:.2f}")
        else:
            super().render(mode=mode)
            
            
# Test Run
env = BirdEnv()
env.reset()
env.render()