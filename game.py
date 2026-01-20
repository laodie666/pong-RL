import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame
import random

screen_h = 150
screen_w = 300

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc1(x)

class Player():
    
    def __init__(self, side):
        self.side = side
    
    def getAction(self):
        pass

class heuristicPlayer(Player):
        
    # side == 0 for left, 1 for right
    def __init__(self, side):
        self.side = side
        
        
    # 1 or 2 to go down or up, 0 to stay still
    def getAction(self, game: "PongGame"):
        
        self.paddle_position = game.lpaddle_position + game.paddle_dimensions[1]/2 if self.side == 0 else game.rpaddle_position + game.paddle_dimensions[1]/2
        
        # print(self.paddle_position, game.ball_position[1])
  
        if self.paddle_position < game.ball_position[1]:
            return 1
        else:
            return 2
        
def uncertain_value(value, error_margin):
    return value + random.randint(-error_margin, error_margin)

class PongGame():
    
    # Paddle is at position 10 and w-10
    paddle_dimensions = (1,20)
    paddle_move_increment = 10

    
    def __init__(self, p1: Player, p2:Player):
        self.ball_position = [uncertain_value(screen_w/2, 10), uncertain_value(screen_h/2, 10)]
        self.ball_velocity = [uncertain_value(15, 5),uncertain_value(5, 2)]
        
        self.lpaddle_position = screen_h/2
        self.rpaddle_position = screen_h/2 
        self.ball_radius = 5
        self.p1 = p1
        self.p2 = p2
        self.reward = [0,0]
        self.bounce_counter = 0
        
    def set_screen(self, screen):
        self.screen = screen
    
    def get_state(self):
        return np.array([
            self.ball_position[0] / screen_w,       
            self.ball_position[1] / screen_h,       
            self.ball_velocity[0] / 20,       
            self.ball_velocity[1] / 20,         
            self.lpaddle_position / screen_h,        
            self.rpaddle_position / screen_h
        ], dtype=np.float32)
        
    def getScreenDimension (self):
        return (screen_w, screen_h)
        
    def move_paddle(self, move, side):

        if move == 1:
            if side == 0:
                self.lpaddle_position += self.paddle_move_increment
            else:
                self.rpaddle_position += self.paddle_move_increment
        elif move == 2:
            if side == 0:
                self.lpaddle_position -= self.paddle_move_increment
            else:
                self.rpaddle_position -= self.paddle_move_increment  

         
    
    def step (self):
        # Update ball position
        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[1] += self.ball_velocity[1] 

        # Update paddle position
        p1_action = self.p1.getAction(self)
        p2_action = self.p2.getAction(self)
        
        if torch.is_tensor(p1_action):
            p1_action = p1_action.item() 
        
        if torch.is_tensor(p2_action):
            p2_action = p2_action.item()
        
        
        # left paddle
        self.move_paddle(p1_action, 0)
        self.lpaddle_position = max(0, min(screen_h - self.paddle_dimensions[1], self.lpaddle_position))
        
        # right paddle
        self.move_paddle(p2_action, 1)
        self.rpaddle_position = max(0, min(screen_h - self.paddle_dimensions[1], self.rpaddle_position))

        # Check collisions with walls
        if self.ball_position[1] <= 0 or self.ball_position[1] >= screen_h:
            self.ball_velocity[1] = -self.ball_velocity[1]   
            self.ball_position[1] = max(0, min(screen_h, self.ball_position[1]))
            self.bounce_counter += 1
            if self.bounce_counter % 3 == 0:
                self.ball_velocity[1] *= 1.2
                self.ball_velocity[1] = int(self.ball_velocity[0]+0.5) 
        
        # Check collisions with paddles
        if (self.ball_position[0] - self.ball_radius <= 10 and self.lpaddle_position <= self.ball_position[1] <= self.lpaddle_position + self.paddle_dimensions[1]):
            self.ball_velocity[0] = -self.ball_velocity[0]   
            self.ball_position[0] = max(11, min(screen_w-11, self.ball_position[0]))
            self.bounce_counter += 1
            self.reward[0] += 0.05
            if self.bounce_counter % 3 == 0:
                self.ball_velocity[0] *= 1.2
                self.ball_velocity[0] = int(self.ball_velocity[0]+0.5) 
            
        if(self.ball_position[0] + self.ball_radius >= screen_w-10 and self.rpaddle_position <= self.ball_position[1] <= self.rpaddle_position + self.paddle_dimensions[1]):
            self.ball_velocity[0] = -self.ball_velocity[0]   
            self.ball_position[0] = max(11, min(screen_w-11, self.ball_position[0]))
            self.bounce_counter += 1
            self.reward[1] += 0.05
            if self.bounce_counter % 3 == 0:
                self.ball_velocity[0] *= 1.2
                self.ball_velocity[0] = int(self.ball_velocity[0]+0.5) 
            
        
        actions = [p1_action, p2_action]
        done = 0
        win = -1
        
        
        if self.ball_position[0] <= 0:
            done = 1
            win = 0
            self.reward[0] -= abs(self.lpaddle_position-self.ball_position[1])/screen_h
            self.reward[0] -= 1
            self.reward[1] += 1
              
        elif self.ball_position[0] >= screen_w:
            done = 1
            win = 1
            self.reward[0] += 1
            self.reward[1] -= abs(self.rpaddle_position-self.ball_position[1])/screen_h
            self.reward[1] -= 1
        
        
        # print(actions, done, win, self.reward)
        return (actions, done, win, self.reward)
            
            
    def display(self):
        self.screen.fill((0,0,1))
        pygame.draw.rect(self.screen, (255,0,0), ((10,self.lpaddle_position), self.paddle_dimensions))
        pygame.draw.rect(self.screen, (255,0,0), ((screen_w-10,self.rpaddle_position), self.paddle_dimensions))
        pygame.draw.circle(self.screen, (255,255,0), self.ball_position, self.ball_radius)
        pygame.display.flip()
        

class NNPlayer(Player):
        
    # side == 0 for left, 1 for right
    def __init__(self, side, policy:PolicyNetwork):
        self.side = side
        self.policy = policy
        
        
    # 1 or 2 to go down or up, 0 to stay still
    def getAction(self, game: "PongGame"):
        state_tensor = torch.from_numpy(game.get_state()).unsqueeze(0)
        probs = self.policy(state_tensor)
        self.m = torch.distributions.Categorical(probs)
        action = self.m.sample()
        
        return action
    
    def getProbabilities(self, game:"PongGame", action):
        
        return self.m.log_prob(action)    

def train(p2: str, checkpoint = None):
    policy = PolicyNetwork(6, 64, 3)
    if checkpoint is not None:
        print("loaded")
        state_dict = torch.load(checkpoint)
        policy.load_state_dict(state_dict)
        policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    episodes = 10000
    
    pygame.init()
    clock = pygame.time.Clock()
    
    
    for episode in range(episodes):
        print(episode)

        if p2 == "heuristic":   
            p2_model = heuristicPlayer(1)
        else: 
            p2_model = NNPlayer(0, policy)
        p1 = NNPlayer(0, policy)
        game = PongGame(p1, p2_model)
        
        log_probs = []
        rewards = []

        entropy_term = 0

        actions, done, win, reward = game.step()
        log_prob = p1.getProbabilities(game, torch.tensor(actions[0]))
        log_probs.append(log_prob)
        p = torch.exp(log_prob)
        entropy = -p * log_prob
        entropy_term += entropy
        
        while done != 1:
            if episode > 1000 and (episode % 200 == 0 or reward[0] > 10):
                screen = pygame.display.set_mode((screen_w, screen_h))
                game.set_screen(screen)
                game.display()
                clock.tick(30)
                
            actions, done, win, reward = game.step()
            log_prob = p1.getProbabilities(game, torch.tensor(actions[0]))
            log_probs.append(log_prob)
            p = torch.exp(log_prob)
            entropy = -p * log_prob
            entropy_term += entropy
            
        
        rewards = [reward[0]]*len(log_probs)
        rewards = torch.tensor(rewards)
        loss = []
        # print(reward)
        for log_prob, R in zip(log_probs, rewards):
            loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        loss = torch.stack(loss).sum() - (0.01 * entropy_term)
        loss.backward()
        optimizer.step()
    
    torch.save(policy.state_dict(), f"checkpoint{episode}.pt")

        

    
if __name__ == "__main__":
    
    train("heuristic", "checkpoint9999.pt")

    