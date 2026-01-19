import sys
import numpy as np
import pygame

class Player():
    
    def getAction(self):
        pass

class heuristicPlayer(Player):
        
    # side == 0 for left, 1 for right
    def __init__(self, side):
        self.side = side
        
        
    # 1 or -1 to go down or up, 0 to stay still
    def getAction(self, game: "PongGame"):
        self.paddle_position = game.lpaddle_position + game.paddle_dimensions[1]/2 if self.side == 0 else game.rpaddle_position + game.paddle_dimensions[1]/2
        
        # print(self.paddle_position, game.ball_position[1])
  
        if self.paddle_position < game.ball_position[1]:
            return 1
        else:
            return -1
        

class PongGame():
    
    # Paddle is at position 10 and w-10
    paddle_dimensions = (1,20)
    paddle_move_increment = 10
    h = 150
    w = 300
    
    def __init__(self, p1: Player, p2:Player):
        self.ball_position = [self.w/2, self.h/2]
        self.ball_velocity = [15,5]
        self.lpaddle_position = self.h/2
        self.rpaddle_position = self.h/2 
        self.ball_radius = 5
        self.p1 = p1
        self.p2 = p2
        
    def getScreenDimension (self):
        return (self.w, self.h)
        
    def move_paddle(self, move, side):
        if move == 1:
            if side == 0:
                self.lpaddle_position += self.paddle_move_increment
            else:
                self.rpaddle_position += self.paddle_move_increment
        elif move == -1:
            if side == 0:
                self.lpaddle_position -= self.paddle_move_increment
            else:
                self.rpaddle_position -= self.paddle_move_increment   
    
    def getNextState (self):
        
        # Update ball position
        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[1] += self.ball_velocity[1] 

        # Update paddle position
        p1_action = self.p1.getAction(self)
        p2_action = self.p2.getAction(self)
        self.move_paddle(p1_action, 0)
        self.move_paddle(p2_action, 1)

        # print(p1_action, p2_action)

        # Check collisions with walls
        if self.ball_position[1] <= 0 or self.ball_position[1] >= self.h:
            self.ball_velocity[1] = -self.ball_velocity[1]   
            self.ball_position[1] = max(0, min(self.h, self.ball_position[1]))
        
        # Check collisions with paddles
        
        if (self.ball_position[0] - self.ball_radius <= 10 and self.lpaddle_position <= self.ball_position[1] <= self.lpaddle_position + self.paddle_dimensions[1]) or\
            (self.ball_position[0] + self.ball_radius >= self.w-10 and self.rpaddle_position <= self.ball_position[1] <= self.rpaddle_position + self.paddle_dimensions[1]):
                self.ball_velocity[0] = -self.ball_velocity[0]   
                self.ball_position[0] = max(11, min(self.w-11, self.ball_position[0]))
        
        elif self.ball_position[0] - self.ball_radius <= 10:
            print(self.lpaddle_position, self.ball_position[1], self.lpaddle_position + self.paddle_dimensions[1])
            print("right wins")
            pygame.quit()
            sys.exit()

        elif self.ball_position[0] + self.ball_radius >= self.w-10:
            print(self.rpaddle_position, self.ball_position[1], self.rpaddle_position + self.paddle_dimensions[1])
            print("left wins")
            pygame.quit()
            sys.exit()

        
        
    
    def display(self, screen):
        pygame.draw.rect(screen, (255,0,0), ((10,self.lpaddle_position), self.paddle_dimensions))
        pygame.draw.rect(screen, (255,0,0), ((self.w-10,self.lpaddle_position), self.paddle_dimensions))
        pygame.draw.circle(screen, (255,255,0), self.ball_position, self.ball_radius)
        

    
if __name__ == "__main__":
    pygame.init()
    p1 = heuristicPlayer(0)
    p2 = heuristicPlayer(1)
    game = PongGame(p1, p2)
    screen = pygame.display.set_mode(game.getScreenDimension())
    pygame.display.set_caption("Pygame Display Example")
    clock = pygame.time.Clock()

    running = True
    while running:
        # 5. Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Drawing code goes here
        # Fill the background with a color
        screen.fill((0,0,1))
        game.display(screen)
        # 6. Update the display
        pygame.display.flip() # Updates the entire screen

        game.getNextState()

        # Cap the frame rate
        clock.tick(15)


 
        
    
    
    