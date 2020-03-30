import pygame
import numpy as np
import random
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import pandas as pd
from operator import add
from collections import deque
import sys

#Most of the  code has been taken from Maurocks's github repo as a learn't from there.

class Agent:
    def __init__(self,lr,maxlen,load_weights=True):
        self.reward=0
        self.gamma=0.9
        self.short_memory=[]
        self.agent_target=0
        self.learning_rate=lr
        self.epsilon=1
        self.memory=deque(maxlen=maxlen)
        self.load_weights=load_weights
        self.weights_path='./weights/weights.hdf5'
        self.dqn_model=self.build_model()
    def build_model(self):
        model=Sequential()
        model.add(Dense(output_dim=150,activation='relu',input_dim=11))
        model.add(Dense(output_dim=150,activation='relu'))
        model.add(Dense(output_dim=150,activation='relu'))
        model.add(Dense(output_dim=3,activation='softmax'))
        model.compile(loss='mse',optimizer=Adam(self.learning_rate))
        if self.load_weights:
            model.load_weights(self.weights_path)
        return model

    def get_state(self, game, player, food):
        state = [
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add, player.position[-1], [20, 0])) in player.position) or
            player.position[-1][0] + 20 >= (game.game_width - 20))) or (player.x_change == -20 and player.y_change == 0 and ((list(map(add, player.position[-1], [-20, 0])) in player.position) or
            player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1], [0, -20])) in player.position) or
            player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1], [0, 20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))),  # danger straight

            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
            player.position[ -1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
            [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
            add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
             -1] + 20 >= (game.game_height-20))),  # danger right

             (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
             player.position[-1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
             add, player.position[-1],[-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
            player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))), #danger left


            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
            ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)


    def set_reward(self,player,crash):
        self.reward=0
        if crash:
            self.reward=-10
            return self.reward
        if player.eaten:
            self.reward=10
        return self.reward

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def replay_new(self,memory,batch_size):
        if len(memory)>batch_size:
            minibatch=random.sample(memory,batch_size)
        else:
            minibatch=memory
        for state,action,reward,next_state,done in minibatch:
            target=reward
            if not done:
                target=reward+self.gamma*np.amax(self.dqn_model.predict(np.array([next_state]))[0])
            target_f=self.dqn_model.predict(np.array([state]))
            target_f[0][np.argmax(action)]=target
            self.dqn_model.fit(np.array([state]),target_f,epochs=1,verbose=0)
    def train_short_memory(self,state,action,reward,next_state,done):
        target=reward
        if not done:
            target=reward+self.gamma*np.amax(self.dqn_model.predict(next_state.reshape((1,11)))[0])
        target_f=self.dqn_model.predict(state.reshape(1,11))
        target_f[0][np.argmax(action)]=target
        self.dqn_model.fit(state.reshape((1,11)),target_f,epochs=1,verbose=0)

#Deep Q Agent implementing Deep Reinforcement learning.

class Game_class:
    def __init__(self,game_width,game_height):
        pygame.display.set_caption('SnakeGame')
        self.game_width=game_width
        self.game_height=game_height
        self.gameDisplay=pygame.display.set_mode((game_width,game_height+60))
        self.bg=pygame.image.load("img/background.png")
        self.crash=False
        self.player=Player(self)
        self.food=Food()
        self.score = 0

class Player(object):
    def __init__(self,game):
        x=0.45*game.game_width
        y=0.5*game.game_height
        self.x=x-x%20
        self.y=y-y%20
        self.position=[]
        self.position.append([self.x,self.y])
        self.food=1
        self.eaten=False
        self.image=pygame.image.load('img/snakeBody.png')
        self.x_change=20
        self.y_change=0
    def update_position(self,x,y):
        if self.position[-1][0]!=x or self.position[-1][1]!=y:
            if self.food>1:
                for i in range(0,self.food-1):
                    self.position[i][0],self.position[i][1]=self.position[i+1]
            self.position[-1][0]=x
            self.position[-1][1]=y
    def do_move(self,move,x,y,game,food,agent):
        move_array=[self.x_change,self.y_change]

        if self.eaten:
            self.position.append([self.x,self.y])
            self.eaten=False
            self.food=self.food+1
        if np.array_equal(move,[1,0,0]):
            move_array=self.x_change,self.y_change
        elif np.array_equal(move,[0,1,0]) and self.y_change==0:
            move_array=[0,self.x_change]
        elif np.array_equal(move,[0,1,0]) and self.x_change==0:
            move_array=[-self.y_change,0]
        elif np.array_equal(move,[0,0,1]) and self.y_change==0:
            move_array=[0,-self.x_change]
        elif np.array_equal(move,[0,0,1]) and self.x_change==0:
            move_array=[self.y_change,0]
        self.x_change,self.y_change=move_array
        self.x=x+self.x_change
        self.y=y+self.y_change

        if self.x<20 or self.x>game.game_width-40 or self.y<20 or self.y>game.game_height-40 or [self.x,self.y] in self.position:
            game.crash=True
        eat(self,food,game)

        self.update_position(self.x,self.y)

    def display_player(self,x,y,food,game):
        self.position[-1][0]=x
        self.position[-1][1]=y

        if game.crash==False:
            for i in range(food):
                x_temp,y_temp=self.position[len(self.position)-1-i]
                game.gameDisplay.blit(self.image,(x_temp,y_temp))
            pygame.display.update()
        else:
            pygame.time.wait(300)
class Food:
    def __init__(self):
        self.x_food=240
        self.y_food=200
        self.image=pygame.image.load('img/food.jpg')

    def food_coord(self,game,player):
        x_rand=random.randint(20,game.game_width-40)
        self.x_food=x_rand-x_rand%20
        y_rand=random.randint(20,game.game_height-40)
        self.y_food=y_rand-y_rand%20
        if [self.x_food,self.y_food] not in player.position:
            return self.x_food,self.y_food
        else:
            self.food_coord(game,player)
    def display_food(self,x,y,game):
        game.gameDisplay.blit(self.image,(x,y))
        pygame.display.update()

def eat(player,food,game):
    if player.x==food.x_food and player.y==food.y_food:
        food.food_coord(game,player)
        player.eaten=True
        game.score=game.score+1

def get_record(score,record):
    if score>=record:
        return score
    else:
        return record

def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))

def display(player,food,game,record):
    game.gameDisplay.fill((255,255,255))
    display_ui(game,game.score,record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)

def initialize_game(player, game, food, agent, batch_size):
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)


pygame.init()
agent=Agent(lr=0.0005,maxlen=2500,load_weights=True)
if agent.load_weights:
    agent.dqn_model.load_weights(agent.weights_path)
    print("Weights loaded")
n_game_played=0
n_episodes=300
batch_size=500
display_option=True
record=0
training=False
epsilon_decay_linear=1/75
speed=60


while n_game_played<n_episodes:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
    game=Game_class(440,440)
    player1=game.player
    food1=game.food

    initialize_game(player1,game,food1,agent,batch_size)
    if display_option:
        display(player1,food1,game,record)
    while not game.crash:
        if not training:
            agent.epsilon=0
        else:
            agent.epsilon=1-(n_game_played*epsilon_decay_linear)
        state_old=agent.get_state(game,player1,food1)

        if random.randint(0,1)<agent.epsilon:
            final_move=to_categorical(random.randint(0,2),num_classes=3)
        else:
            prediction=agent.dqn_model.predict(state_old.reshape((1,11)))
            final_move=to_categorical(np.argmax(prediction[0]),num_classes=3)
        player1.do_move(final_move,player1.x,player1.y,game,food1,agent)
        state_new=agent.get_state(game,player1,food1)

        reward=agent.set_reward(player1,game.crash)

        if training:
             agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
             agent.remember(state_old, final_move, reward, state_new, game.crash)

        record = get_record(game.score, record)
        if display_option:
                display(player1, food1, game, record)
                pygame.time.wait(speed)

    if training:
        agent.replay_new(agent.memory,batch_size)
        n_game_played+=1
        print(f"Game {n_game_played}")
    if training:
        agent.dqn_model.save_weights(agent.weights_path)
