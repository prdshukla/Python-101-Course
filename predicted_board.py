import numpy as np 
import pandas as pd 
import random
import tensorflow as tf
from kaggle_environments import make
from kaggle_environments import evaluate, make, utils
import os
Loading environment lux_ai_s2 failed: No module named 'vec_noise'

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, reward):
        experience = (state, reward)
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self):
        state, reward = zip(*self.buffer)
        return np.stack(state), np.array(reward)
def train_agent(obs):
    
    training = True
    
    board = obs.board
    
    if obs.mark == 1:
        replacer = {0:0, 1:1, 2:-1}
    elif obs.mark == 2:
        replacer = {0:0, 1:-1, 2:1}
        
    board = [replacer[i] for i in board]

    return agent.act(board, training)


def test_agent(obs):
    
    training = False
    
    board = obs.board
    
    if obs.mark == 1:
        replacer = {0:0, 1:1, 2:-1}
    elif obs.mark == 2:
        replacer = {0:0, 1:-1, 2:1}
        
    board = [replacer[i] for i in board]

    return agent.act(board, training)
def make_model():
    import tensorflow as tf
    import keras
    from keras.models import Sequential
    from keras.layers import Conv2D, Flatten, Dense, Dropout
    from keras import regularizers
    

    model = Sequential()
    input_shape=(6, 7, 1)

    model.add(Conv2D(32, kernel_size=(2, 2),activation='relu', input_shape=input_shape,
                    activity_regularizer=regularizers.L2(1e-5)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))

    opt = keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(loss='mean_squared_error', optimizer=opt)
       
    return model
checkpoint_path = 'training_5/checkpoint.ckpt'

checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                verbose=0)

model = make_model()
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 5, 6, 32)          160       
                                                                 
 flatten (Flatten)           (None, 960)               0         
                                                                 
 dense (Dense)               (None, 128)               123008    
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 200)               25800     
                                                                 
 dropout_1 (Dropout)         (None, 200)               0         
                                                                 
 dense_2 (Dense)             (None, 128)               25728     
                                                                 
 dropout_2 (Dropout)         (None, 128)               0         
                                                                 
 dense_3 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 174,825
Trainable params: 174,825
Non-trainable params: 0
_________________________________________________________________
env = make("connectx", {"rows": 6, "columns": 7, "inarow": 4})
def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))
class MLAgent():
    def __init__(self, epsilon = 1, epsilon_decay = 0.99, min_epsilon = 0.01, action_dim=7):
        self.network = model
        self.replay_buffer = ReplayBuffer(500)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim
    
    
    def get_board_values(self, board: list()) -> list():
        board_values = [0,0,0,0,0,0,0]
        all_possible_boards = []

        for i in range(0,7):
            print(f"i: {i}")
            for j in range(0,6):
                index = 41 - 7*j - 6+i
                num = board[index]
                print(num)
                if num == 0:
                    print(index)
                

                    possible_board = board.copy()
                    possible_board[index] = 1
                    all_possible_boards.append(possible_board)
                    break

            else:
                all_possible_boards.append(board)
                board_values[i] = -100
                print(f"invalid row {i}")
                
            print()
        
        all_possible_boards = np.reshape(all_possible_boards, (-1,6,7))

        score = self.network.predict(all_possible_boards, verbose = False)
        
        for i in range(7):
            board_values[i] += score[i]

        return board_values
        

    def act(self, state, training):
        
        state = list(state)
        
        illegal_moves = state[:7]
        
        # no available moves
        if illegal_moves.count(0) == 0:
            return 0
        
        if training: 
            # in training randomly takes a random move
            # and the first 2 moves are always random, this helps to cut repitition
            # and it shouldn't advantage either of the algos too much
            # this is disabled when not training
            if (np.random.random() < self.epsilon) or state.count(0) in [41,42]:
                while True:
                    action = np.random.choice(self.action_dim)
                    if illegal_moves[action] == 0:
                        return action             
        

        board_values = self.get_board_values(state)

        while True:     
            action = np.argmax(board_values)
            action = int(action)

            if illegal_moves[action] == 0:
                return action
            else:
                # if max value is illegal alction, we won't take it
                # we will instead get the 2.nd best value
                board_values[action] = -10

    
    def update(self):

        state, reward = self.replay_buffer.sample()

        # we retrain the model with the new rewards that were nudged in the direction that they lead the game in to
        # model is also save via the callbacks
        #reward = np.array(reward)
        
        self.network.fit(state, reward, verbose = False, callbacks = [checkpoint])
        
     
    def train(self,episodes=100000, visualize=True):
        
        for episode in range(episodes):
            for game in range(50):
                env.reset()
                
                # the agent randomly plays against itself and negamax
                # also negamax and train_agent can both be the starters
                # some games against a random target since that helps it
                # not become too narrow
                if game % 6 == 0:
                    agents = [train_agent, "random"]
                else:
                    agents = [train_agent,train_agent, "negamax"]
                
                random.shuffle(agents)
                playing_agents = agents[:2]
                
                steps = env.run(playing_agents)
                
                first_agent_score = 0
                second_agent_score = 0
                
                end_board = steps[-1][0]['observation']["board"]
                first_agent_next_obs = end_board
                second_agent_next_obs = end_board
                            

                # if board is full, both lose
                if end_board.count(0) == 0:
                    first_agent_score = -1
                    second_agent_score = -1
                else:
                    # winner and loser reward is colleced
                    first_agent_score = steps[-1][0]["reward"]
                    second_agent_score = steps[-1][1]["reward"]
                    
                first_agent_rewards = [first_agent_score]
                second_agent_rewards = [second_agent_score]
      
                # normalizing the obs to be the same for both players point of view
                # 1 = you and -1 =  enemy  
                replacer = {0:0, 1:1, 2:-1}
                first_agent_next_obs = [replacer[i] for i in first_agent_next_obs]  

                replacer = {0:0, 1:-1, 2:1}
                second_agent_next_obs = [replacer[i] for i in second_agent_next_obs]
                
                
                # reshaped into a 2d numpy array for keras CNN
                first_agent_next_obs = np.reshape(first_agent_next_obs, (-1,6,7))
                second_agent_next_obs = np.reshape(second_agent_next_obs, (-1,6,7))
                
                # boards and their rewards pushed into a memory buffer
                for i in list(zip(first_agent_next_obs, first_agent_rewards)):
                    self.replay_buffer.push(*i)
                for i in list(zip(second_agent_next_obs, second_agent_rewards)):
                    self.replay_buffer.push(*i)
            
            
            # every 100 training cycles we visualize the most recent board
            if episode % 10 == 0:
                print(f"episode {episode} is done")
                print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", ["random", test_agent], num_episodes=100)))
                if visualize:
                    env.render(mode="ipython")
 
            self.update()
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        return
# if no checkpoint saved, we make a new agent
try:
    agent = MLAgent(epsilon = 0.05, epsilon_decay = 0.99, min_epsilon = 0.02)
    agent.network.load_weights(checkpoint_path)
    print("model loaded")    
    
except Exception as e:
    agent = MLAgent(epsilon = 0.1, epsilon_decay = 0.99, min_epsilon = 0.02)
    print(e)
Unsuccessful TensorSliceReader constructor: Failed to find any matching files for training_5/checkpoint.ckpt
#agent.train(episodes = 100000, visualize = False)
#print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", ["random", test_agent], num_episodes=100)))
'''env.reset()
env.run(["random",test_agent])
env.render(mode="ipython")'''
'env.reset()\nenv.run(["random",test_agent])\nenv.render(mode="ipython")'
#print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", ["random", test_agent], num_episodes=100)))
#print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", ["negamax", test_agent], num_episodes=100)))
import codecs
import pickle

obj = agent.network.get_weights()

# the pickled replaces the string in encoded_model
pickled = codecs.encode(pickle.dumps(obj), "base64").decode()
# this is the model decoded into a string

#pickled
# it is really janky but there is no other way to really do this, kaggle doesn't provide a way to add a model to the output file
# and the output file needs to be self contained, so the model is encoded into a string
# and the string is decoded back to a keras model within the return file
# also the support functions need to be included

# you also only encode the weights, but this is simpler because i dont need to redefine the model
# within the return agent, but the string will be longer this way

def return_agent(obs):
    import tensorflow as tf
    import keras
    import codecs
    import pickle
    import numpy as np 

    
    board = obs.board
    if obs.mark == 1:
        replacer = {0:0, 1:1, 2:-1}
    else:
        replacer = {0:0, 1:-1, 2:1}
        
    board = [replacer[i] for i in board]
    
    
    def make_model():
        import tensorflow as tf
        import keras
        from keras.models import Sequential
        from keras.layers import Conv2D, Flatten, Dense, Dropout
        from keras import regularizers


        model = Sequential()
        input_shape=(6, 7, 1)

        model.add(Conv2D(32, kernel_size=(2, 2),activation='relu', input_shape=input_shape,
                        activity_regularizer=regularizers.L2(1e-5)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='tanh'))

        opt = keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(loss='mean_squared_error', optimizer=opt)

        return model

    model = make_model()
    
    encoded_weights = "token"   
    
    
    weights = pickle.loads(codecs.decode(encoded_weights.encode(), "base64"))
    
    model.set_weights(weights)
    
    
    # generates all the possible boards the agent can go into from the current board
    # then the agent evaluates the board value and adds all values to a list
    # then it makes a move based on what board has the most value
    def get_board_values(board: list()) -> list():
        board_values = [0,0,0,0,0,0,0]
        opponent_board_values = [0,0,0,0,0,0,0]
        all_possible_boards = []

        for i in range(0,7):
            for j in range(0,6):
                index = 41 - 7*j - 6+i
                num = board[index]
                if num == 0:  

                    possible_board = board.copy()
                    possible_board[index] = 1
                    
                    all_enemy_values = get_opponent_board_values(possible_board)
                    opponent_board_values[i] = max(all_enemy_values)
                    
                    all_possible_boards.append(possible_board)
                    break

            else:
                all_possible_boards.append(board)
                board_values[i] = -100
        
        all_possible_boards = np.reshape(all_possible_boards, (-1,6,7))

        score = model.predict(all_possible_boards, verbose = False)
        
        for i in range(7):
            board_values[i] += (score[i] - opponent_board_values[i])

        return board_values
    
    
    def get_opponent_board_values(original_board: list()) -> list():
        board_values = [0,0,0,0,0,0,0]
        all_possible_boards = []
        board = original_board.copy()
        replacer = {0:0, 1:-1, -1:1}
        board = [replacer[i] for i in board]    

        for i in range(0,7):
            for j in range(0,6):
                index = 41 - 7*j - 6+i
                num = board[index]
                if num == 0:  

                    possible_board = board.copy()
                    possible_board[index] = 1
                    all_possible_boards.append(possible_board)
                    break

            else:
                all_possible_boards.append(board)
                board_values[i] = 0
        
        all_possible_boards = np.reshape(all_possible_boards, (-1,6,7))

        score = model.predict(all_possible_boards, verbose = False)
        
        for i in range(7):
            board_values[i] += score[i]

        return board_values
    
    
    def act(state):
        
        state = list(state)
        
        # top row of the board
        # if the row is occupied (not 0)
        # then the move is illegal
        illegal_moves = state[:7]
        
        # no available moves
        if illegal_moves.count(0) == 0:
            return 0

        board_values = get_board_values(state)

        while True:     
            action = np.argmax(board_values)
            action = int(action)

            if illegal_moves[action] == 0:
                return action
            else:
                # if max value is illegal alction, we won't take it
                # we will instead get the 2.nd best value
                board_values[action] = -10
    

    return act(board)
#print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [return_agent, "negamax"], num_episodes=100)))

import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(return_agent, "submission.py")
<function return_agent at 0x7f7cb46f4170> written to submission.py