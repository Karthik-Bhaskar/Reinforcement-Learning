from copy import deepcopy
import numpy as np
import random
from IPython.display import clear_output
import time
import os


class State:
    
    def __init__(self, grid, agent_pos):
        self.grid = grid
        self.agent_pos = agent_pos
        
    def __eq__(self, other):
        return isinstance(other, State) and self.grid == other.grid and self.agent_pos == other.agent_pos
    
    def __hash__(self):
        return hash(str(self.grid) + str(self.agent_pos))
    
    def __str__(self):
        return f"State(grid={self.grid}, agent_pos={self.agent_pos})"



Agent = "A"
Tresure = "T"
Poison = "P"
EMPTY = "."
Wardrobe = "W"

#4x4
grid1 = [
    [EMPTY,EMPTY,EMPTY,Agent],
    [EMPTY,EMPTY,Wardrobe,EMPTY],
    [Poison,EMPTY,EMPTY,EMPTY],
    [EMPTY,EMPTY,Tresure,EMPTY]
]

#5x5
grid2 = [[EMPTY, EMPTY, EMPTY, Wardrobe, Agent],
                                  [EMPTY, EMPTY, Wardrobe, EMPTY, EMPTY],
                                  [Wardrobe, EMPTY, EMPTY, EMPTY, Wardrobe],
                                  [EMPTY, EMPTY, Poison, EMPTY, EMPTY],
                                  [Poison, EMPTY, EMPTY, EMPTY, Tresure]]



for row in grid1:
    print(' '.join(row))

up = 0
down = 1
left = 2
right =3

actions = [up, down, left, right]

env = State(grid=grid1, agent_pos=[0,3])

def play(state,actions):

    def new_agent_pos(state, action):
        pos = deepcopy(state.agent_pos)
        if action == up:
            pos[0] = max(0, pos[0] - 1)
        elif action == down:
            pos[0] = min(len(state.grid) - 1, pos[0] + 1)
        elif action == left:
            pos[1] = max(0, pos[1] - 1)
        elif action == right:
            pos[1] = min(len(state.grid[0]) - 1, pos[1] + 1)
        else:
            raise ValueError(f"Unknown action {action}")
        return pos
    
    pos = new_agent_pos(state,action)
    grid_item = state.grid[pos[0]][pos[1]]
    new_grid = deepcopy(state.grid)

    if grid_item == Poison:
        reward = -10
        done = True
        old_pos = state.agent_pos
        new_grid[old_pos[0]][old_pos[1]] = EMPTY
        new_grid[pos[0]][pos[1]] += Agent

    elif grid_item == Tresure:
        reward = 10
        done = True
        old_pos = state.agent_pos
        new_grid[old_pos[0]][old_pos[1]] = EMPTY
        new_grid[pos[0]][pos[1]] += Agent

    elif grid_item == EMPTY:
        reward = -1
        done = False
        old_pos = state.agent_pos
        new_grid[old_pos[0]][old_pos[1]] = EMPTY
        new_grid[pos[0]][pos[1]] = Agent

    elif grid_item == Agent:
        reward = -1
        done = False

    elif grid_item == Wardrobe:
        reward = -8
        done = True # It was False
        old_pos = state.agent_pos
        new_grid[old_pos[0]][old_pos[1]] = Wardrobe 
        new_grid[pos[0]][pos[1]] = Agent

    return State(grid=new_grid, agent_pos=pos), reward, done, new_grid



episodes = 30
time_steps = 100
alpha_min = 0.02
alpha_ = np.linspace(1.0,alpha_min,episodes)
gamma = 1
epsilon = 0.1
Qtable = dict()



def Q(state, action=None):

	if state not in Qtable:
		Qtable[state] = np.zeros(len(actions))

	if action is None:
		return Qtable[state]

	return Qtable[state][action]

def select_action(state):
	if random.uniform(0,1) < epsilon:
		return random.choice(actions)
	else:
		return np.argmax(Q(state))



for episode in range(episodes):
    state = env
    total_reward = 0
    alpha = alpha_[episode]


    for time_step in range(time_steps):
        action = select_action(state)
        next_state, reward, done, new_grid = play(state,action)
        total_reward += reward
        for i in new_grid:
            print(' '.join(i))
        time.sleep(0.3)
        target = reward + gamma * np.max(Q(next_state))
        Q(state)[action] = Q(state,action) + alpha * (target - Q(state,action))
        state = next_state


        if done:
        	print(f"Episode {episode+1} reward: {total_reward}")
        	
        	break
        
        print(f"Episode {episode+1} reward {total_reward}")

        time.sleep(0.3)
        clear_output()
        os.system('cls'if os.name == 'nt' else 'clear')









 


