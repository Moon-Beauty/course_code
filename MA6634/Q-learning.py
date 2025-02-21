import numpy as np
import random
# Gridworld setup
grid_size=3
states=[(i,j) for i in range(grid_size)
        for j in range(grid_size)]
# print(states)
actions=['up','down','left','right']
rewards=np.full((grid_size,grid_size),-1) #Reward of -1 for each step

# Step penalty
rewards[2,2]=10 #Goal state reward
gamma=0.9 # Discount factor
alpha=0.1 #Learning rate
epsilon=0.1 #Exploration rate
episodes=5000 # Number of episodes

# Transition function
def next_state(state,action):
    i,j=state
    if action=='up':
        return max(0,i-1),j
    elif action=='down':
        return min(grid_size-1,i+1),j
    elif action=='left':
        return i,max(0,j-1)
    elif action=='right':
        return i,min(grid_size-1,j+1);

# Q-learning
def q_learning():
    Q={state:{action:0 for action in actions} for state in states}

    for episode in range(episodes):
        # start at a random state
        state=random.choice(states)
        while state!=(2,2): # Until the goal is reached
              # choose action using epsilon-greedy strategy
              if random.random()<epsilon:
                 action=random.choice(actions) # Explore
              else:
                 action=max(Q[state],key=Q[state].get) # Exploit
              # Take action, observe next state and reward
              next_state_=next_state(state,action)
              reward=rewards[next_state_]
              # Update Q-value
              Q[state][action]+=alpha*(
                      reward+gamma*max(Q[next_state_].values())-Q[state][action]
              )

              # Move to next state
              state=next_state_
        return Q

#Derive Optimal Policy
def extract_policy(Q):
    policy={}
    for state in Q:
        policy[state]=max(Q[state],key=Q[state].get)
    return policy

# Run Q-learning
Q_values=q_learning()
optimal_policy=extract_policy(Q_values)

# Display results
print("Optimal Q-values:")
for state,actions in Q_values.items():
    print(f"State: {state} | Actions: {actions}")
print("\nOptimal policy:")
for state,action in optimal_policy.items():
    print(f"State: {state} | Action: {action}")