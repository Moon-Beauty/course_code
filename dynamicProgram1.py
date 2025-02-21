import numpy as np

# Gridworld setup
grid_size=3
states=[(i,j) for i in range(grid_size)
        for j in range(grid_size)]
# print(states)
actions=['up','down','left','right']
rewards=np.full((grid_size,grid_size),-1) #Reward of -1 for each step
rewards[2,2]=10 #Goal state reward
gamma=0.9 # Discount factor
transition_prob=0.8
epsilon=1e-4 #convergence threshold

# Transition function
def next_state_reward(state,action):
    i,j=state
    if action=='up':
        return max(0,i-1),j
    elif action=='down':
        return min(grid_size-1,i+1),j
    elif action=='left':
        return i,max(0,j-1)
    elif action=='right':
        return i,min(grid_size-1,j+1);

# Value iteration
def value_iteration():
    V=np.zeros((grid_size,grid_size))
    while True:
        delta=0
        new_V=np.copy(V)
        for state in states:
            i,j=state
            max_value=float('-inf')
            for action in actions:
                next_i,next_j=next_state_reward(state,action)
                value=(transition_prob*(rewards[next_i,next_j]+
                gamma*V[next_i,next_j]))+((1-transition_prob)*(rewards[i,j]
                +gamma*V[i,j]))
                max_value=max(max_value,value)
            new_V[i,j]=max_value
            delta=max(delta,abs(V[i,j]-new_V[i,j]))
        V=new_V
        if delta<epsilon:
            break
        return V

# Optimal policy extraction
def extract_policy(V):
    policy={}
    for state in states:
        i,j=state
        best_action=None
        max_value=float('-inf')
        for action in actions:
            next_i,next_j=next_state_reward(state,action)
            value=(transition_prob*(rewards[next_i,next_j]+
            gamma*V[next_i,next_j]))+((1-transition_prob)*(rewards[i,j]+
            gamma*V[i,j]))
        if value>max_value:
            max_value=value
            best_action=action
        policy[state]=best_action
    return policy

# Run value iteration
optimal_values=value_iteration()
optimal_policy=extract_policy(optimal_values)

# Display Results
print("optimal value Function:")
print(optimal_values)
print("\nOptimal Policy:")
for state,action in optimal_policy.items():
    print(f"State: {state} Action: {action}")