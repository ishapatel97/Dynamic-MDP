import random
from math import floor

states = []

start_state = (0,9,2)

UAV_state = {}

UAV_state[0] = start_state

goal_state = []


# Create 20*20*5 grid world
for i in range(20):
    for j in range(20):
        for k in range(5):
            states.append((i,j,k))

            if i == 19:
                goal_state.append((i,j,k))

# List all 5 actions( Forward, Left, Right, Up, Down)
actions = ['F', 'L', 'R', 'U', 'D']

###----------------------------------------------------------------------------------------------------###

# Specify and Initialize Bird Parameters

Bird_num = 100 #50

Bird_real_state = {}

Bird_real_location = {}

Bird_real_vel = {}

bird_start_state = []

bird_start_vel_real = []


for i in range(Bird_num):

    x = random.randint(5,19)
    y = random.randint(0,19)
    z = random.randint(0,2) 

    u = random.uniform(0.4, 0.8)
    v = random.uniform(-0.6, 0.6)
    w = 0 #random.uniform(-0.2, 0.2) 

    bird_start_state.append((x,y,z))
    bird_start_vel_real.append((u,v,w))
   

Bird_real_state[0] = bird_start_state

Bird_real_location[0] = bird_start_state


### Assuming constant bird velocity throughout its motion ###

for i in range(20):
    Bird_real_vel[i] = bird_start_vel_real


###----------------------------------------------------------------------------------------------------###

# Define Bird State Tansition Model

def Bird_Transition(old_bird_loc, old_bird_vel):
    
    loc_update = []
    state_update = []

    for bd in range(Bird_num):
        new_x = old_bird_loc[bd][0] - old_bird_vel[bd][0]
        new_y = old_bird_loc[bd][1] - old_bird_vel[bd][1]
        new_z = old_bird_loc[bd][2] - old_bird_vel[bd][2]
        
        loc_update.append((new_x, new_y, new_z))
        state_update.append((floor(new_x), floor(new_y), floor(new_z)))

    return (loc_update, state_update)


###----------------------------------------------------------------------------------------------------###

# Predict Next Bird State 

loc_obs_error = 0.1
vel_obs_error = 0.1

def Predict_Next_Bird_State(old_bird_loc, old_bird_vel):

    predicted_state = []
    for bd in range(Bird_num):
        predicted_x = random.uniform(old_bird_loc[bd][0] - loc_obs_error, old_bird_loc[bd][0] + loc_obs_error) - random.uniform(old_bird_vel[bd][0] - vel_obs_error, old_bird_vel[bd][0] + vel_obs_error)
        predicted_y = random.uniform(old_bird_loc[bd][1] - loc_obs_error, old_bird_loc[bd][1] + loc_obs_error) - random.uniform(old_bird_vel[bd][1] - vel_obs_error, old_bird_vel[bd][1] + vel_obs_error)
        predicted_z = old_bird_loc[bd][2] #random.uniform(old_bird_loc[bd][2] - loc_obs_error, old_bird_loc[bd][2] + loc_obs_error) - random.uniform(old_bird_vel[bd][2] - vel_obs_error, old_bird_vel[bd][2] + vel_obs_error)

        predicted_state.append((floor(predicted_x), floor(predicted_y), floor(predicted_z)))

    return predicted_state


###----------------------------------------------------------------------------------------------------###

def Reward(UAV_current, action, UAV_next, bird_current, bird_predict):

    price = []

    if UAV_current not in goal_state and UAV_next in goal_state:
        price.append(100)
    
    if UAV_current in goal_state:
        price.append(0)
    
    if UAV_next in bird_predict:
        price.append(-100)
    
    if UAV_current in bird_current:
        price.append(-100)
    
    if action == 'F':
        if UAV_next[1] - UAV_current[1] == 0:
            price.append(-1)
        else:
            price.append(-2)
        
    if action == 'L' or action == 'R':
        if UAV_next[1] - UAV_current[1] == 0:
            price.append(-3)
        else:
            price.append(-2)

    if action == 'U':
        if UAV_next[2] - UAV_current[2] == 0:
            price.append(-7)
        else:
            price.append(-5)

    if action == 'D':
        if UAV_next[2] - UAV_current[2] == 0:
            price.append(-7)
        else:
            price.append(-2)

    return sum(price)


###----------------------------------------------------------------------------------------------------###

# Define a Transition function that returns 3 dimensional tuple containing 3 new states with first element having the highest probability

def UAV_Transition(UAV_current, action):
    
    if action == 'F':
        if UAV_current[0]<19:
            new_state_1 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])
        
        if UAV_current[0] == 19:
            new_state_1 = UAV_current
        
        # Forward Left
        if UAV_current[0]<19:
            if UAV_current[1]<19:
                new_state_2 = (UAV_current[0]+1, UAV_current[1]+1, UAV_current[2])
            if UAV_current[1] == 19:
                new_state_2 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])

        if UAV_current[0] == 19:
            new_state_2 = UAV_current

        # Forward Right
        if UAV_current[0]<19:
            if UAV_current[1]>0:
                new_state_3 = (UAV_current[0]+1, UAV_current[1]-1, UAV_current[2])
            if UAV_current[1] == 0:
                new_state_3 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])

        if UAV_current[0] == 19:
            new_state_3 = UAV_current
    #--------------------------------------------------------------------------------------------------#

    if action == 'L':
        
        if UAV_current[0]<19:
            if UAV_current[1]<19:
                new_state_1 = (UAV_current[0]+1, UAV_current[1]+1, UAV_current[2])
            if UAV_current[1] == 19:
                new_state_1 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])

        if UAV_current[0] == 19:
            new_state_1 = UAV_current
        
        # Forward 
        if UAV_current[0]<19:
            new_state_2 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])
                
        if UAV_current[0] == 19:
            new_state_2 = UAV_current

        new_state_3 = new_state_2

    #--------------------------------------------------------------------------------------------------#

    if action == 'R':
        
        if UAV_current[0]<19:
            if UAV_current[1]>0:
                new_state_1 = (UAV_current[0]+1, UAV_current[1]-1, UAV_current[2])
            if UAV_current[1] == 0:
                new_state_1 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])

        if UAV_current[0] == 19:
            new_state_1 = UAV_current
        
        # Forward 
        if UAV_current[0]<19:
            new_state_2 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])
                
        if UAV_current[0] == 19:
            new_state_2 = UAV_current

        new_state_3 = new_state_2

    #--------------------------------------------------------------------------------------------------#

    if action == 'U':
        
        if UAV_current[0]<19:
            if UAV_current[2]<4:
                new_state_1 = (UAV_current[0]+1, UAV_current[1], UAV_current[2]+1)
            if UAV_current[2] == 4:
                new_state_1 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])

        if UAV_current[0] == 19:
            new_state_1 = UAV_current
        
        # Forward 
        if UAV_current[0]<19:
            new_state_2 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])
                
        if UAV_current[0] == 19:
            new_state_2 = UAV_current

        new_state_3 = new_state_2

    #--------------------------------------------------------------------------------------------------#

    if action == 'D':
        
        if UAV_current[0]<19:
            if UAV_current[2]>0:
                new_state_1 = (UAV_current[0]+1, UAV_current[1], UAV_current[2]-1)
            if UAV_current[2] == 0:
                new_state_1 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])

        if UAV_current[0] == 19:
            new_state_1 = UAV_current
        
        # Forward 
        if UAV_current[0]<19:
            new_state_2 = (UAV_current[0]+1, UAV_current[1], UAV_current[2])
                
        if UAV_current[0] == 19:
            new_state_2 = UAV_current

        new_state_3 = new_state_2

    return (new_state_1, new_state_2, new_state_3)


###----------------------------------------------------------------------------------------------------###
###----------------------------------------------------------------------------------------------------###
###----------------------------------------------------------------------------------------------------###


def Policy_Iteration(t):
    policy = {}
    Value = {}

    predicted_bird_state = Predict_Next_Bird_State(Bird_real_location[t], Bird_real_vel[t])

    # Initialize Policy and Value Function
    for state in states:
        policy[state] = 'F'
        Value[state] = 0

    # Immediate Expected Reward
    def q_value(state):
        x = UAV_Transition(state, policy[state])
        return 0.98*Reward(state, policy[state], x[0], Bird_real_state[t], predicted_bird_state) + 0.01*Reward(state, policy[state], x[1], Bird_real_state[t], predicted_bird_state) + 0.01*Reward(state, policy[state], x[2], Bird_real_state[t], predicted_bird_state)
    
    gamma = 0.99
    policy_evaluation_threshold = 0.5
    Value_previous = {}
    
    def Policy_Evaluation():
        for state in states:
            Value_previous[state] = Value[state]
            Value[state] = q_value(state) + gamma*(0.98*Value[UAV_Transition(state, policy[state])[0]] + 0.01*Value[UAV_Transition(state, policy[state])[1]] + 0.01*Value[UAV_Transition(state, policy[state])[2]])

    Policy_Evaluation()

    for state in states:
        while abs(Value[state] - Value_previous[state]) > policy_evaluation_threshold:
            Policy_Evaluation()

    # Returns Updated Policy
    def Policy_Improvement():

        count = 0
        for state in states:
            opt_pol = {}
            for action in actions:
                y = UAV_Transition(state, action)
                opt_pol[action] = 0.98*(Reward(state, action, y[0], Bird_real_state[t], predicted_bird_state) + gamma*Value[y[0]]) + 0.01*(Reward(state, action, y[1], Bird_real_state[t], predicted_bird_state) + gamma*Value[y[1]]) + 0.01*(Reward(state, action, y[2], Bird_real_state[t], predicted_bird_state) + gamma*Value[y[2]])
            a = list(opt_pol.values())
            b = list(opt_pol.keys())
           
            opt_action = b[a.index(max(a))]
            if opt_action != policy[state]:
                count = count + 1
            policy[state] = opt_action

        if count == 0:
            return False
        
        else:
            return True
        
    
    while Policy_Improvement():
        
        Policy_Evaluation()

        for state in states:
            while abs(Value[state] - Value_previous[state]) > policy_evaluation_threshold:
                Policy_Evaluation()

    return policy


cumulative_policy = {}
cumulative_policy[0] = Policy_Iteration(0)

time = 1

while time<20:

    bt = Bird_Transition(Bird_real_location[time-1], Bird_real_vel[time - 1]) 
    Bird_real_location[time] = bt[0]
    Bird_real_state[time] = bt[1]
    cumulative_policy[time] = Policy_Iteration(time)
    time = time + 1


###----------------------------------------------------------------------------------------------------###

# Determine the UAV Trajectory in the Dynamic Environment

t = 0
while t<20:
    print("-------------------", t, "-------------------")
    print()
    print('UAV  State : ', UAV_state[t], ' ; ', 'Policy   --->   ', cumulative_policy[t][UAV_state[t]])
    print()
    print('Bird States :- ', Bird_real_state[t])
    t = t+1
    UAV_state[t] = UAV_Transition(UAV_state[t-1], cumulative_policy[t-1][UAV_state[t-1]])[0]
    print()
    print()