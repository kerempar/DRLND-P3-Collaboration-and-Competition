from ddpg import DDPGAgent
from buffer import ReplayBuffer

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
NOISE_SCALE = 2.0       # initial noise scale
NOISE_DECAY = 0.999     # noise decay 

class MADDPG:
    """Interacts with the environment and orchectrates its DDGP agents."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        super(MADDPG, self).__init__()
        
        self.num_agents = num_agents
        self.ddpg_agents = []
        
        # create DDGP agents
        for i in range(num_agents):
            self.ddpg_agents.append(DDPGAgent(state_size, action_size, random_seed))
        
        # create common replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # initialize noise scale
        self.noise_scale = NOISE_SCALE

    def reset(self):
        for i in range(self.num_agents):
            self.ddpg_agents[i].reset()       

    def act(self, states):
        """get actions from all agents in the MADDPG object"""
        # update noise scale to be applied to actions
        self.noise_scale *= NOISE_DECAY  
        actions = [agent.act(state, self.noise_scale) for agent, state in zip(self.ddpg_agents, states)]
        return actions

    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for i in range(self.num_agents):
                experiences = self.memory.sample() 
                self.ddpg_agents[i].learn(experiences, GAMMA)


