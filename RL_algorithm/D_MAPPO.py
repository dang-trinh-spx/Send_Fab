import os
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import argparse
from data_preprocessing import line_product as action_masks
from data_preprocessing import product_num, layer_num, max_machine_num

#Initiate RL hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--agent_nb", default=3, type=int)
parser.add_argument("--actor_input_dims", 
                    default=(max_machine_num + 2, product_num, layer_num), type=tuple) #Actor dim 3x3 (max machine num + 2, product num, layer num)
parser.add_argument("--critic_input_dims", 
                    default=(max_machine_num + 2, product_num, product_num), type=tuple) #Critic dim 3x3 (max machine num + 2, product num x agent num, layer num)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--alpha", default=0.0003, type=float) #0.0003
parser.add_argument("--beta", default=0.0003, type=float) #0.0003
parser.add_argument("--gae_lambda", default=0.95, type=float)
parser.add_argument("--policy_clip", default=0.2, type=float)
parser.add_argument("--batch_size", default=140, type=int) #105
parser.add_argument("--n_epochs", default=5, type=int) #5
parser.add_argument("--lr_decay", default=10000, type=int) 
parser.add_argument("--decay_interval", default=7000, type=int) 
config = parser.parse_args()

#Define 2 normalizing methods (min-max norm and standardization)
class MinMaxNorm:
    def __init__(self):
        self.min_val = None
        self.max_val = None
        
    def update(self, x):
        x = np.array(x)
        self.min_val = np.min(x)
        self.max_val = np.max(x)
        return self.min_val, self.max_val
        
    def normalize(self,x):
        x = np.array(x)
        normalized_x = (x - self.min_val) / (self.max_val - self.min_val + 1e-10)
        return normalized_x

class Standardization:
    def __init__(self):
        self.mean_val = None
        self.std_val = None
        
    def update(self, x):
        x = np.array(x)
        self.mean_val = np.mean(x, axis=0)
        self.std_val = np.std(x, axis=0)
        return self.mean_val, self.std_val
    
    def standard(self,x):
        x = np.array(x)
        standard_x = (x - self.mean_val) / (self.std_val + 1e-10)
        return standard_x

#Create memory class for agent
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
                
    def generate_batches(self):  
        #Generate random batches    
        return  np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\

    def store_memory(self, states, actions, probs):
        self.states.append(states)
        self.actions.append(actions)
        self.probs.append(probs)
            
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        
#Create memory class for agent
class CriticMemory:
    def __init__(self,batch_size):
        self.global_state = []
        self.vals = []
        self.rewards = []
        self.done = []
        self.batch_size = batch_size
                
    def generate_batches(self):  
        #Generate random batches    
        n_states = len(self.global_state)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return  np.array(self.global_state),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.done),\
                batches                         

    def store_memory(self, global_state, vals, rewards, done):
        self.global_state.append(global_state)
        self.vals.append(vals)
        self.rewards.append(rewards)
        self.done.append(done)
                
    def clear_memory(self):
        self.global_state = []
        self.vals = []
        self.rewards = []
        self.done = []
         
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, chkpt_dir, file_name, lr_decay = config.lr_decay, decay_interval = config.decay_interval):
        super(ActorNetwork, self).__init__()

        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
            
        self.checkpoint_file = os.path.join(chkpt_dir, file_name)

        # Unpack the input dimensions
        in_channels, height, width = input_dims

        # Define layers for critic NN
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
                        
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),  # Add MaxPool2
            
            nn.Flatten(),
            nn.Linear(128 * (height//2) * (width//2), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
                        
            nn.Linear(512, n_actions),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        if state.size(0) == 1:
            self.actor.eval()

        else:
            self.actor.train()
            
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist
    
    def save_checkpoint(self):
        T.save({
            'actor': self.actor.state_dict()}, self.checkpoint_file)

    def load_checkpoint(self):
        checkpoint = T.load(self.checkpoint_file)
        self.actor.load_state_dict(checkpoint['actor'])

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, beta, chkpt_dir, file_name, lr_decay = config.lr_decay, decay_interval = config.decay_interval):
        super(CriticNetwork, self).__init__()

        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
            
        self.checkpoint_file = os.path.join(chkpt_dir, file_name)

        # Unpack the input dimensions
        in_channels, height, width = input_dims

        # Define layers for critic NN
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
                        
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),  # Add MaxPool2
            
            nn.Flatten(),
            nn.Linear(128 * (height//2) * (width//2), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
                        
            nn.Linear(512, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.lr_decay_rate = lr_decay
        self.lr_decay_interval = decay_interval
        
    def update_lr(self,episode):
        new_lr = config.beta * (self.lr_decay_rate ** (episode // self.lr_decay_interval))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        if episode >= self.lr_decay_interval:
            print('... lr decaying ...')
        
    def forward(self, state):
        # Ensure the input is a 4D tensor: (batch_size, channels, height, width)
        if state.dim() == 3:
            state = state.unsqueeze(0)
            
        if state.size(0) == 1:
            self.critic.eval()
        else:
            self.critic.train()
        
        state = state.to(self.device)  # Ensure the state is on the correct device
        value = self.critic(state)  # Forward pass
        return value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

#Define agent class (shared class among agents)
class Agent:
    def __init__(self, n_actions, chkpt_dir):
        self.actor_input_dims = config.actor_input_dims  
        self.critic_input_dims = config.critic_input_dims
        self.agent_nb = config.agent_nb
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.beta = config.beta
        self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs        
                
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        #Actor NNs
        actor_0 = ActorNetwork(n_actions, self.actor_input_dims, self.alpha, chkpt_dir, file_name = 'actor_NN')
        actor_1 = ActorNetwork(n_actions, self.actor_input_dims, self.alpha, chkpt_dir, file_name = 'actor_NN')
        actor_2 = ActorNetwork(n_actions, self.actor_input_dims, self.alpha, chkpt_dir, file_name = 'actor_NN')
                
        #Critic NNs
        self.critic = CriticNetwork(self.critic_input_dims, self.beta, chkpt_dir, file_name = 'shared_critic_NN')        

        #Memory
        memory_0 = PPOMemory()
        memory_1 = PPOMemory()
        memory_2 = PPOMemory()
        self.memory = [memory_0, memory_1, memory_2]
        self.critic_memory = CriticMemory(self.batch_size)
        
        #Define agents group
        self.actors =[actor_0, actor_1, actor_2]
            
    #Function to save data to memory    
    def remember(self, states, observation, actions, probs, vals, reward, done):
        for a in range(self.agent_nb):
            self.memory[a].store_memory(observation[a], actions[a], probs[a])
        self.critic_memory.store_memory(states, vals, reward[0], done) #Shared reward among agents

    #Function to save model
    def save_models(self):
        print('... saving models ...')
        for a in range(self.agent_nb):
            self.actors[a].save_checkpoint()
        self.critic.save_checkpoint()

    #Function to load model (testing phase)
    def load_models(self):
        print('... loading models ...')
        for a in range(self.agent_nb):
            self.actors[a].load_checkpoint()
        self.critic.load_checkpoint()
        
    def apply_action_mask(self, dist, mask):
        mask = T.tensor(mask, dtype=T.float).to(self.device)
        dist_probs = dist.probs
        masked_probs = dist_probs * mask
        masked_probs /= masked_probs.sum()  # Normalize the masked probabilities
        return T.distributions.Categorical(masked_probs)
    
    def normalize_observations(self, observations):
        M = MinMaxNorm()
        
        # Concatenate and normalize observations for all agents
        all_f1 = np.concatenate([obs[0] for obs in observations])
        all_f2 = np.concatenate([obs[1] for obs in observations])
        
        M.update(all_f1)
        all_f1 = M.normalize(all_f1)
        
        M.update(all_f2)
        all_f2 = M.normalize(all_f2)

        norm_f1 = np.split(all_f1, self.agent_nb)
        norm_f2 = np.split(all_f2, self.agent_nb)
        
        for a in range(self.agent_nb):
            observations[a][0] = norm_f1[a]
            observations[a][1] = norm_f2[a]
        
        return observations
    
    #Function to choose action by sampling (training phase)
    def choose_action(self, state, observations):
        observations = self.normalize_observations(observations)
        observations = [T.tensor(obs, dtype=T.float).to(self.device) for obs in observations]

        actions = [[] for _ in range(self.agent_nb)]    
        probs = [[] for _ in range(self.agent_nb)]    
        critic_values = []
        
        for agent in range(self.agent_nb):
            #Generate action distributions
            dist = self.actors[agent](observations[agent])
            
            action = dist.sample()
            acton_probs = T.squeeze(dist.log_prob(action)).item()
            action = T.squeeze(action).item()
            
            #Add each agent's actions/probs into action/prob lists
            actions[agent] = action
            probs[agent] = acton_probs
        
        #Normalize global state
        for l in range(2):
            M = MinMaxNorm()
            M.update(state[l])
            state[l] = M.normalize(state[l])
        
        #Calculate critic values
        state = T.tensor(state, dtype = T.float).to(self.device)
        value = self.critic(state)
        value = T.squeeze(value).item()
        critic_values = value
        
        return actions, probs, critic_values
        
    #Function to choose action by highest probability (evaluating step in the training phase)
    def evaluate_action(self, observations):
        observations = self.normalize_observations(observations)
        observations = [T.tensor(obs, dtype=T.float).to(self.device) for obs in observations]

        actions = [[] for _ in range(self.agent_nb)]    
        
        for agent in range(self.agent_nb):
            #Generate action distributions
            dist = self.actors[agent](observations[agent])
            
            #Sample actions from distribution/Take probability
            action = T.argmax(dist.probs, dim=-1).item()

            #Add each agent's actions/probs into action/prob lists
            actions[agent] = np.array(action)     
            
        return actions
    
    def learn(self):
        #self.entropy_coef = 0.01
        self.max_grad_norm = 1
        total_loss = 0
        
        #Train each agent seperately
        for _ in range(self.n_epochs):
            #Update critic NN
            global_state_arr, vals_arr, reward_arr, dones_arr, batches = self.critic_memory.generate_batches() 

            #Critic values and Normalized reward
            #S = Standardization()
            #S.update(vals_arr)
            #std_vals = S.standard(vals_arr)
            #std_vals = T.tensor(std_vals).to(self.device)
            std_vals = T.tensor(vals_arr).to(self.device)
            
            S = Standardization()
            S.update(reward_arr)
            std_reward = S.standard(reward_arr)
            std_reward = T.tensor(std_reward).to(self.device)
            #std_reward = T.tensor(reward_arr).to(self.device)
            
            #Calculate advantage seperately for each sample
            advantage = np.zeros(len(std_reward), dtype=np.float32)  
            start_idx = 0
            end_idx = 21
            for _ in range(len(std_reward)//21):
                for t in range(start_idx, end_idx-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, end_idx-1):
                        a_t += discount*(std_reward[k] + self.gamma*std_vals[k+1]*(1-int(dones_arr[k])) - std_vals[k])
                        discount *= self.gamma*self.gae_lambda
                    advantage[t] = a_t  
                start_idx += 21
                end_idx += 21
            
            #Standardize advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
            advantage = T.tensor(advantage).to(self.device)
            
            for batch in batches:
                global_state = T.tensor(global_state_arr[batch],dtype=T.float).to(self.device)
                
                #Calculate critic loss
                #self.critic.update_lr(episode)
                critic_value = self.critic(global_state)
                critic_value = T.squeeze(critic_value)
                
                returns = advantage[batch] + std_vals[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()
                
                #Backpropagation critic loss
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic.optimizer.step()
                total_loss += critic_loss.item()
                
            #Update actor NNs
            for i in range(self.agent_nb): 
                #Sample data from agent's memory
                state_arr, action_arr, old_prob_arr = self.memory[i].generate_batches()
                                 
                for batch in batches:  
                    states = T.tensor(state_arr[batch],dtype=T.float).to(self.device)
                            
                    old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
        
                    actions = T.tensor(action_arr[batch]).to(self.device)
        
                    dist = self.actors[i](states)
                                            
                    new_probs = dist.log_prob(actions)
                    #entropy0 = dist_0.entropy().mean()
                    prob_ratio = (new_probs - old_probs).exp()
                            
                            
                    #Calculate actor loss
                    prob_ratio = prob_ratio
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                                            
                    #Calculate actor loss (without adding entropy)
                    #entropy = (entropy0 + entropy1 + entropy2)/3
                    #actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean() - self.entropy_coef*entropy
                    actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                        
                    #Backpropagation actor loss
                    self.actors[i].optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                    self.actors[i].optimizer.step()
                    total_loss += actor_loss.item()
                    
        return total_loss/self.n_epochs














