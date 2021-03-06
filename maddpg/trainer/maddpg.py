
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

import scipy

def minimize_and_clip(optimizer, objective, var_list, clip_val=0.5):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """    
    if clip_val is None:
        return optimizer.minimize(objective, var_list=var_list)
    else:
        gradients = optimizer.get_gradients(objective, params=var_list)
        print(gradients)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        return optimizer.apply_gradients(gradients)

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def update_target(model, target_model, polyak = 1.0 - 1e-2):
    new_weight = np.array(model.get_weights())
    old_weight = np.array(target_model.get_weights())
    target_model.set_weights(polyak * old_weight + (1-polyak) * new_weight)

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        u = tf.random.uniform(tf.shape(logits))
        return tf.nn.softmax(logits - tf.math.log(-tf.math.log(u)), axis=-1)


class Actor(tf.keras.Model):
    def __init__(self, obs_size, act_size, name="Actor"):
        super().__init__(name=name)
        
        self.original_dim = 4
    
        self.l1 = Dense(64, name="L1")
        self.l2 = Dense(64, name="L2")
        self.l3 = Dense(act_size, name="L3")

        self.dist = ProbabilityDistribution()
    
    def action(self, obs):
        # used when running the model in an environment
        #logits = self.predict_on_batch(obs)
        #action = self.dist.predict_on_batch(logits)
        action = self.dist(self(obs))
        return action

    def predict_many(self, obs):
        # used when training
        logits = self.predict(obs)
        action = self.dist.predict(logits)
        return action

    @tf.function
    def call(self, inputs):
        x = tf.nn.relu(self.l1(inputs))
        x = tf.nn.relu(self.l2(x))
        logits = self.l3(x)  
        return logits


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, learning_rate, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.learning_rate = learning_rate
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.obs_size = obs_shape_n[agent_index]
        self.joint_obs_size = np.sum(obs_shape_n)
        self.act_size = act_space_n[agent_index].n
        self.act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        
        self.joint_act_size = 0
        for i_act in act_space_n:
            self.joint_act_size += i_act.n
        self.args = args
        self.actor = Actor(self.obs_size, self.act_size)
        self.actor_target = Actor(self.obs_size, self.act_size)
        self.critic = self.build_critic()
        self.critic_target = self.build_critic()
        update_target(self.actor, self.actor_target, 0)
        update_target(self.critic, self.critic_target, 0)
        #self.actor, self.critic = self.build_model()
        #self.actor_target, self.critic_target = self.build_model()
        self.actor_optimizer = self.build_actor_optimizer()

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        
        gpu=-1

        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
    
    def build_model(self):
        """ actor (policy) neural network """
        inp = Input(self.obs_size)
        x = Dense(64, activation='relu')(inp)
        x = Dense(64, activation='relu')(x)
        actor_out = Dense(self.act_size)(x)
        
        actor = Model(inp, actor_out)
        # Note: "actor" is not compiled because we want customize the training process

        """ critic (value) neural network """
        inp = Input((self.joint_obs_size + self.joint_act_size,))
        x = Dense(64, activation='relu')(inp)
        x = Dense(64, activation='relu')(x)
        critic_out = Dense(1, activation='linear')(x)
        
        critic = Model(inp, critic_out)

        critic.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate, clipnorm=0.5))

        return actor, critic
    

    def build_critic(self):
        """ critic (value) neural network """
        inp = Input((self.joint_obs_size + self.joint_act_size,))
        x = Dense(64, activation='relu')(inp)
        x = Dense(64, activation='relu')(x)
        critic_out = Dense(1, activation='linear')(x)
        
        critic = Model(inp, critic_out)

        critic.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate, clipnorm=0.5))

        return critic
    

    def build_actor_optimizer(self):
        return Adam(learning_rate=self.learning_rate, clipnorm=0.5)
    
    def action(self, obs):
        #a = self.sample_action(obs[None])
        #print(obs[None].shape)
        #a = self._get_action_body(tf.constant(obs[None], dtype='float32'))
        #a = self.actor.predict_on_batch(tf.constant(obs[None], dtype='float32'))
        #print(a)
        a = self.actor.action(tf.constant(obs[None], dtype='float32'))
        ##a = self.actor(self.actor.dist(obs[None]))
        return a[0]
    
    def sample_action(self, obs):
        logits = self.actor.predict(obs, batch_size=len(obs))
        u = np.random.uniform(size=logits.shape)
        return a
    """
    @tf.function
    def _get_action_body(self, obs_tensor):
        with tf.device(self.device):
            logits = self.actor(obs_tensor)
            act_pd = self.act_pdtype_n[self.agent_index].pdfromflat(logits)
            a = act_pd.sample()
            return a
    """
    @tf.function
    def _get_action_body(self, obs_tensor):
        logits = self.actor(obs_tensor)
        u = tf.random.uniform(tf.shape(logits))
        a = tf.nn.softmax(logits - tf.math.log(-tf.math.log(u)), axis=-1)  
        return a

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        rew = np.expand_dims(rew, axis=-1)
        done = np.expand_dims(done, axis=-1)
        # train q network
        num_sample = 1
        target_q = 0.0
        """
        next_logits = self.actor_target.predict(obs_next)
        next_act_pd = self.act_pdtype_n[self.agent_index].pdfromflat(next_logits)
        new_next_act = next_act_pd.sample()
        """
        
        # train critic
        for i in range(num_sample):
            target_act_next_n = []
            for j in range(self.n):
                new_next_act = agents[j].actor_target.predict_many(obs_next_n[j]) 
                target_act_next_n.append(new_next_act) #TODO: mode
            #target_act_next_n[self.agent_index] = new_next_act
            next_state_action_n = np.concatenate((obs_next_n, target_act_next_n), axis=-1)
            next_state_action_attached = np.concatenate(next_state_action_n, axis=0)
            target_q_next = self.critic_target.predict(next_state_action_attached)
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        
        target_q /= num_sample
        state_action_n = np.concatenate((obs_n, act_n), axis=-1)
        state_action_attached = np.concatenate(state_action_n, axis=0)
        hist = self.critic.fit(state_action_attached, target_q, epochs=1, verbose=0)
        #q_loss = self.critic.train_on_batch(state_action_attached, target_q)
        
        q_loss = hist.history['loss'][0]

        
        obs_tensor = tf.constant(obs, dtype=tf.float32)
        obs_n_tensor = tf.constant(obs_n, dtype=tf.float32)
        act_n_tensor = tf.constant(act_n, dtype=tf.float32)
        #obs_tensor = tf.Variable(obs, dtype=tf.float32)
        #obs_n_tensor = tf.Variable(np.array(obs_n), dtype=tf.float32)
        #act_n_tensor = tf.Variable(np.array(act_n), dtype=tf.float32)

        # train actor network
        #p_loss = self.update_actor(obs, obs_n, act_n)
        p_loss = self.update_actor(obs_tensor, obs_n_tensor, act_n_tensor)
        
        """
        logits = self.actor.predict(obs)
        act_pd = self.act_pdtype_n[self.agent_index].pdfromflat(logits)
        new_act = act_pd.mode()
        act_n[self.agent_index] = new_act
        
        grads = self.critic.gradients(obs_n, act_n)

        np.concatenate(act_n, ) 

        state_action_n = np.concatenate((obs_n, act_n), axis=-1)
        state_action_attached = np.concatenate(state_action_n, axis=-1)
        hist = self.actor.fit(obs, state_action_attached, epochs=1, verbose=0)
        p_loss = hist.history['loss'][0]
        """
        update_target(self.actor, self.actor_target)
        update_target(self.critic, self.critic_target)
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
    
    @tf.function
    def update_actor(self, obs, obs_n, act_n):
        with tf.GradientTape() as tape:
            logits = self.actor(obs)  
            #new_act = self.act_pdtype_n[self.agent_index].pdfromflat(logits).mode()
            new_act = self.actor.dist(logits) 
            new_act = tf.expand_dims(new_act, axis=0)
            new_act_head = act_n[:self.agent_index]
            new_act_tail = act_n[self.agent_index + 1:]
            
            new_act_n = tf.concat((new_act_head, new_act, new_act_tail), axis=0)

            state_action_n = tf.concat((obs_n, new_act_n), axis=-1)
            state_action_attached = tf.squeeze(state_action_n)
            q_val = self.critic(state_action_attached)[:,0]
            p_loss = -tf.reduce_mean(q_val)
            reg_loss = tf.reduce_mean(tf.square(logits))
            total_loss = p_loss + reg_loss * 1e-3

        actor_grad = tape.gradient(total_loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_weights)) 
        return total_loss
    
    def _actor_loss(self, actions_and_values, logits):
        # A trick to input actions and advantages through the same API.
        actions, values = tf.split(actions_and_values, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        
        #weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        #actions = tf.cast(actions, tf.int32)
        #policy_loss = weighted_sparse_ce(actions, logits, sample_weight=values)

        # Entropy loss can be calculated as cross-entropy over itself.
        #probs = tf.nn.softmax(logits)
        #entropy_loss = kls.categorical_crossentropy(probs, probs)
        print(values)
        print(tf.shape(logits))
        policy_loss = -tf.reduce_mean(values)
        logits_loss = tf.reduce_mean(logits)
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss + 1e-3 * logits_loss
    
    def _actor_loss(self, values, logits):
        p_loss = -tf.reduce_mean(values)
        reg_loss = tf.reduce_mean(tf.square(logits))
        total_loss = p_loss + reg_loss * 1e-3
        return total_loss



    def load_models(self, path, version_name):
        file_name = 'a' + str(self.agent_index) + 'A' + version_name
        self.actor.load_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'C' + version_name
        self.critic.load_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'AT' + version_name
        self.actor_target.load_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'CT' + version_name
        self.critic_target.load_weights(path + file_name)

    def save_models(self, path, version_name):
        file_name = 'a' + str(self.agent_index) + 'A' + version_name
        self.actor.save_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'C' + version_name
        self.critic.save_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'AT' + version_name
        self.actor_target.save_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'CT' + version_name
        self.critic_target.save_weights(path + file_name)
    
        