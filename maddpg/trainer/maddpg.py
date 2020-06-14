import numpy as np
import random
import tensorflow as tf
from keras.optimizers import Adam

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """    
    if clip_val is None:
        return optimizer.minimize(objective, var_list=var_list)
    else:
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
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

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[collaborationp_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.compat.v1.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

def update_target(model, target_model):
    polyak = 1.0 - 1e-2
    old_weight = model.get_weights()
    new_weight = target_model.get_weights()
    target_model.set_weights(polyak * old_weight + (1-polyak) * new_weight)

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, learning_rate, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.learning_rate = learning_rate
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.obs_size = obs_shape_n[agent_index]
        self.joint_obs_size = (np.sum(obs_shape_n),)
        self.act_size = act_space_n[agent_index].n
        self.act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        self.joint_act_size = 0
        for i_act in act_space_n:
            self.joint_act_size += i_act.n
        self.joint_act_size = (self.joint_act_size,)
        self.args = args
        self.actor, self.critic = self.build_model()
        self.actor_target, self.critic_target = self.build_model()
        
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
    
    def build_model(self):
        """ actor (policy) neural network """
        from keras import Input
        from keras.layers import Dense
        from keras.models import Model

        inp = Input(self.obs_size)
        x = Dense(64, activation='relu')(inp)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        actor_out = Dense(self.act_size, activation='linear')(x)
        
        actor = Model(inp, actor_out)

        """ critic (value) neural network """
        inp = Input(self.joint_obs_size + self.joint_act_size)
        x = Dense(64, activation='relu')(inp)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        critic_out = Dense(1, activation='linear')(x)
        
        critic = Model(inp, critic_out)

        critic.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
       
        return actor, critic

    def action(self, obs):
        a = self.sample_action(obs[None])
        #print(a.shape)
        return a[0]

    def sample_action(self, obs):
        logits = self.actor.predict(obs)
        #print(logits)
        act_pd = self.act_pdtype_n[self.agent_index].pdfromflat(logits)
        a = act_pd.sample()
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

        # train q network
        num_sample = 1
        target_q = 0.0
        
        for i in range(num_sample):
            target_act_next_n = []
            for j in range(self.n):
                action = agents[j].sample_action(obs_next_n[j])
                target_act_next_n.append(action)
            target_q_next = self.critic_target.predict(np.concatenate((obs_next_n, target_act_next_n), axis=1))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        
        target_q /= num_sample
        q_input = np.concatenate((obs_n, act_n), axis=1)
        hist = self.critic.fit(q_input, target_q, epochs=1, verbose=0)
        q_loss = hist.history['loss'][0]

        # train p network
        
        optimizer = Adam(learning_rate=self.args.lr)
        
        def get_loss(obs_n=obs_n, act_n=act_n):
            # compute -E[q]
            q_val = self.critic.predict(np.concatenate((obs_n, act_n), axis=1))
            return -np.mean(q_val)
        
        p_loss = minimize_and_clip(optimizer, get_loss, self.actor.trainable_weights, clip_val=10)

        update_target(self.actor, self.actor_target)
        update_target(self.critic, self.critic_target)

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
    
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
