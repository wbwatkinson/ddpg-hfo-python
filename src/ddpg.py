#/!usr/bin/env python
# encoding: utf-8

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, concatenate, merge, Lambda, Activation
from keras.utils.training_utils import multi_gpu_model
from keras.activations import relu
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import numpy as np
import random

from collections import deque

from absl import flags

import logging

kStateInputCount = 1
kMinibatchSize = 32
kActionSize = 4
kActionParamSize = 6

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, "Seed the RNG. Default: time")
flags.DEFINE_float('tau', 0.001, "Step size for soft updates")
flags.DEFINE_integer('soft_update_freq', 1, "Do SoftUpdateNet this frequently")
flags.DEFINE_float('gamma', 0.99, "Discount factor of future rewards (0,1]")
flags.DEFINE_integer('memory', 500000, "Capacity of replay memory")
flags.DEFINE_integer('memory_threshold', 1000, "Number of transitions required to start learning")
flags.DEFINE_integer('loss_display_iter', 1000, "Frequency of loss display")
flags.DEFINE_integer('snapshot)freq', 10000, "Frequency (steps) snapshorts")
flags.DEFINE_bool('remove_old_snapshots', True, "Remove old snapshots when writing more recent ones.")
flags.DEFINE_bool('snapshot_memory', True, "Snapshot the replay memory along with the network.")
flags.DEFINE_float('beta', 0.5, "Mix between off-policy and on-policy updates")


class DDPG:
	"""
	Implementation of the Actor-Critic DDPG network
	"""


	def __init__(self, state_size, action_size, action_param_size, actor_hidden,
		critic_hidden, actor_lr, critic_lr, final_epsilon, explore):

		"""
		Initializes the actor-critic network

		Args:
			state_size (int): Number of features in the state description
			action_size (int): Number of possible discrete actions
			action_param_size (int): Total number of continuous parameters associated with actions
			actor_hidden (List(int)): The size of the layers within the actor network
			critic_hidden (List(int)): the size of the layers within the critic network
			actor_lr (float): Learning rate of the actor network
			critic_lr (float): Learning rate of the critic network
			final_epsilon (float): The frequency the agent will explore after the exploration phase is Solver
			explore (int): The number of iterations during which the agent will explore the action space
		"""

		self._tensorflow_session = self._generate_tensorflow_session()

		self._explore = explore
		self._final_epsilon = final_epsilon

		self._tau = FLAGS.tau
		self._gamma = FLAGS.gamma

		self._batch_size = kMinibatchSize

		self._state_size = state_size
		self._action_size = action_size
		self._action_param_size = action_param_size

		self._memory = deque(maxlen = FLAGS.memory)

		self._iter = 0


		# ==========================================
		# Create Actor Model
		# ==========================================
		logging.info('*** Creating Actor Model ***')
		self._actor_model, self._actor_state_input = self._create_actor_model('actor', actor_hidden, actor_lr)
		self._target_actor_model, _ = self._create_actor_model('actor_target', actor_hidden, actor_lr)

		self._actor_gradients = tf.placeholder(tf.float32, [None, action_size + action_param_size]) # or state size?

		actor_model_weights = self._actor_model.trainable_weights

		self._actor_grads = tf.gradients(self._actor_model.output, actor_model_weights, -self._actor_gradients)

		grads = zip(self._actor_grads, actor_model_weights)

		self._optimize = tf.train.AdamOptimizer(actor_lr).apply_gradients(grads)


		# ==========================================
		# Create Critic Model
		# ==========================================
		logging.info('*** Creating Critic Model ***')
		self._critic_model, self._critic_state_input, self._critic_action_input = self._create_critic_model('critic', critic_hidden, critic_lr)
		self._target_critic_model, _, _ = self._create_critic_model('critic_target', critic_hidden, critic_lr)

		self._critic_grads = tf.gradients(self._critic_model.output, self._critic_action_input)


		# ==========================================
		# Initialize variables for gradient calculations
		# ==========================================
		self._tensorflow_session.run(tf.global_variables_initializer())


	def _epsilon(self):
		"""
		Calculates and returns the current epsilon value with the expected value to be final_epsilon after explore iterations

		Returns:
			The value of epsilon
		"""
		if (self._iter < self._explore):
			return 1.0 - (1.0 - self._final_epsilon) * (self._iter / self._explore)
		else:
			return self._final_epsilon


	def act(self, state):
		"""
		Returns an action vector.
		The action is random with probability epsilon and is determined by the actor-critic policy with probability (1 - epsilon)

		Args:
			state (Numpy Array(float)): state features

		Returns:
			Numpy Array(float): action vector
		"""
		if np.random.random() < self._epsilon():
			act = []
			for i in range (0,10):
				if i < 4:
					act.append(random.uniform(0.0, 1.0))
				else:
					act.append(random.uniform(-1.0, 1.0))
			act = np.asarray(act)
			logging.info("Exploring Action")
			return act
		logging.info("Exploiting Action")
		return self._actor_model.predict(state.reshape(1, state.shape[0]))[0]


	def remember(self, state, action, reward, next_state, done):
		"""
		Store s, a, r, s' into replay memory

		Args:
			state (Numpy Array(float)):
			action (Numpy Array(float)):
			reward (float):
			next_state (Numpy Array(float)):
		"""
		self._memory.append((state, action, reward, next_state, done))


	def train(self):
		"""
		Train actor critic network from batch of samples from memory
		"""
		self._iter += 1
		if len(self._memory) < self._batch_size:
			return

		states, actions, rewards, next_states, dones = self._get_samples()

		self._train_critic(states, actions, next_states, rewards, dones)
		self._train_actor(states)


	def _generate_tensorflow_session(self):
		"""
		Create Tensorflow session for use
		"""
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		return tf.Session(config = config)


	def _create_actor_model(self, model_name, hidden, lr):
		input_layer = Input(shape=[self._state_size], name='input_layer')

		layer_size = hidden[0]
		layer = Dense(layer_size, activation='linear', name='dense' + str(layer_size) + '_layer')(input_layer)
		layer = Activation(lambda x: relu(x, alpha=0.01), name='activation_' + str(layer_size))(layer)

		layers = iter(hidden)
		next(layers)

		for layer_size in layers:
			layer = Dense(layer_size, activation='linear', name='dense' + str(layer_size) + '_layer')(layer)
			layer = Activation(lambda x: relu(x, alpha=0.01), name='activation_' + str(layer_size))(layer)

		output_actions = Dense(self._action_size, activation='sigmoid', name='action_layer')(layer)
		output_action_params = Dense(self._action_param_size, activation='sigmoid', name='action_param_layer')(layer)
		output_layer = concatenate([output_actions, output_action_params], name='output_layer')

		model = Model(inputs=input_layer, outputs=output_layer, name=model_name)

		#model = multi_gpu_model(model, gpus=3)

		model.compile(loss='mse', optimizer=Adam(lr=lr))
		logging.info(model_name + ' model:')
		model.summary()

		return model, input_layer


	def _create_critic_model(self, model_name, hidden, lr):
		state_input = Input(shape=[self._state_size], name='state_inputs')
		action_input = Input(shape=[self._action_size + self._action_param_size], name='action_inputs')

		input_layer = concatenate([state_input, action_input], name='input_layer')

		layer_size = hidden[0]
		layer = Dense(layer_size, activation='linear', name='dense' + str(layer_size) + '_layer')(input_layer)
		layer = Activation(lambda x: relu(x, alpha=0.01), name='activation_' + str(layer_size))(layer)

		layers = iter(hidden)
		next(layers)

		for layer_size in layers:
			layer = Dense(layer_size, activation='linear', name='dense' + str(layer_size) + '_layer')(layer)
			layer = Activation(lambda x: relu(x, alpha=0.01), name='activation_' + str(layer_size))(layer)

		output_layer = Dense(1, activation = 'sigmoid', name = 'q_values_layer')(layer)

		model = Model(inputs=[state_input, action_input], outputs=output_layer, name=model_name)

		#model = multi_gpu_model(model, gpus=3)

		model.compile(loss='mse', optimizer=Adam(lr=lr))
		logging.info(model_name + ' model:')
		model.summary()

		return model, state_input, action_input


	def _get_samples(self):
		"""
		Gets a random sample of batch_size from agent's replay memory

		:return: Tuple(List(Float, Boolean))) denoting the sample of states, actions, rewards, next_states, and dones
		"""

		#samples = list(zip(*random.sample(self._memory, self._batch_size)))

		states, actions, rewards, next_states, dones = list(zip(*random.sample(self._memory, self._batch_size))) #samples #list(zip(*samples))

		return np.asarray(states), np.asarray(actions), rewards, np.asarray(next_states), dones


	def _train_actor(self, states):
	#def _train_actor(self, states):
		predicted_actions = self._actor_model.predict(states)
		#logging.info('run 1')
		grads = self._tensorflow_session.run(self._critic_grads, feed_dict ={
			self._critic_state_input: states,
			self._critic_action_input: predicted_actions
			})[0]
		#logging.info('run 2: %s\n%s' % (states.shape, states))
		#logging.info('run 2: %s\n%s' % (type(grads), grads)) #.shape))

		self._tensorflow_session.run(self._optimize, feed_dict={
			self._actor_state_input: states,
			self._actor_gradients: grads
			})


	def _train_critic(self, states, actions, next_states, rewards, dones):
		#for sample in samples:

		#states, actions, rewards, next_states, dones = sample
		# next_states = []
		# for i in range (0, 32):
		# 	state = []
		# 	for i in range (0, 59):
		# 		state.append(random.uniform(-1.0, 1.0))
		# 	state = np.array(state)
		# 	logging.info('State shape: %s' % (state.shape))
		# 	state = state.reshape((1, state.shape[0]))
		# 	#logging.info('State shape: %s' % (state.shape))
		# 	next_states.append(state)

		# next_states = np.asarray(next_states) #.reshape((1, next_states.shape[0]))

		#logging.info('Next States: %s\n%s' % (type(next_states), next_states))

		#ben cook method



		target_actions = self._target_actor_model.predict(next_states)
		next_q_values = self._target_critic_model.predict([next_states, target_actions])
		#target_q_values = rewards if dones else rewards + self._gamma * next_q_values

		target_q_values = [reward if this_done else reward + self._gamma * (next_q_value)
						   for (reward, next_q_value, this_done)
						   in zip(rewards, next_q_values, dones)]
		target_q_values = np.asarray(target_q_values)
		#logging.info('Fit critic (states): %s\n%s' % (states.shape, states))
		#logging.info('Fit critic (actions): %s\n%s' % (actions.shape, actions))
		self._critic_model.fit([states, actions], target_q_values, verbose=0)


	def _update_targets(self):
		self._update_actor_target()
		self._update_critic_target()


	def _update_actor_target(self):
		actor_weights = self._actor_model.get_weights()
		actor_target_weights = self.target_actor_model.get_weights()

		for i in xrange(len(actor_weights)):
			actor_target_weights[i] = self._tau * actor_weights[i] + (1 - self._tau) * actor_target_weights[i]

		self._target_actor_model.set_weights(actor_target_weights)


	def _update_critic_target(self):
		critic_weights = self._critic_model.get_weights()
		critic_target_weights = self.target_critic_model.get_weights()

		for i in xrange(len(critic_weights)):
			critic_target_weights[i] = self._tau * critic_weights[i] + (1 - self._tau) * critic_target_weights[i]

		self._critic_actor_model.set_weights(critic_target_weights)
