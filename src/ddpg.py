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

from abc import ABC, abstractmethod

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


class DDPG(ABC):
	"""
	Implementation of the Actor-Critic DDPG network
	"""


	def __init__(self, id, state_size, action_size, action_param_size, actor_hidden,
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
		K.set_session(self._tensorflow_session)

		self._explore = explore
		self._final_epsilon = final_epsilon

		self._tau = FLAGS.tau
		self._gamma = FLAGS.gamma

		self._memory = deque(maxlen = FLAGS.memory)

		# properties
		self.__id = id
		self.__action_size = action_size
		self.__action_param_size = action_param_size
		self.__iter = 0
		self.__state_size = state_size
		self.__batch_size = kMinibatchSize


		# ==========================================
		# Create Actor Model
		# ==========================================
		logging.debug('*** Creating Actor Model ***')
		self._actor_model, self._actor_state_input = self._create_actor_model('actor', actor_hidden, actor_lr)
		self._target_actor_model, _ = self._create_actor_model('actor_target', actor_hidden, actor_lr)

		self._actor_gradients = tf.placeholder(tf.float32, [None, action_size + action_param_size]) # or state size?

		actor_model_weights = self._actor_model.trainable_weights
		# Change this back to -self._actor_gradients
		self._actor_grads = tf.gradients(self._actor_model.output, actor_model_weights, -self._actor_gradients)

		grads = zip(self._actor_grads, self._actor_model.trainable_weights)

		self._actor_optimize = tf.train.AdamOptimizer(actor_lr).apply_gradients(grads)


		# ==========================================
		# Create Critic Model
		# ==========================================
		logging.debug('*** Creating Critic Model ***')
		self._critic_model, self._critic_state_input, self._critic_action_input = self._create_critic_model('critic', critic_hidden, critic_lr)
		self._target_critic_model, _, _ = self._create_critic_model('critic_target', critic_hidden, critic_lr)

		self._critic_grads = tf.gradients(self._critic_model.output, self._critic_action_input)


		# ==========================================
		# Initialize variables for gradient calculations
		# ==========================================
		self._tensorflow_session.run(tf.global_variables_initializer())


	@property
	def batch_size(self):
		return self.__batch_size


	@property
	def id(self):
		return self.__id

	@property
	def action_size(self):
		return self.__action_size


	@property
	def action_param_size(self):
		return self.__action_param_size


	@property
	def iter(self):
		return self.__iter


	@property
	def state_size(self):
		return self.__state_size


	def __del__(self):
		logging.debug('Destroying Agent')


	def _epsilon(self):
		"""
		Calculates and returns the current epsilon value with the expected value to be final_epsilon after explore iterations

		Returns:
			The value of epsilon
		"""
		if (self.__iter < self._explore):
			return 1.0 - (1.0 - self._final_epsilon) * (self.__iter / self._explore)
		else:
			return self._final_epsilon


	def select_action(self, state, epsilon):
		"""
		Returns an action vector.
		The action is random with probability epsilon and is determined by the actor-critic policy with probability (1 - epsilon)

		Args:
			state (Numpy Array(float)): state features
			epsilon (float): epsilon value

		Returns:
			Numpy Array(float): action vector
		"""

		assert epsilon >= 0 and epsilon <= 1.0, 'Epsilon out of range error'

		if np.random.random() < epsilon:
			# generate fake kick
			# act = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0])
			# logging.debug('Exploring action: %s' % (act))
			# return act
			act = self.random_action()
			#logging.debug('Exploring action: %s' % (act))
			return act
		else:
			act = self._actor_model.predict(state.reshape(1, state.shape[0]))[0]
			#logging.debug('Exploiting action: %s' % (act))
			return act

	def get_action(self, action_vector):
		action_vector_copy = action_vector.copy()
		action_vector_copy[2] = -9999
		action_index = np.argmax(action_vector_copy[0:self.action_size])
		action = self.action_string(action_index)

		param_index = self._get_param_index(action)

		num_params = self.num_params(action)

		params = []
		for i in range(num_params):
			params.append(action_vector[param_index + i])

		#logging.debug('Action(%i): %s Params(%i): %s' % (action_index, action, param_index, params))

		return action, params


	def evaluate_action(self, state, action_vector):
		#logging.debug('State: %s' % (state))
		#logging.debug('Action" %s' % (action_vector))
		#return self._critic_model.predict([state, action_vector])
		return self._critic_model.predict([state.reshape(1, state.shape[0]), action_vector.reshape(1, action_vector.shape[0])])


	def label_transitions(self, episode):
		transitions = reversed(episode)
		transition_num = len(episode)
		next_transition = next(transitions)
		next_transition[3] = next_transition[2]
		#logging.debug('Transition %i: %f, %f' % (transition_num, reward))
		#logging.debug('Transition %i: %f, %f' % (transition_num, next_transition[3], next_transition[2]))


		for transition in transitions:
			_, _, next_reward, next_q_val, _, _ = next_transition
			_, _, reward, _, _, _ = transition
			transition[3] = reward + FLAGS.gamma * next_q_val
			transition_num -= 1
			next_transition = transition

		for transition in episode:
			logging.debug('Transition reward %f QVal: %f' % (transition[2], transition[3]))


	#def update(self):

	#def add_transitions(self, episode):


	# def act(self, hfo, action_vector):
	# 	action_index = np.argmax(action_vector[0:self._action_size])
	# 	action = self.action_string(action_index)
	# 	# logging.info('Action Type: Index(%i) Type(%s)' % (action_type_index, action_type))
	# 	arg_index = self._get_param_offset(action_index) + self._action_size
	# 	#arg_1 = arg_0 + 1
	#
	# 	num_params = self.num_params(action) #hfo_lib.numParams(action_type)
	#
	# 	args = []
	# 	for i in range(num_params):
	# 		args.append(action_vector[arg_index+i])
	#
	# 	# lookup enum value of action string (note that our implementation does not require that the actions be enumerated
	# 	# in the same order as the ACTION STRINGS enum type
	# 	action_val = list(ACTION_STRINGS.keys())[list(ACTION_STRINGS.values()).index(action)]
	#
	# 	#logging.info('Action Type: Index(%i) Val(%i) Name(%s) N_Params(%i) %s' % (action_index,action_val, action, num_params, args))
	# 	logging.info('Action: %s  %s' % (action, args))
	#
	# 	#logging.info('Action:', action, '[', *args, sep=',', ']')
	#
	# 	hfo.act(action_val, *args)

	# def get_action(self, state):
	# 	"""
	# 	Returns an action vector.
	# 	The action is random with probability epsilon and is determined by the actor-critic policy with probability (1 - epsilon)
	#
	# 	Args:
	# 		state (Numpy Array(float)): state features
	#
	# 	Returns:
	# 		Numpy Array(float): action vector
	# 	"""
	# 	if np.random.random() < self._epsilon():
	# 		act = self.random_action()
	# 		logging.info("Exploring Action")
	# 		return act
	# 	logging.info("Exploiting Action")
	# 	return self._actor_model.predict(state.reshape(1, state.shape[0]))[0]

	def add_transitions(self, episode):
		for transition in episode:
			state, action_vector, reward, q_val, next_state, terminal = transition
			self.remember(state, action_vector, reward, q_val, next_state, terminal)


	def remember(self, state, action_vector, reward, q_val, next_state, terminal):
		"""
		Store s, a, r, s' into replay memory

		Args:
			state (Numpy Array(float)):
			action (Numpy Array(float)):
			reward (float):
			next_state (Numpy Array(float)):
		"""
		self._memory.append((state, action_vector, reward, q_val, next_state, terminal))


	def train(self):
		"""
		Train actor critic network from batch of samples from memory
		"""
		if len(self._memory) < self.batch_size or len(self._memory) < FLAGS.memory_threshold:
			return

		self.__iter += 1

		# critic_loss, avg_q = UpdateActorCritic

		critic_loss = random.random()
		avg_q = random.random()

		#smoothed_critic_loss += critic_loss / FLAGS.loss_display_iter
		#smoothed_actor_loss += actor_loss / FLAGS.loss_display_iter

		if self.iter % FLAGS.loss_display_iter == 0:
			logging.info('[Agent %i] Critic Iteration %i, loss = %f' %
				(self.id, self.iter, critic_loss))
			logging.info('[Agent %i] Actor Iteration %i, avg_q_value = %f' %
				(self.id, self.iter, avg_q))


		#list(zip(*random.sample(self._memory, count))

		# <editor-fold> Trying something else

		# non_terminal_next_states = []
		# for next_state, terminal in zip(next_states, terminals):
		# 	if not terminal:
		# 		logging.debug(next_state)
		# 		non_terminal_next_states.append(next_state)
		#
		#
		#
		# #non_terminal_next_states, _ = list(zip(*[(next_state, terminal) for next_state, terminal in zip(next_states, terminals) if not terminal]))
		#
		# non_terminal_next_states = np.asarray(non_terminal_next_states)
		#
		# #logging.debug('Unzip %s %s' % (type(np.array(non_terminal_next_states)), np.array(non_terminal_next_states)))
		#
		#
		# #non_terminal_next_states, _ = [(next_state, terminal) for next_state, terminal in zip(next_states, terminals) if terminal]
		#
		# # critic forward through actor
		# # returns target_q_values
		# logging.debug('[Forward] %s Through % s' % (self._target_critic_model.name, self._target_actor_model.name))
		# logging.debug('[Forward] %s' % (self._target_actor_model.name))
		# actor_outputs = self._target_actor_model.predict(non_terminal_next_states)
		# #exit(0)
		# logging.debug('[Forward] %s' % (self._target_critic_model.name))
		# target_q_vals = iter(self._target_critic_model.predict([non_terminal_next_states, actor_outputs]))
		#
		#
		# targets = []
		# for reward, terminal, on_policy_target in zip(rewards, terminals, on_policy_targets):
		# 	if terminal:
		# 		off_policy_target = reward
		# 		#off_policy_targets.append(reward)
		# 	else:
		# 		#off_policy_targets.append(reward + self._gamma * next(target_q_vals))
		# 		off_policy_target = reward + self._gamma * next(target_q_vals)
		#
		# 	target = FLAGS.beta * on_policy_target + (1 - FLAGS.beta) * off_policy_target
		#
		# 	assert np.isfinite(target), "Target not finite!"
		# 	targets.append(target)
		#
		# targets = np.asarray(targets)
		#
		# logging.debug('[Step] Critic')
		#
		# hist = self._critic_model.fit([states, actions], targets, verbose=0)
		#
		# logging.debug('History: %s' % (hist.history))
		#
		# critic_loss = hist.history['loss']
		#
		# #logging.debug('Loss: %f' % (critic_loss))
		#
		# actor_output = self._actor_model.predict(states)
		#
		# logging.debug('ActorOutput: %s' % (actor_output))
		#
		# q_values = self._critic_model.predict([states, actor_output])
		#
		# avg_q = np.mean(q_values)

		# </editor-fold> Trying something else



		# states, actions, rewards, next_states, dones = self._get_samples()
		#
		states, actions, rewards, on_policy_targets, next_states, terminals = self._get_samples(self.batch_size)

		#self._train_critic(states, actions, rewards, on_policy_targets, next_states, terminals)
		#logging.debug('Critic grads: %s' % (self._critic_grads))
		#self._train_actor(states)

		# <editor-fold> yanplau method for updating
        # target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
		#
        # for k in range(len(batch)):
        #     if dones[k]:
        #         y_t[k] = rewards[k]
        #     else:
        #         y_t[k] = rewards[k] + GAMMA*target_q_values[k]
		#
        # if (train_indicator):
        #     loss += critic.model.train_on_batch([states,actions], y_t)
        #     a_for_grad = actor.model.predict(states)
        #     grads = critic.gradients(states, a_for_grad)
        #     actor.train(states, grads)
        #     actor.target_train()
        #     critic.target_train()
		#
        # total_reward += r_t
        # s_t = s_t1

	    # def gradients(self, states, actions):
	    #     return self.sess.run(self.action_grads, feed_dict={
	    #         self.state: states,
	    #         self.action: actions
	    #     })[0]
		# </editor-fold> yanplau method for updating

		target_q_values = self._target_critic_model.predict([next_states, self._target_actor_model.predict(next_states)])
		targets = []
		for reward, terminal, target_q_value, on_policy_target in zip(rewards, terminals, target_q_values, on_policy_targets): #, on_policy_targets):
			if terminal:
				off_policy_target = reward
			else:
				off_policy_target = reward + self._gamma * target_q_value
			target = FLAGS.beta * on_policy_target + (1 - FLAGS.beta) * off_policy_target
			assert np.isfinite(target), "Target not finite!"
			targets.append(off_policy_target)

		targets = np.asarray(targets)
		#logging.debug('[%i] Targets: %s' % (self.iter, targets))

		history = self._critic_model.fit([states, actions], targets, verbose=0)
		a_for_grad = self._actor_model.predict(states)
		#logging.debug('[%i] A for Grads: %s' % (self.iter, a_for_grad))

		grads = self._tensorflow_session.run(self._critic_grads, feed_dict={
			self._critic_state_input: states,
			self._critic_action_input: a_for_grad
		})[0]
		#logging.debug('[%i] Grads: %s' % (self.iter, grads))

		actions = self._actor_model.predict(states)
		#logging.debug('[%i] Old Actions: %s' % (self.iter, actions))

		# implement gradient clipping/squishing right here with grads
		# for grad, action in zip(grads, actions):
		# 	for g in grad:
		# 		g = 0

		# <editor-fold> zeroing gradients
		# for i in range(len(actions)):
		# 	for j in range(self.action_size):
		# 		if actions[i][j] < -1.0 or actions[i][j] > 1.0:
		# 			grads[i][j] = 0.0
		# 	for k in range(self.action_size, self.action_size + self.action_param_size):
		# 		if k == 4:
		# 			min = -100
		# 			max = 100
		# 		elif k == 5 or k == 6 or k == 7 or k == 9:
		# 			min = -180
		# 			max = 180
		# 		elif k == 8:
		# 			min = 0
		# 			max = 100
		# 		if actions[i][k] < min or actions[i][k] > max:
		# 			grads[i][k] = 0
		# </editor-fold> zeroing gradients

		# <editor-fold> squashing gradients

		# </editor-fold> squashing gradients

		# <editor-fold> inverting gradients
		for i in range(len(actions)):
			for j in range(self.action_size):
				min = -1.0
				max = 1.0
				if grads[i][j] > 0:
					grads[i][j] *= (max - actions[i][j]) / (2.0)
				elif grads[i][j] < 0:
					grads[i][j] *= (actions[i][j] - min) / (2.0)
			for k in range(self.action_size, self.action_size + self.action_param_size):
				if k == 4:
					min = -100
					max = 100
				elif k == 5 or k == 6 or k == 7 or k == 9:
					min = -180
					max = 180
				elif k == 8:
					min = 0
					max = 100
				if grads[i][k] > 0:
					grads[i][k] *= (max - actions[i][k]) / (max - min)
				elif grads[i][k] < 0:
					grads[i][k] *= (actions[i][k] - min) / (max - min)
		# </editor-fold> inverting gradients

		# for i, grad in enumerate(grads):
		# 	logging.debug('grads val: %s' % (grads[i]))
		# 	for j, g in enumerate(grads[i]):
		# 		logging.debug('val: %s' % g)
		# 		exit(0)
		# 		g[j] = 0
		#logging.debug('[%i] New Grads: %s' % (self.iter, grads))

		self._tensorflow_session.run(self._actor_optimize, feed_dict={
			self._actor_state_input: states,
			self._actor_gradients: grads
		})

		actions = self._actor_model.predict(states)
		#logging.debug('[%i] New Actions: %s' % (self.iter, actions))
		#self._actor_model.train(states, grads)


		# <editor-fold old train critic method
		# target_actions = self._target_actor_model.predict(next_states)
		# target_q_values = self._target_critic_model.predict([next_states, target_actions])
		# #target_q_values = rewards if dones else rewards + self._gamma * next_q_values
		# targets = []
		# for reward, terminal, target_q_val, on_policy_target in zip(rewards, terminals, target_q_values, on_policy_targets): #, on_policy_targets):
		# 	if terminal:
		# 		off_policy_target = reward
		# 		#off_policy_targets.append(reward)
		# 	else:
		# 		#off_policy_targets.append(reward + self._gamma * next(target_q_vals))
		# 		off_policy_target = reward + self._gamma * target_q_val
		#
		# 	target = FLAGS.beta * on_policy_target + (1 - FLAGS.beta) * off_policy_target
		#
		# 	#assert np.isfinite(target), "Target not finite!"
		# 	targets.append(off_policy_target)
		#
		# target_q_values = np.asarray(targets)
		#
		# # target_q_values = [reward if this_done else reward + self._gamma * (next_q_value)
		# # 				   for (reward, next_q_value, this_done)
		# # 				   in zip(rewards, next_q_values, dones)]
		# # target_q_values = np.asarray(target_q_values)
		# #logging.info('Fit critic (states): %s\n%s' % (states.shape, states))
		# #logging.info('Fit critic (actions): %s\n%s' % (actions.shape, actions))
		# self._critic_model.fit([states, actions], target_q_values, verbose=0)
		# </editor-fold> old train critic method

		self._update_targets()


	@classmethod
	@abstractmethod
	def action_params(cls):
		pass


	def _generate_tensorflow_session(self):
		"""
		Create Tensorflow session for use
		"""
		config = tf.ConfigProto() #device_count = {'GPU': 2})
		config.gpu_options.allow_growth = True
		return tf.Session(config = config) #tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="2")))


	def _create_actor_model(self, model_name, hidden, lr):
		input_layer = Input(shape=[self.__state_size], name='input_layer')

		layer_size = hidden[0]
		layer = Dense(layer_size, activation='linear', name='dense' + str(layer_size) + '_layer')(input_layer)
		layer = Activation(lambda x: relu(x, alpha=0.01), name='activation_' + str(layer_size))(layer)

		layers = iter(hidden)
		next(layers)

		for layer_size in layers:
			layer = Dense(layer_size, activation='linear', name='dense' + str(layer_size) + '_layer')(layer)
			layer = Activation(lambda x: relu(x, alpha=0.01), name='activation_' + str(layer_size))(layer)

		output_actions = Dense(self.__action_size, activation='linear', name='action_layer')(layer)
		output_action_params = Dense(self.__action_param_size, activation='linear', name='action_param_layer')(layer)
		output_layer = concatenate([output_actions, output_action_params], name='output_layer')

		model = Model(inputs=input_layer, outputs=output_layer, name=model_name)

		#model = multi_gpu_model(model, gpus=3)

		model.compile(loss='mse',
			optimizer=Adam(lr=lr, beta_1= FLAGS.momentum, beta_2=FLAGS.momentum2,
						   clipnorm=FLAGS.clip_grad, decay=0.0, epsilon=0.00000001))
		logging.debug(model_name + ' model:')
		logging.debug(model.summary())

		return model, input_layer


	def _create_critic_model(self, model_name, hidden, lr):
		state_input = Input(shape=[self.__state_size], name='state_inputs')
		action_input = Input(shape=[self.__action_size + self.__action_param_size], name='action_inputs')

		input_layer = concatenate([state_input, action_input], name='input_layer')

		layer_size = hidden[0]
		layer = Dense(layer_size, activation='linear', name='dense' + str(layer_size) + '_layer')(input_layer)
		layer = Activation(lambda x: relu(x, alpha=0.01), name='activation_' + str(layer_size))(layer)

		layers = iter(hidden)
		next(layers)

		for layer_size in layers:
			layer = Dense(layer_size, activation='linear', name='dense' + str(layer_size) + '_layer')(layer)
			layer = Activation(lambda x: relu(x, alpha=0.01), name='activation_' + str(layer_size))(layer)

		output_layer = Dense(1, activation = 'linear', name = 'q_values_layer')(layer)

		model = Model(inputs=[state_input, action_input], outputs=output_layer, name=model_name)

		#model = multi_gpu_model(model, gpus=3)

		model.compile(loss='mse',
			optimizer=Adam(lr=lr, beta_1= FLAGS.momentum, beta_2=FLAGS.momentum2,
						   clipnorm=FLAGS.clip_grad, decay=0.0, epsilon=0.00000001))
		logging.debug(model_name + ' model:')
		logging.debug(model.summary())

		return model, state_input, action_input


	def _get_samples(self, count):
		"""
		Gets a random sample of batch_size from agent's replay memory

		:return: Tuple(List(Float, Boolean))) denoting the sample of states, actions, rewards, next_states, and dones
		"""

		#samples = list(zip(*random.sample(self._memory, self._batch_size)))

		# generate fake samples
		# samples = []
		# for i in range(count):
		# 	samples.append(self._memory[i])

		# states, actions, rewards, q_vals, next_states, terminals = list(zip(*samples)) #samples #list(zip(*samples))
		# logging.debug('Fake actions: %s' % (np.asarray(actions)))

		states, actions, rewards, q_vals, next_states, terminals = list(zip(*random.sample(self._memory, count))) #samples #list(zip(*samples))
		#logging.debug('Next states type: %s %s %s' % (np.asarray(next_states).shape, type(next_states), next_states))
		return np.asarray(states), np.asarray(actions), np.asarray(rewards), \
			np.asarray(q_vals), np.asarray(next_states), np.asarray(terminals)


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


	def _train_critic(self, states, actions, rewards, on_policy_targets, next_states, terminals):
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
		target_q_values = self._target_critic_model.predict([next_states, target_actions])
		#target_q_values = rewards if dones else rewards + self._gamma * next_q_values
		targets = []
		for reward, terminal, target_q_val, on_policy_target in zip(rewards, terminals, target_q_values, on_policy_targets): #, on_policy_targets):
			if terminal:
				off_policy_target = reward
				#off_policy_targets.append(reward)
			else:
				#off_policy_targets.append(reward + self._gamma * next(target_q_vals))
				off_policy_target = reward + self._gamma * target_q_val

			target = FLAGS.beta * on_policy_target + (1 - FLAGS.beta) * off_policy_target

			#assert np.isfinite(target), "Target not finite!"
			targets.append(target)

		target_q_values = np.asarray(targets)

		# target_q_values = [reward if this_done else reward + self._gamma * (next_q_value)
		# 				   for (reward, next_q_value, this_done)
		# 				   in zip(rewards, next_q_values, dones)]
		# target_q_values = np.asarray(target_q_values)
		#logging.info('Fit critic (states): %s\n%s' % (states.shape, states))
		#logging.info('Fit critic (actions): %s\n%s' % (actions.shape, actions))
		self._critic_model.fit([states, actions], target_q_values, verbose=0)


	def _update_targets(self):
		self._update_actor_target()
		self._update_critic_target()


	def _update_actor_target(self):
		actor_weights = self._actor_model.get_weights()
		actor_target_weights = self._target_actor_model.get_weights()

		for i in range(len(actor_weights)):
			actor_target_weights[i] = self._tau * actor_weights[i] + (1 - self._tau) * actor_target_weights[i]

		self._target_actor_model.set_weights(actor_target_weights)


	def _update_critic_target(self):
		critic_weights = self._critic_model.get_weights()
		critic_target_weights = self._target_critic_model.get_weights()

		for i in range(len(critic_weights)):
			critic_target_weights[i] = self._tau * critic_weights[i] + (1 - self._tau) * critic_target_weights[i]

		self._target_critic_model.set_weights(critic_target_weights)
