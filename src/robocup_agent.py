import ddpg
import numpy as np
from hfo import *
import logging
import random

from abc import ABC, abstractmethod



class RoboCupAgent(ddpg.DDPG, ABC):
	def set_unum(self, unum):
		self._unum = unum


	def get_unum(self):
		return self._unum


	# def select_action(self, action_vector):
	# 	#action_vector = self.act(state)
	# 	#action_vector[2] = 0.0
	# 	action_type = np.argmax(action_vector[0:self._action_size])
	# 	arg1 = self._get_param_offset(action_type, 0)
	# 	arg2 = arg1 + 1
	# 	#logging.info("DASH: %i, TURN: %i, TACKLE: %i, KICK: %i, CATCH: %i, Action: %i, Action param: %i" % (DASH, TURN, TACKLE, KICK, CATCH, action, arg1))
	# 	logging.info(self.action_params_string(action_vector))
	# 	return action_type, arg1, arg2

	def random_action(self):
		act = []
		for i in range (0,self.action_size):
			act.append(random.uniform(-1.0, 1.0))

		for action, params in self.action_params().items():
			for param_val in range(params['arg_count']):
				act.append(random.uniform(params['arg_'+str(param_val)]['min'], params['arg_'+str(param_val)]['max']))

		return np.asarray(act)


	@classmethod
	@abstractmethod
	def action_params(cls):
		pass


	def act(self, hfo, action_vector):
		action_index = np.argmax(action_vector[0:self._action_size])
		action = self.action_string(action_index)
		# logging.info('Action Type: Index(%i) Type(%s)' % (action_type_index, action_type))
		arg_index = self._get_param_offset(action_index) + self._action_size
		#arg_1 = arg_0 + 1

		num_params = self.num_params(action) #hfo_lib.numParams(action_type)

		args = []
		for i in range(num_params):
			args.append(action_vector[arg_index+i])

		# lookup enum value of action string (note that our implementation does not require that the actions be enumerated
		# in the same order as the ACTION STRINGS enum type
		action_val = list(ACTION_STRINGS.keys())[list(ACTION_STRINGS.values()).index(action)]

		#logging.info('Action Type: Index(%i) Val(%i) Name(%s) N_Params(%i) %s' % (action_index,action_val, action, num_params, args))
		logging.info('Action: %s  %s' % (action, args))

		#logging.info('Action:', action, '[', *args, sep=',', ']')

		hfo.act(action_val, *args)

		# logging.info('%s', action_vector)
		# logging.info('Action Type: Index(%i) Type(%s) N_Params(%i) %s' % (action_index, action, num_params, args))

		# logging.info(self.action_params_string(action_vector))
		# logging.info('Action: %s, Params: %i' % (ACTION_STRINGS.get(action_type), n_params))
		#
		# if n_params == 1:
		# 	logging.info('Action with 1 param: %i (%f)' % (action_type, action_vector[self._action_size+arg1]))
		# 	hfo.act(action_type, action_vector[self._action_size+arg1])
		# elif n_params == 2:
		# 	logging.info('Action with 2 param: %i (%f, %f)' %
		# 		(action_type, action_vector[self._action_size+arg1], action_vector[self._action_size+arg2]))
		# 	hfo.act(action_type, action_vector[self._action_size+arg1], action_vector[self._action_size+arg2])


	def num_params(self, action):
		return self.action_params()[action]['arg_count']

class StrikerAgent(RoboCupAgent):
	"""
	A Striker agent has 4 different actions with associated parameters:

		Dash: power[-100, 100]
		Turn: direction[-180, 180]
		Tackle: direction[-180, 180]
		Kick: power[0, 100], direction[-180, 180]
	"""

	def __init__(self, id, num_features, actor_lr, critic_lr, epsilon, explore):
		actor_hidden = [1024, 512, 256, 128]
		critic_hidden = [1024, 512, 256, 128]

		super().__init__(id, num_features, 4, 6, actor_hidden, critic_hidden,
				actor_lr, critic_lr, epsilon, explore)


	actions_dict = {
		'Dash' : {'arg_count': 2,
					'offset': 0,
					'arg_0': {'name': 'power', 'min': -100.0, 'max': 100.0, 'bound_method': 'inverting'},
					'arg_1': {'name': 'power', 'min': -180.0, 'max': 180.0, 'bound_method': 'inverting'}
				 },
		'Turn' : {'arg_count': 1,
					'offset': 2,
					'arg_0': {'name': 'direction', 'min': -180.0, 'max': 180.0, 'bound_method': 'inverting'}
				 },
		'Tackle' : {'arg_count': 1,
					'offset': 3,
					'arg_0': {'name': 'direction', 'min': -180.0, 'max': 180.0, 'bound_method': 'inverting'}
				 },
		'Kick' : {'arg_count': 2,
					'offset': 4,
					'arg_0': {'name': 'power', 'min': 0.0, 'max': 100.0, 'bound_method': 'inverting'},
					'arg_1': {'name': 'direction', 'min': -180.0, 'max': 180.0, 'bound_method': 'inverting'}
				 }
	}

	def action_params(cls):
		return StrikerAgent.actions_dict


	def _get_param_index(self, action):
		return self.action_size + self.actions_dict[action]['offset']



			#logging.debug('Action: %s Arg count: %s' % (action, self.actions_dict[action]['arg_count']))
			#logging.debug('Actions iter: %s' % (i))

		#return list(StrikerAgent.actions_dict.keys()).index(action)

		# action = self.action_string(action_val)
		# switcher = {
		# 	"Dash": 0,
		# 	"Turn": 2,
		# 	"Tackle": 3,
		# 	"Kick": 4,
		# }
		# arg = switcher.get(action, -1)
		#
		# if arg == -1:
		# 	logging.critical('Unrecognized action %i' % (action))

		# return arg


	def action_string(self, action_val):
		action_string = list(StrikerAgent.actions_dict.keys())[action_val]

		if action_string == -1:
			logging.fatal('Unrecognized action %i' % (action))

		return action_string


	def action_params_string(self, action_vector):
		act = self.action_string(np.argmax(action_vector[0:self._action_size]))
		return ('Dash(%7.4f, %7.4f)=%7.4f, Turn(%7.4f)=%7.4f, Tackle(%7.4f)=%7.4f, Kick(%7.4f, %7.4f)=%7.4f [%s]' %
				(action_vector[4], action_vector[5], action_vector[0],
				 action_vector[6], action_vector[1],
				 action_vector[7], action_vector[2],
				 action_vector[8], action_vector[9], action_vector[3],
				 act))
