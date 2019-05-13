#/!usr/bin/env python
# encoding: utf-8

import os
import sys
import time
import threading
import numpy
import datetime

from pathlib import Path

import hfo_game
import ddpg

from hfo import *

import random
import logging

from absl import app
from absl import flags

from ddpg import DDPG as Agent

import timeit

FLAGS = flags.FLAGS

flags.DEFINE_bool('gpu', True, "Use GPU to brew Caffe")
flags.DEFINE_bool('benchmark', False, "Benchmark the network and exit")
flags.DEFINE_bool('learn_offline', False, "Just do updates on a fixed replaymemory.")

# Load/Save Args
flags.DEFINE_string('save', "", "Prefix for saving snapshots")
flags.DEFINE_string('resume', "", "Prefix for resuming from. Default=save_path")
flags.DEFINE_string('actor_weights', "", "The actor pretrained weights load (*.caffemodel).")
flags.DEFINE_string('critic_weights', "", "The critic pretrained weights load (*.caffemodel).")
flags.DEFINE_string('actor_snapshot', "", "The actor solver state to load (*.solverstate).")
flags.DEFINE_string('critic_snapshot', "", "The critic solver state to load (*.solverstate).")
flags.DEFINE_string('memory_snapshot', "", "The replay memory to load (*.replaymemory).")

# Solver Args
flags.DEFINE_string('solver', "Adam", "Solver Type.")
flags.DEFINE_float('momentum', .95, "Solver momentum.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float('momentum2', .999, "Solver momentum2.", lower_bound=0.0,  upper_bound=1.0)
flags.DEFINE_float('actor_lr', .00001, "Solver learning rate.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float('critic_lr', .001, "Solver learning rate.", lower_bound=0.0,  upper_bound=1.0)
flags.DEFINE_float('clip_grad', 10, "Clip gradients.")
flags.DEFINE_string('lr_policy', "fixed", "LR Policy.")
flags.DEFINE_integer('max_iter', 10000000, "Custom max iter.", lower_bound=0) #TODO change to 10000000

# Epsilon-Greedy Args
flags.DEFINE_integer('explore', 10000, "Iterations for epsilon to reach given value.", lower_bound=0)
flags.DEFINE_float('epsilon', .1, "Value of epsilon after explore iterations.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float('evaluate_with_epsilon', 0, "Epsilon value to be used in evaluation mode", lower_bound=0.0, upper_bound=1.0)

# Evaluation Args
flags.DEFINE_bool('evaluate', False, "Evaluation mode: only playing a game, no updates")
flags.DEFINE_integer('evaluate_freq', 10000, "Frequency (steps) between evaluations", lower_bound=0)
flags.DEFINE_integer('repeat_games', 100, "Number of games played in evaluation mode", lower_bound=0)

# Misc Args
flags.DEFINE_float('update_ratio', 0.1, "Ratio of new experiences to updates.", lower_bound=0.0, upper_bound=1.0)

# Sharing
flags.DEFINE_integer('share_actor_layers', 0, "Share layers between actor networks.")
flags.DEFINE_integer('share_critic_layers', 0, "Share layers between critic networks.")
flags.DEFINE_bool('share_replay_memory', False, "Shares replay memory between agents.")

# Game configuration
flags.DEFINE_integer('offense_agents', 1, "Number of agents playing offense")
flags.DEFINE_integer('offense_npcs', 0, "Number of npcs playing offense")
flags.DEFINE_integer('defense_agents', 0, "Number of agents playing defense")
flags.DEFINE_integer('defense_npcs', 0, "Number of npcs playing defense")
flags.DEFINE_integer('offense_dummies', 0, "Number of dummy npcs playing offense")
flags.DEFINE_integer('defense_dummies', 0, "Number of dummy npcs playing defense")
flags.DEFINE_integer('defense_chasers', 0, "Number of chasers playing defense")

class MyFormatter(logging.Formatter):
	converter=datetime.datetime.fromtimestamp
	def formatTime(self, record, datefmt=None):
		ct = self.converter(record.created)
		if datefmt:
			s = ct.strftime(datefmt)
		else:
			t = ct.strftime("%Y-%m-%d %H:%M:%S")
			s = "%s,%03d" % (t, record.msecs)
		return s


def _verify_cmd_args():
	if FLAGS.save == "" and not FLAGS.evaluate:
		logging.error("Save path (or evaluate) required but not set.")
		logging.error("Usage: python3 ddpg_main.y --[evaluate|save [path]]")
		exit(1)

	assert ((FLAGS.critic_snapshot=="" or FLAGS.critic_weights=="") and
	         (FLAGS.actor_snapshot=="" or FLAGS.actor_weights=="")), \
	       "Give a snapshot or weights but not both."


def _set_log_files():

	logging.getLogger().handlers = []

	info_fh = logging.FileHandler(Path(FLAGS.save).parent / (Path(FLAGS.save).stem + ('_ddpg.INFO')))
	warning_fh = logging.FileHandler(Path(FLAGS.save).parent / (Path(FLAGS.save).stem + ('_ddpg.WARNING')))
	error_fh = logging.FileHandler(Path(FLAGS.save).parent / (Path(FLAGS.save).stem + ('_ddpg.ERROR')))
	fatal_fh = logging.FileHandler(Path(FLAGS.save).parent / (Path(FLAGS.save).stem + ('_ddpg.FATAL')))

	info_ch = logging.StreamHandler()
	warning_ch = logging.StreamHandler()
	error_ch = logging.StreamHandler()
	fatal_ch = logging.StreamHandler()

	info_fh.setLevel(logging.INFO)
	warning_fh.setLevel(logging.WARNING)
	error_fh.setLevel(logging.ERROR)
	fatal_fh.setLevel(logging.FATAL)

	info_ch.setLevel(logging.INFO)
	warning_ch.setLevel(logging.WARNING)
	error_ch.setLevel(logging.ERROR)
	fatal_ch.setLevel(logging.FATAL)

	#.%(msecs)d
	#%(relativeCreated)6d
	#formatter = MyFormatter('%(levelname).1s %(asctime)s %(module)s:%(lineno)d] %(message)s', '%m%d %H:%M:%S.%f')
	formatter = MyFormatter('%(asctime)s: %(levelname).1s %(module)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S.%f')


	info_fh.setFormatter(formatter)
	warning_fh.setFormatter(formatter)
	error_fh.setFormatter(formatter)
	fatal_fh.setFormatter(formatter)

	info_ch.setFormatter(formatter)
	warning_ch.setFormatter(formatter)
	error_ch.setFormatter(formatter)
	fatal_ch.setFormatter(formatter)

	logging.getLogger().addHandler(info_fh)
	logging.getLogger().addHandler(warning_fh)
	logging.getLogger().addHandler(error_fh)
	logging.getLogger().addHandler(fatal_fh)

	logging.getLogger().addHandler(info_ch)
	logging.getLogger().addHandler(warning_ch)
	logging.getLogger().addHandler(error_ch)
	logging.getLogger().addHandler(fatal_ch)


def run(agent, state, num_trials):
	for episode in range(num_trials):

		action = agent.act(state)

		#action = action.reshape((1, action.shape[0]))
		#episode += 1
		chosen_act = np.argmax(action[0:4])
		logging.info('Episode %i, Decay: %f Action: %s%s%s' % (episode, agent._epsilon(), chosen_act, action.shape, action))

		next_state = []
		for i in range (0, 59):
			next_state.append(random.uniform(-1.0, 1.0))
		next_state = np.array(next_state)
		#next_state = next_state.reshape((1, next_state.shape[0]))



		if chosen_act == 0:
			reward = 1.0
		else:
			reward = 0.0

		#reward = action[0] * 10#np.random.random()
		logging.info('Reward: %s' % (reward))
		done = False

		agent.remember(state, action, reward, next_state, done)
		agent.train()

		state = next_state

def main(argv):
	_verify_cmd_args()
	_set_log_files()

	actor_hidden = [1024, 512, 256, 128]
	critic_hidden = [1024, 512, 256, 128]

	agent = Agent(59, 4, 6, actor_hidden, critic_hidden,
		FLAGS.actor_lr, FLAGS.critic_lr, FLAGS.epsilon, FLAGS.explore)

	num_trials = 1000
	trial_len = 500

	state = []
	for i in range (0, 59):
		state.append(random.uniform(-1.0, 1.0))
	state = np.array(state)
	#state = state.reshape((1, state.shape[0]))
	#state = np.array([state])
	#state = [state]
	episode = 0

	run(agent, state, num_trials)
	#print(timeit.timeit for x in range(num_trials): run(agent, state, x))
	#timeit.Timer(run(agent, state, num_trials).repeat(repeat=10, number=10))

if __name__ == '__main__':
	app.run(main)
