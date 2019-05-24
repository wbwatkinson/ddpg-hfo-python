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

import ddpg

from robocup_agent import RoboCupAgent, StrikerAgent

from statistics import mean, pstdev

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
flags.DEFINE_integer('max_iter', 100000, "Custom max iter.", lower_bound=0) #TODO change to 10000000

# Epsilon-Greedy Args
flags.DEFINE_integer('explore', 10000, "Iterations for epsilon to reach given value.", lower_bound=0)
flags.DEFINE_float('epsilon', .1, "Value of epsilon after explore iterations.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float('evaluate_with_epsilon', 0, "Epsilon value to be used in evaluation mode", lower_bound=0.0, upper_bound=1.0)

# Evaluation Args
flags.DEFINE_bool('evaluate', False, "Evaluation mode: only playing a game, no updates")
flags.DEFINE_integer('evaluate_freq', 1000, "Frequency (steps) between evaluations", lower_bound=0)
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

flags.DEFINE_string('loglevel', 'INFO', "Logging level")

num_agents_connected = 0

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

	logging.getLogger().setLevel(logging.DEBUG)

	#logging.addLevelName(15, 'verbose')

	#logging.getLogger().setLevel(logging.DEBUG)

	#logging.basicConfig(level=getattr(logging, 'DEBUG')) #FLAGS.loglevel.upper()))

	debug_fh = logging.FileHandler(Path(FLAGS.save).parent / (Path(FLAGS.save).stem + ('_ddpg.DEBUG')))
	info_fh = logging.FileHandler(Path(FLAGS.save).parent / (Path(FLAGS.save).stem + ('_ddpg.INFO')))
	warning_fh = logging.FileHandler(Path(FLAGS.save).parent / (Path(FLAGS.save).stem + ('_ddpg.WARNING')))
	error_fh = logging.FileHandler(Path(FLAGS.save).parent / (Path(FLAGS.save).stem + ('_ddpg.ERROR')))
	critical_fh = logging.FileHandler(Path(FLAGS.save).parent / (Path(FLAGS.save).stem + ('_ddpg.CRITICAL')))

	console = logging.StreamHandler()
	# info_ch = logging.StreamHandler()
	# warning_ch = logging.StreamHandler()
	# error_ch = logging.StreamHandler()
	# fatal_ch = logging.StreamHandler()

	#logging.getLogger().setLevel(level=getattr(logging, FLAGS.loglevel.upper())) # logging.DEBUG)
	console.setLevel(getattr(logging, FLAGS.loglevel.upper()))

	debug_fh.setLevel(logging.DEBUG)
	info_fh.setLevel(logging.INFO)
	warning_fh.setLevel(logging.WARNING)
	error_fh.setLevel(logging.ERROR)
	critical_fh.setLevel(logging.CRITICAL)


	#console.setLevel(level=getattr(logging, FLAGS.loglevel.upper())) # logging.DEBUG)
	# info_ch.setLevel(logging.INFO)
	# warning_ch.setLevel(logging.WARNING)
	# error_ch.setLevel(logging.ERROR)
	# fatal_ch.setLevel(logging.FATAL)

	formatter = MyFormatter('%(asctime)s: %(levelname).1s %(module)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S.%f')

	debug_fh.setFormatter(formatter)
	info_fh.setFormatter(formatter)
	warning_fh.setFormatter(formatter)
	error_fh.setFormatter(formatter)
	critical_fh.setFormatter(formatter)

	console.setFormatter(formatter)
	# info_ch.setFormatter(formatter)
	# warning_ch.setFormatter(formatter)
	# error_ch.setFormatter(formatter)
	# fatal_ch.setFormatter(formatter)

	logging.getLogger().addHandler(debug_fh)
	logging.getLogger().addHandler(info_fh)
	logging.getLogger().addHandler(warning_fh)
	logging.getLogger().addHandler(error_fh)
	logging.getLogger().addHandler(critical_fh)

	logging.getLogger().addHandler(console)
	# logging.getLogger().addHandler(info_ch)
	# logging.getLogger().addHandler(warning_ch)
	# logging.getLogger().addHandler(error_ch)
	# logging.getLogger().addHandler(fatal_ch)


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

		logging.info('Reward: %s' % (reward))
		done = False

		agent.remember(state, action, reward, next_state, done)
		agent.train()

		state = next_state


def calculate_epsilon(iter):
	if iter < FLAGS.explore:
		return 1.0 - (1.0 - FLAGS.epsilon) * (iter / FLAGS.explore)
	else:
		return FLAGS.epsilon

def _keep_playing_games(tid, save_prefix, port, thread_lock):
	logging.info("Thread %i port=%i save_prefix=%s" % (tid, port, save_prefix))

	#TODO CPU only mode
	#TODO find latest snapshot

	num_players = (FLAGS.offense_agents + FLAGS.offense_npcs + FLAGS.offense_dummies +
				   FLAGS.defense_agents + FLAGS.defense_npcs + FLAGS.defense_dummies +
				   FLAGS.defense_chasers)

	num_features = hfo_game.num_state_features(num_players)

	num_features = hfo_game.num_state_features(num_players)

	agent = StrikerAgent(tid, num_features, FLAGS.actor_lr, FLAGS.critic_lr, FLAGS.epsilon, FLAGS.explore)

	env = hfo.HFOEnvironment()
	hfo_game.connect_to_server(env, port)
	agent.set_unum(env.getUnum())
	logging.debug("Thread %i port=%i player=%i Connected to HFO!", tid, port, agent.get_unum())

	with thread_lock:
		global num_agents_connected
		num_agents_connected += 1

	# wait for all DDPGs to connect and be ready
	while num_agents_connected < FLAGS.offense_agents:
		# yield to other threads
		time.sleep(1)

	best_score = float('-inf')
	last_eval_iter = agent.iter

	_evaluate(env, agent, tid)

	for episode in range(0, FLAGS.max_iter):
		total_reward = 0
		# play one Episode

		epsilon = calculate_epsilon(agent.iter)
		total_reward, steps, status, extrinsic_reward = _play_one_episode(env, agent, epsilon, True, episode, tid)
		logging.info("[Agent %i] Episode %i reward = %f" % (tid, episode, total_reward))

		n_updates = int(steps * FLAGS.update_ratio)

		for i in range(n_updates):
			agent.train()

		if agent.iter >= last_eval_iter + FLAGS.evaluate_freq:
			avg_score = _evaluate(env, agent, tid)
			if avg_score > best_score:
				logging.info('[Agent %i] New High Score: %f, agent_iter = %i' % (tid, avg_score, agent.iter))
				best_score = avg_score

			last_eval_iter = agent.iter

	del agent
	env.act(QUIT)
	env.step()


def _play_one_episode(env, agent, epsilon, update, episode, tid):
	game = hfo_game.HFOGameState(agent.get_unum())
	#env.act(DASH, 0, 0)
	#game.update(env)
	logging.debug('Episode status: %s' % (hfo.STATUS_STRINGS[game.status]))

	assert not game.episode_over, "Episode should not be over at beginning!"

	episode_states = []
	while not game.episode_over:

		current_state = env.getState()
		assert current_state.shape[0] == agent.state_size, \
			'Current state size mismatch: ' + str(current_state.shape[0]) + ' /= ' + str(agent.state_size)

		#episode_states.append(current_state)

		# select agent action
		action_vector = agent.select_action(current_state, epsilon)
		logging.debug('Episode %i Step %i Actor output: %s' % (episode, game.steps, action_vector))

		# get agent action from vector
		action, params = agent.get_action(action_vector)
		action_val = list(hfo.ACTION_STRINGS.keys())[list(hfo.ACTION_STRINGS.values()).index(action)]
		logging.debug('q_value: %f Action: %s' % (agent.evaluate_action(current_state, action_vector), action))

		# act
		env.act(action_val, *params)

		# update game
		game.update(env)

		reward = game.reward()

		if update:
			next_state = env.getState()
			assert next_state.shape[0] == agent.state_size, \
				'Next state size mismatch: ' + str(next_state.shape[0]) + ' /= ' + str(agent.state_size)
			if game.status == IN_GAME:
				#agent.remember(current_state, action_vector, reward, next_state, False)
				episode_states.append([current_state, action_vector, reward, 0, next_state, False])
			else:
				#agent.remember(current_state, action_vector, reward, None, False)
				# logging.debug('End state: %s' % (current_state))
				#
				# logging.debug('End state: %s' % (next_state))
				# exit(0)
				episode_states.append([current_state, action_vector, reward, 0, next_state, True])

	if update:
		agent.label_transitions(episode_states)
		agent.add_transitions(episode_states)

	state_num = 1
	for state in episode_states:
		_, _, reward, q_val, _, _ = state
		logging.debug('Transition %i: %f, %f' % (state_num, reward, q_val))
		state_num += 1

	#agent.add_transitions(episode_states)
	# logging.debug('Total: %f Steps: %i Status: %s Extrinsic Reward: %f' %
	# 	(game.total_reward, game.steps, hfo.STATUS_STRINGS[game.status], game.extrinsic_reward))
	return game.total_reward, game.steps, game.status, game.extrinsic_reward
		#action_vector = agent.select_action(current_state)

		#agent.act(env, action_vector)
		#hfo.act(act, arg1, arg2)
		#logging.info("State: %ss\n%s" % (type(state), state))
		#exit(0)
		#logging.info("Action: %s\n%s" % (type(action), action))
		#game.update(env)
		#action_type = np.argmax(action_vector[0:4])

		# if action_type == DASH:
		# 	reward = 1.0
		# else:
		# 	reward = 0.0
		#
		# if game.status == IN_GAME:
		# 	done = False
		# else:
		# 	done = True

		#logging.info('Episode %i, Step: %i Decay: %f, Reward: %f' % (episode, game.steps, agent._epsilon(), reward)) # agent._epsilon(),

		#next_state = env.getState()

		#agent.remember(current_state, action_vector, reward, next_state, done)
		#agent.train()

	return 0, game.steps, game.status, 0


def _evaluate(env, agent, tid):
	logging.info('[Agent %i] Evaluating for %i episodes with epsilon = %f' % (tid, FLAGS.repeat_games, FLAGS.evaluate_with_epsilon))
	scores = []
	steps = []
	successful_trial_steps = []
	goals = 0

	for i in range(FLAGS.repeat_games):
		trial_reward, trial_steps, trial_status, _ = _play_one_episode(env, agent, FLAGS.evaluate_with_epsilon, False, i, tid)
		scores.append(trial_reward)
		steps.append(trial_steps)
		if trial_status == GOAL:
			goals += 1
			successful_trial_steps.append(trial_steps)

	score_avg, score_std = np.mean(scores), np.std(scores)
	steps_avg, steps_std = np.mean(steps), np.std(steps)
	successful_steps_avg, successful_steps_std = np.mean(successful_trial_steps), np.std(successful_trial_steps)

	goal_percent = goals / FLAGS.repeat_games

	logging.info('[Agent %i] Evaluation: agent_iter = %i, reward_avg = %f, reward_std = %f, steps_avg = %f, steps_std = %f, success_steps_avg = %f, success_steps_std = %f, goal_perc = %f' %
		(tid, agent.iter, score_avg, score_std, steps_avg, steps_std, successful_steps_avg, successful_steps_std, goal_percent))

	return goal_percent


def main(argv):
	_verify_cmd_args()
	_set_log_files()

	port = random.randint(20000, 59999)

	server_thread = threading.Thread(target=hfo_game.start_hfo_server, args=(
		port, FLAGS.offense_agents + FLAGS.offense_dummies, FLAGS.offense_npcs,
		FLAGS.defense_agents + FLAGS.defense_dummies + FLAGS.defense_chasers, FLAGS.defense_npcs))
	server_thread.start()
	time.sleep(5)

	player_threads = []

	thread_lock = threading.Lock()


	for players in range(FLAGS.offense_agents):
		player_num = len(player_threads)
		save_prefix = Path(FLAGS.save).parent / (Path(FLAGS.save).stem + "_agent" + str(player_num))
		#save_prefix = os.path.dirname(FLAGS.save) + "/_agent" + str(player_num)
		player_threads.append(threading.Thread(target=_keep_playing_games, args=(player_num, save_prefix, port, thread_lock)))
		player_threads[-1].start()
		time.sleep(10)

	for players in range(FLAGS.offense_dummies):
		player_threads.append(threading.Thread(target=hfo_game.start_dummy_teammate, args=(port,)))
		player_threads[-1].start()

	for players in range(FLAGS.defense_dummies):
		player_threads.append(threading.Thread(target=hfo_game.start_dummy_goalie, args=(port,)))
		player_threads[-1].start()


	for players in range(FLAGS.defense_chasers):
		if players == 0:
			player_threads.append(threading.Thread(target=hfo_game.start_chaser, args=(port, "base_right", 1)))
		else:
			player_threads.append(threading.Thread(target=hfo_game.start_chaser, args=(port, "base_right", 0)))
		player_threads[-1].start()


	for thread in player_threads:
		thread.join()


	server_thread.join()

	# exit(0)
	#
	# actor_hidden = [1024, 512, 256, 128]
	# critic_hidden = [1024, 512, 256, 128]
	#
	# agent = Agent(59, 4, 6, actor_hidden, critic_hidden,
	# 	FLAGS.actor_lr, FLAGS.critic_lr, FLAGS.epsilon, FLAGS.explore)
	#
	# num_trials = 1000
	# trial_len = 500
	#
	# state = []
	# for i in range (0, 59):
	# 	state.append(random.uniform(-1.0, 1.0))
	# state = np.array(state)
	# #state = state.reshape((1, state.shape[0]))
	# #state = np.array([state])
	# #state = [state]
	# episode = 0
	#
	# run(agent, state, num_trials)
	#print(timeit.timeit for x in range(num_trials): run(agent, state, x))
	#timeit.Timer(run(agent, state, num_trials).repeat(repeat=10, number=10))

if __name__ == '__main__':
	app.run(main)
