#/!usr/bin/env python
# encoding: utf-8

import itertools
import subprocess
import os

from hfo import *

from absl import app
from absl import flags

import logging

import time
import math

FLAGS = flags.FLAGS

bin_path = "../bin/"

kPassVelThreshold = 0.5

#TODO change frames per trial to 500
flags.DEFINE_string('server_cmd', bin_path + "HFO --fullstate --frames-per-trial 500", "Command executed to start the HFO server.")
flags.DEFINE_string('config_dir', bin_path + "formations-dt", "Directory containing HFO config files.")
flags.DEFINE_bool('gui', False, "Open a GUI window.")
flags.DEFINE_bool('log_game', False, "Log the HFO game.")
flags.DEFINE_string('server_addr', "localhost", "Address of rcssserver.")
flags.DEFINE_string('team_name', "base_left", "Name of team for agents.")
flags.DEFINE_bool('play_goalie', False, "Should the agent play goalie.")
flags.DEFINE_string('record_dir', "", "Directory to record states,actions,rewards.")
flags.DEFINE_float('ball_x_min', 0, "Ball X-Position initialization minimum.")
flags.DEFINE_float('ball_x_max', 0.2, "Ball X-Position initialization maximum.")
flags.DEFINE_float('ball_y_min', -0.8, "Ball Y-Position initialization minimum.", lower_bound=-1.0, upper_bound=1.0)
flags.DEFINE_float('ball_y_max', 0.8, "Ball Y-Position initialization maximum.", lower_bound=-1.0, upper_bound=1.0)
flags.DEFINE_integer('offense_on_ball', 0, "Offensive player to give the ball to.")
flags.DEFINE_bool('verbose', True, "Server prints verbose output.") # TODO Change to false


def start_hfo_server(port, offense_agents, offense_npcs, defense_agents, defense_npcs):
	cmd = (FLAGS.server_cmd +
	       " --port " + str(port) +
	       " --offense-agents " + str(offense_agents) +
	       " --offense-npcs " + str(offense_npcs) +
	       " --defense-agents " + str(defense_agents) +
	       " --defense-npcs " + str(defense_npcs) +
	       " --ball-x-min " + str(FLAGS.ball_x_min) +
	       " --ball-x-max " + str(FLAGS.ball_x_max) +
	       " --ball-y-min " + str(FLAGS.ball_y_min) +
	       " --ball-y-max " + str(FLAGS.ball_y_max) +
	       " --offense-on-ball " + str(FLAGS.offense_on_ball))

	if not FLAGS.gui:
		cmd += " --headless"

	if not FLAGS.log_game:
		cmd += " --no-logging"

	if not FLAGS.verbose:
		cmd += " --verbose"

	logging.debug('Starting server with command: %s ', cmd)

	try:
		# os.system(cmd)
		subprocess.run(cmd, shell=True)
	except(OSError):
		logging.error('Unable to start the HFO server.')
	time.sleep(10)


def start_dummy_teammate(port):
	cmd = bin_path + 'dummy_teammate ' + str(port)
	cmd += ' > /dev/null'
	logging.info("Starting dummy teammate with command: %s", cmd)
	try:
		subprocess.run(cmd, shell=True)
	except(OSError):
		logging.error('Unable to start dummy teammate.')
	time.sleep(5)

def start_dummy_goalie(port):
	cmd = bin_path + 'dummy_goalie ' + str(port)
	cmd += ' > /dev/null'
	logging.info('Starting dummy goalie with command: %s', cmd)
	try:
		subprocess.run(cmd, shell=True)
	except(OSError):
		logging.error('Unable to start dummy goalie.')
	time.sleep(5)


def start_chaser(port, team_name, goalie):
	cmd = bin_path + 'chaser ' + str(port) + ' ' + team_name + ' ' + str(goalie)
	cmd += ' > /dev/null'
	logging.info('Starting chaser with command: %s', cmd)
	try:
		subprocess.run(cmd, shell=True)
	except(OSError):
		logging.error('Unable to start chaser.')
	time.sleep(5)


def stop_HFO_server():
	pass


def get_random_action():
	pass


def num_state_features(num_players):
	return 50 + 9 * num_players


def connect_to_server(hfo_env, port):
	logging.debug("Trying to connect at port %i", port)
	logging.debug("Connecting with params %s, %i, %s, %s, %s, %s", FLAGS.config_dir, port, FLAGS.server_addr, FLAGS.team_name, FLAGS.play_goalie, FLAGS.record_dir)
	hfo_env.connectToServer(LOW_LEVEL_FEATURE_SET,
							FLAGS.config_dir,
						 	port,
						 	FLAGS.server_addr,
						 	FLAGS.team_name,
						 	FLAGS.play_goalie,
						 	FLAGS.record_dir)
	logging.debug("Connected on port %i", port)
	time.sleep(5)

class HFOGameState(object):

	def __init__(self, unum):

		self.__old_ball_prox = 0.0
		self.__ball_prox_delta = 0.0
		self.__old_kickable = 0.0
		self.__kickable_delta = 0.0
		self.__old_ball_dist_goal = 0.0
		self.__ball_dist_goal_delta = 0.0
		self.__steps = 0
		self.__total_reward = 0.0
		self.__extrinsic_reward = 0.0
		self.__status = IN_GAME
		self.__episode_over = False
		self.__got_kickable_reward = False
		self.__our_unum = unum
		self.__pass_active = False

		self.__old_player_on_ball = None
		self.__player_on_ball = None

		logging.debug("Creating new HFOGameState")


	def __del__(self):
		logging.debug("Destroying HFOGameState")


	# <editor-fold> Properties
	@property
	def old_ball_prox(self):
		return self.__old_ball_prox


	@property
	def ball_prox_delta(self):
		return self.__ball_prox_delta


	@property
	def old_kickable(self):
		return self.__old_kickable


	@property
	def kickable_delta(self):
		return self.__kickable_delta


	@property
	def old_ball_dist_goal(self):
		return self.__old_ball_dist_goal


	@property
	def ball_dist_goal_delta(self):
		return self.__ball_dist_goal_delta


	@property
	def steps(self):
		return self.__steps


	@property
	def total_reward(self):
		return self.__total_reward


	@property
	def extrinsic_reward(self):
		return self.__extrinsic_reward


	@property
	def status(self):
		return self.__status


	@property
	def episode_over(self):
		return self.__episode_over


	@property
	def got_kickable_reward(self):
		return self.__got_kickable_reward


	@property
	def our_unum(self):
		return self.__our_unum


	@property
	def pass_active(self):
		return self.__pass_active


	@property
	def old_player_on_ball(self):
		return self.__old_player_on_ball


	@property
	def player_on_ball(self):
		return self.__player_on_ball

	# </editor-fold> Properties


	def update(self, hfo):
		self.__status = hfo.step()
		if self.status == SERVER_DOWN:
			logging.critical("Server Down!")
			exit(1)
		elif self.status != IN_GAME:
			self.__episode_over = True

		current_state = hfo.getState()
		ball_proximity = current_state[53]
		goal_proximity = current_state[15]
		ball_dist = 1.0 - ball_proximity
		goal_dist = 1.0 - goal_proximity
		kickable = current_state[12]
		ball_ang_sin_rad = current_state[51]
		ball_ang_cos_rad = current_state[52]
		ball_ang_rad = math.acos(ball_ang_cos_rad)
		if ball_ang_sin_rad < 0:
			ball_ang_rad *= -1.0
		goal_ang_sin_rad = current_state[13]
		goal_ang_cos_rad = current_state[14]
		goal_ang_rad = math.acos(goal_ang_cos_rad)
		if goal_ang_sin_rad < 0:
			goal_ang_rad *= -1.0

		alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
		ball_dist_goal = math.sqrt(ball_dist * ball_dist +
							  	   goal_dist * goal_dist -
							  	   2.0 * ball_dist * goal_dist * math.cos(alpha))

		logging.debug('Step: %i, BallProx: %f, GoalProx: %f, BallAngSinRad: %f, BallAngCosRad: %f, GoalAngSinRad: %f, GoalAngCosRad: %f' %
			(self.steps, ball_proximity, goal_proximity, ball_ang_sin_rad, ball_ang_cos_rad, goal_ang_sin_rad, goal_ang_cos_rad))
		logging.debug("BallProx: %f BallDistGoal: %f" % (ball_proximity, ball_dist_goal))

		ball_vel_valid = current_state[54]
		ball_vel = current_state[55]
		if ball_vel_valid and ball_vel > kPassVelThreshold:
			self.__pass_active = true

		if self.steps > 0:
			self.__ball_prox_delta = ball_proximity - self.__old_ball_prox
			self.__kickable_delta = kickable - self.__old_kickable
			self.__ball_dist_goal_delta = ball_dist_goal - self.__old_ball_dist_goal

		self.__old_ball_prox = ball_proximity
		self.__old_kickable = kickable
		self.__old_ball_dist_goal = ball_dist_goal

		if self.episode_over:
			self.__ball_prox_delta = 0.0
			self.__kickable_delta = 0.0
			self.__ball_dist_goal_delta = 0.0

		self.__old_player_on_ball = self.__player_on_ball
		self.__player_on_ball = hfo.playerOnBall()
		logging.debug("Player on Ball: %i", self.__player_on_ball.unum)
		# logging.debug("Player on Ball side: %i", self.__player_on_ball.side)
		# logging.debug("Old Player on Ball side: %i", self.__old_player_on_ball.side)
		self.__steps += 1


	def reward(self):
		move_to_ball_reward = self._move_to_ball_reward()
		kick_to_goal_reward = 3.0 * self._kick_to_goal_reward()
		#pass_reward = 3 * self._pass_reward()
		eot_reward = self._eot_reward()
		reward = move_to_ball_reward + kick_to_goal_reward + eot_reward
		self.__extrinsic_reward += eot_reward
		self.__total_reward += reward
		logging.debug('Step: %i Overall Reward: %f MTB: %f KTG: %f EOT: %f' %
					 (self.steps, reward, move_to_ball_reward, kick_to_goal_reward, eot_reward))
		return reward


	def _move_to_ball_reward(self):
		reward = 0.0

		if self.__player_on_ball.unum < 0 or self.__player_on_ball.unum == self.__our_unum:
			reward += self.__ball_prox_delta

		if self.__kickable_delta >= 1 and not self.__got_kickable_reward:
			reward += 1.0
			self.__got_kickable_reward = True

		return reward


	def _kick_to_goal_reward(self):
		if self.__player_on_ball.unum == self.__our_unum:
			return -self.__ball_dist_goal_delta
		elif self.__got_kickable_reward:
			return 0.2 * -self.__ball_dist_goal_delta
		return 0


	def _eot_reward(self):
		if self.__status == GOAL:
			#assert (self.__old_player_on_ball.side == LEFT), 'Unexpected side: {}'.format(self.__old_player_on_ball.side)
			assert (self.__player_on_ball.side == LEFT), 'Unexpected side: {}'.format(self.__player_on_ball.side)

			if self.__player_on_ball.unum == self.__our_unum:
				logging.debug('We Scored!')
				return 5.0
			else:
				logging.debug('Teammate Scored')
				logging.debug('Teammate: %i Self: %i' % (self.__player_on_ball.unum, self.__our_unum))
				exit(0)
				return 1.0

		elif self.__status == CAPTURED_BY_DEFENSE:
			return 0.0

		return 0.0


	def _pass_reward(self):
		if self.__pass_active and self.__player_on_ball.unum > 0 and self.__player_on_ball.unum != self.__old_player_on_ball.unum:
			self.__pass_active = False
			logging.debug('Unum %i steps %i got pass reward!' % (self.__our_unum, self.__steps))
			return 1.0
		return 0


	def _EOT_reward(self):
		return 0.0
