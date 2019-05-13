#/!usr/bin/env python
# encoding: utf-8

import itertools
import subprocess
import os

from hfo import *

from absl import app
from absl import flags
from absl import logging

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

	logging.info('Starting server with command: %s ', cmd)

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
	logging.info("Trying to connect at port %i", port)
	logging.info("Connecting with params %s, %i, %s, %s, %s, %s", FLAGS.config_dir, port, FLAGS.server_addr, FLAGS.team_name, FLAGS.play_goalie, FLAGS.record_dir)
	hfo_env.connectToServer(LOW_LEVEL_FEATURE_SET,
							FLAGS.config_dir,
						 	port,
						 	FLAGS.server_addr,
						 	FLAGS.team_name,
						 	FLAGS.play_goalie,
						 	FLAGS.record_dir)
	logging.info("Connected on port %i", port)
	time.sleep(5)

class HFOGameState():

	def __init__(self, unum):
		self._old_ball_prox = 0
		self._ball_prox_delta = 0
		self._old_kickable = 0 
		self._kickable_delta = 0
		self._old_ball_dist_goal = 0
		self._ball_dist_goal_delta = 0
		self._steps = 0
		self._total_reward = 0
		self._extrinsic_reward = 0
		self._status = IN_GAME
		self.episode_over = False
		self._got_kickable_reward = False
		self._our_unum = unum
		self._pass_active = False

		self._old_player_on_ball = hfo.Player()
		self._player_on_ball = hfo.Player()

		logging.info("Creating new HFOGameState")

		#		self._hfo = HFOEnvironment
		self.total_reward = 0
		self.steps = 0
		self.status = 0
		self.extrinsic_reward = 0

	def update(self, hfo):
		self._status = hfo.step()
		if self._status == SERVER_DOWN:
			logging.fatal("Server Down!")
			exit(1)
		elif self._status != IN_GAME:
			self.episode_over = True
		
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

		alpha = max(ball_ang_rad, goal_ang_rad) * min(ball_ang_rad, goal_ang_rad)
		ball_dist_goal = math.sqrt(ball_dist * ball_dist + 
							  	   goal_dist * goal_dist - 
							  	   2.0 * ball_dist * goal_dist * math.cos(alpha))
		logging.info("BallProx: %f BallDistGoal: %f", ball_proximity, ball_dist_goal)

		ball_vel_valid = current_state[54]
		ball_vel = current_state[55]
		if ball_vel_valid and ball_vel > kPassVelThreshold:
			self._pass_active = true

		if self.steps > 0:
			self._ball_prox_delta = ball_proximity
			self._kickable_delta = kickable - self._old_kickable
			self._ball_dist_goal_delta = ball_dist_goal - self._old_ball_dist_goal

		self._old_ball_prox = ball_proximity
		self._old_kickable = kickable
		self._old_ball_dist_goal = ball_dist_goal

		if self.episode_over:
			self._ball_prox_delta = 0
			self._kickable_delta = 0
			self._ball_dist_goal_delta = 0

		self._old_player_on_ball = self._player_on_ball
		self._player_on_ball = hfo.playerOnBall()
		logging.info("Player on Ball: %i", self._player_on_ball.unum)
		self.steps += 1


	def reward(self):
		#TODO reward
		return 0;


	def __del__(self):
		logging.info("Destroying HFOGameState")

