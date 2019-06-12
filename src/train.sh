
#rm state0/*
#python3.7 ddpg_main.py --save state0/test --offense_agents 1 --offense_on_ball 0 --beta 0.2 --gpu 1 > /dev/null &

#rm state1/*
#python3.7 ddpg_main.py --save state1/test --offense_agents 1 --offense_on_ball 0 --beta 0.2 --gpu 1 > /dev/null &

python3.7 plot.py state0/test_ddpg.INFO state1/test_ddpg.INFO state2/test_ddpg.INFO state3/test_ddpg.INFO state4/test_ddpg.INFO state5/test_ddpg.INFO state6/test_ddpg.INFO state7/test_ddpg.INFO -x 30000 &

#python3.7 plot.py ../../dqn-hfo/state0/dqn.INFO ../../dqn-hfo/state1/dqn.INFO ../../dqn-hfo/state2/dqn.INFO ../../dqn-hfo/state3/dqn.INFO ../../dqn-hfo/state4/dqn.INFO ../../dqn-hfo/state5/dqn.INFO ../../dqn-hfo/state6/dqn.INFO ../../dqn-hfo/state7/dqn.INFO -e 6 -r 9 -c 16 -x 50000 &

# rm state2/*
# python3.7 ddpg_main.py --save state2/test --offense_agents 1 --offense_on_ball 0 --beta 0.2 --gpu 2 > /dev/null &
#
# rm state3/*
# python3.7 ddpg_main.py --save state3/test --offense_agents 1 --offense_on_ball 0 --beta 0.2 --gpu 2 > /dev/null &

# rm state4/*
# python3.7 ddpg_main.py --save state4/test --offense_agents 1 --offense_on_ball 0 --beta 0.2 --gpu 1 > /dev/null &
#
# rm state5/*
# python3.7 ddpg_main.py --save state5/test --offense_agents 1 --offense_on_ball 0 --beta 0.2 --gpu 1 > /dev/null &
#
# rm state6/*
# python3.7 ddpg_main.py --save state6/test --offense_agents 1 --offense_on_ball 0 --beta 0.2 --gpu 2 > /dev/null &
#
# rm state7/*
# python3.7 ddpg_main.py --save state7/test --offense_agents 1 --offense_on_ball 0 --beta 0.2 --gpu 2 > /dev/null &

## states 0-7 use dash power -100 to 100 in inverting gradients
