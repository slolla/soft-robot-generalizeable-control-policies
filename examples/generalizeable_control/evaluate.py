import numpy as np
import torch
import os, sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
sys.path.insert(0, root_dir)
from ppo.envs import make_vec_envs
from process_states import RolloutsProcessor
import shutil
import matplotlib.pyplot as plt

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

def evaluate_robot(
    model, 
    env_name, 
    robot_structure, 
    num_processes, 
    device,
    path,
    robot,
    num_evals=1):
    
    rollout_storage = RolloutsProcessor(robot=robot, robot_structure=robot_structure[0], num_processes=num_processes)
    
    eval_envs = make_vec_envs(env_name, robot_structure, 0, num_processes,
                              None, path, device, True)

    '''
    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms
    '''

    eval_episode_rewards = []
    obs = eval_envs.reset()
    
    eval_masks = torch.zeros(num_processes, 1, device=device)


    while len(eval_episode_rewards) < num_evals:
        object_points_rel_pos = obs[:, 2:].reshape(num_processes, 2, -1)
        object_points_vel = np.array(eval_envs.env_method('object_vel_at_time', eval_envs.env_method('get_time', indices=[0])[0], "robot"))
        object_vel_com = np.mean(object_points_vel, axis=-1)
        object_points_rel_vel = object_points_vel - np.expand_dims(object_vel_com, -1)
        object_orientation = np.array(eval_envs.env_method('get_ort_obs', "robot"))
        
        cnn_obs = rollout_storage.get_obs_at_step(object_points_rel_pos, object_points_rel_vel, object_vel_com, object_orientation)
        
        with torch.no_grad():
            action = rollout_storage.cnn_action_to_env_action(model(cnn_obs)[0])

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()
    return np.mean(eval_episode_rewards)