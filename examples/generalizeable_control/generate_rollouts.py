import numpy as np
import torch
import os, sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
sys.path.insert(0, root_dir)
from ppo.envs import make_vec_envs
from generalizeable_rollouts import GeneralizeableRollouts

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

def generate_rollouts(
    actor_critic, 
    env_name, 
    robot_structure, 
    seed, 
    num_processes, 
    device,
    path,
    robot,
    num_steps=128):
    
    rollout_storage = GeneralizeableRollouts(path=path, robot=robot, num_steps=num_steps, robot_structure=robot_structure[0])
    
    eval_envs = make_vec_envs(env_name, robot_structure, seed + num_processes, num_processes,
                              None, path, device, True)

    '''
    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms
    '''

    eval_episode_rewards = []
    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)


    for i in range(num_steps):
        object_points_rel_pos = obs[0, 2:].reshape(2, -1)
        object_points_vel = np.array(eval_envs.env_method('object_vel_at_time', eval_envs.env_method('get_time', indices=[0])[0], "robot"))[0]
        object_vel_com = np.mean(object_points_vel, axis=1)
        object_points_rel_vel = object_points_vel - np.array([object_vel_com]).T
        object_orientation = np.array(eval_envs.env_method('get_ort_obs', "robot"))[0]
        
        rollout_storage.update_state(i, object_points_rel_pos, object_points_rel_vel, object_vel_com, object_orientation)

        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        rollout_storage.update_action(i, action)
        
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()
    rollout_storage.save()

if __name__ == '__main__':
    for generation in range(10, 11):
        path = "/home/sadhana/soft-robot-generalizeable-control-policies/data/walker-no-normalization/generation_{}".format(generation)
        os.makedirs(os.path.join(path, "rollouts"), exist_ok=True)
        for robot in range(25):
            actor_critic = torch.load(os.path.join(path, "controller/robot_{}_controller.pt".format(robot)))
            morphology = np.load(os.path.join(path, "structure/{}.npz".format(robot)))
            morphology = morphology.f.arr_0, morphology.f.arr_1
            generate_rollouts(actor_critic[0], "Walker-v0", morphology, 0, 1, "cpu", path, robot)
            print("finished rollout for {}-{}".format(generation, robot))
