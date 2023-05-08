import numpy as np
import os
import matplotlib.pyplot as plt 
import torch
class RolloutsProcessor:
    def __init__(self, robot, robot_structure, num_processes):
        self.robot = robot

        self.num_points = robot_structure.shape[0] + 1
        self.structure = robot_structure

        self.one_hot_structure_and_actuator_mask()
        self.get_filled_mask()

        self.unique_indices = None
        self.num_processes = num_processes
    
    def one_hot_structure_and_actuator_mask(self):
        self.one_hot = np.zeros((5, self.num_points, self.num_points))
        for r_idx in range(self.structure.shape[0]):
            for c_idx in range(self.structure.shape[1]):
                self.one_hot[int(self.structure[r_idx, c_idx]), r_idx, c_idx + 1] = 1

        self.actuator_mask = np.where((self.one_hot[3, :] + self.one_hot[4, :]).flatten().astype(int))

    def get_filled_mask(self):
        full = np.full((self.num_points, self.num_points), False)

        for i in range(self.num_points - 1):
            for j in range(self.num_points - 1):
                full[i, j] = full[i, j] or self.structure[i, j] != 0
                full[i, j + 1] = full[i, j + 1] or self.structure[i, j] != 0
                full[i + 1, j] = full[i + 1, j] or self.structure[i, j] != 0
                full[i + 1, j + 1] = full[i + 1, j + 1] or self.structure[i, j] != 0
        
        self.full_mask = np.where(full.flatten())[0]
    
    def get_unique_indices(self, relative_position):
        _, self.unique_indices = np.unique(relative_position.T, return_index=True, axis=0)
        self.unique_indices = sorted(self.unique_indices)
        
    
    def get_obs_at_step(self, relpos, relvel, com_vel, orientation):
        if self.unique_indices is None:
            self.get_unique_indices(relpos)
        
        relative_position = np.zeros((self.num_processes, 2, self.num_points**2))
        relative_position[:, :, self.full_mask] = relpos[:, :, self.unique_indices]
        relative_velocity = np.zeros((self.num_processes, 2, self.num_points**2))
        relative_velocity[:, :, self.full_mask] = relvel[:, :, self.unique_indices]
    
        stacked_state = np.zeros((self.num_processes, 13, self.num_points, self.num_points))
        reshaped_relative_position = relative_position.reshape(self.num_processes, 2, self.num_points, self.num_points)
        reshaped_relative_velocity = relative_velocity.reshape(self.num_processes, 2, self.num_points, self.num_points)
        reshaped_com_vel = np.repeat(com_vel, self.num_points**2, axis=1).reshape(self.num_processes, 2, self.num_points, self.num_points)
        reshaped_orientation = np.repeat(orientation, self.num_points**2, axis=1).reshape(self.num_processes, self.num_points, self.num_points)
        reshaped_one_hot = np.repeat(np.expand_dims(self.one_hot, 0), self.num_processes, axis=0)
        
        stacked_state[:, :5] = reshaped_one_hot
        stacked_state[:, 5:7] = reshaped_relative_position
        stacked_state[:, 7:9] = reshaped_relative_velocity
        stacked_state[:, 9:11] = reshaped_com_vel
        stacked_state[:, 12] = reshaped_orientation
        
        return torch.FloatTensor(stacked_state)
    
    def cnn_action_to_env_action(self, action):
        return action[:, self.actuator_mask[0]]


