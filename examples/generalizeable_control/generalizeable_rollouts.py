import numpy as np
import os
import matplotlib.pyplot as plt 
class GeneralizeableRollouts:
    def __init__(self, path, robot, num_steps, robot_structure):
        self.robot = robot
        self.num_steps = num_steps
        self.output_path = os.path.join(path, "rollouts", str(robot))

        self.num_points = robot_structure.shape[0] + 1
        self.relative_position = np.zeros((num_steps, self.num_points**2, 2))
        self.relative_velocity = np.zeros((num_steps, self.num_points**2, 2))
        self.com_vel = np.zeros((num_steps, 2))
        self.orientation = np.zeros((num_steps, 1))
        self.action = np.zeros((num_steps, self.num_points**2))

        self.structure = robot_structure

        self.one_hot_structure_and_actuator_mask()
        self.get_filled_mask()

        self.unique_indices = None
    
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
        
        self.full_mask = np.where(full.flatten())
    
    def get_unique_indices(self, relative_position):
        _, self.unique_indices = np.unique(relative_position.T, return_index=True, axis=0)
        
    
    def update_state(self, step, relative_position, relative_velocity, com_vel, orientation):
        if self.unique_indices is None:
            self.get_unique_indices(relative_position)
        
        self.relative_position[step][self.full_mask] = relative_position.T[self.unique_indices]
        self.relative_velocity[step][self.full_mask] = relative_velocity.T[self.unique_indices]
        self.com_vel[step] = com_vel
        self.orientation[step] = orientation
    
    def update_action(self, step, action):
        self.action[step][self.actuator_mask] = action
        
    
    def reshape(self):
        stacked_state = np.zeros((self.num_steps, 13, self.num_points, self.num_points))
        reshaped_relative_position = self.relative_position.reshape(self.num_steps, 2, self.num_points, self.num_points)
        reshaped_relative_velocity = self.relative_velocity.reshape(self.num_steps, 2, self.num_points, self.num_points)
        reshaped_com_vel = np.repeat(self.com_vel, self.num_points**2, axis=1).reshape(self.num_steps, 2, self.num_points, self.num_points)
        reshaped_orientation = np.repeat(self.orientation, self.num_points**2, axis=1).reshape(self.num_steps, self.num_points, self.num_points)
        reshaped_one_hot = np.repeat(np.expand_dims(self.one_hot, 0), self.num_steps, axis=0)
        
        stacked_state[:, :5] = reshaped_one_hot
        stacked_state[:, 5:7] = reshaped_relative_position
        stacked_state[:, 7:9] = reshaped_relative_velocity
        stacked_state[:, 9:11] = reshaped_com_vel
        stacked_state[:, 12] = reshaped_orientation
        
        action = self.action.reshape(self.num_steps, self.num_points, self.num_points)

        return stacked_state, action

    def save(self, path_add=''):
        final_state, final_action = self.reshape()
        np.savez(self.output_path + str(path_add) + ".npz", state=final_state, action=final_action)


