import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from model import UniversalController

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(self, output, target, mask):
        loss = (torch.flatten(output) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        loss = torch.sum(loss)/torch.sum(mask)
        return loss

class GeneralizeableControllerTrainer:
    def __init__(self, data_path, generations, within_robot=False, sequential=True):
        self.data_path = data_path
        self.within_robot = within_robot
        self.sequential=sequential
        self.generations = generations
        self.actor_loss = MaskedMSELoss()
        self.model = UniversalController(13)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print("USING", self.device)
    
    def create_across_robot_datasets(self):
        if self.sequential:
            train_gens = np.arange(self.generations)
            test_gens = [self.generations]
        else:
            test_gens = [np.random.choice(self.generations)]
            train_gens = [i for i in np.arange(self.generations) if i not in test_gens]
        
        train_dataset = self.get_dataset_from_gens(train_gens)
        test_dataset = self.get_dataset_from_gens(test_gens)
        return train_dataset, test_dataset
    
    def create_within_robot_datasets(self):
        train_states = []
        train_actions = []
        train_masks = []

        test_states = []
        test_actions = []
        test_masks = []

        for gen in range(self.generations):
            path = os.path.join(self.data_path, "generation_{}/rollouts/*.npz".format(gen))
            filenames = glob.glob(path)
            for f in filenames:
                rollout = np.load(f)
                state, action = rollout["state"], rollout["action"]
                mask = state[:, 3, :, :,] + state[:, 4, :, :]
                train_indices = np.random.choice(len(state), int(0.7*len(state)))
                test_indices = [i for i in range(len(state)) if i not in train_indices]

                train_states.extend(state[train_indices])
                test_states.extend(state[test_indices])

                train_actions.extend(action[train_indices])
                test_actions.extend(action[test_indices])

                train_masks.extend(mask[train_indices])
                test_masks.extend(mask[test_indices])
        
        train_dataset = TensorDataset(torch.FloatTensor(train_states), torch.FloatTensor(train_actions), torch.FloatTensor(train_masks))
        test_dataset = TensorDataset(torch.FloatTensor(test_states), torch.FloatTensor(test_actions), torch.FloatTensor(test_masks))
        return train_dataset, test_dataset
        
    def get_dataset_from_gens(self, gens):
        all_states = []
        all_actions = []
        all_masks = []

        for gen in gens:
            path = os.path.join(self.data_path, "generation_{}/rollouts/*.npz".format(gen))
            filenames = glob.glob(path)
            for f in filenames:
                rollout = np.load(f)
                all_states.extend(rollout["state"])
                all_actions.extend(rollout["action"])
                all_masks.extend(rollout["state"][:, 3, :, :,] + rollout["state"][:, 4, :, :])

        return TensorDataset(torch.FloatTensor(all_states), torch.FloatTensor(all_actions), torch.FloatTensor(all_masks))

    def _train(self, dataloader, epoch):
        self.model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            states, actions, masks = data
            states, actions, masks = states.to(self.device), actions.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            output_actions, _ = self.model(states)
            
            actor_loss = self.actor_loss(output_actions, actions, masks)
            loss = actor_loss
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
        print("dividing by", i)
        print('epoch', epoch, 'loss', epoch_loss/i)
        return epoch_loss/(i * 16)
        
    
    def _test(self, dataloader, epoch):
        self.model.eval()
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            states, actions, masks = data
            states, actions, masks = states.to(self.device), actions.to(self.device), masks.to(self.device)
            
            with torch.no_grad():

                output_actions, _ = self.model(states)
                actor_loss = self.actor_loss(output_actions, actions, masks)
                loss = actor_loss

            running_loss += loss.item()
        print("dividing by", i)
        print("testing epoch", epoch, "loss: ", running_loss/i)
        return running_loss/(i * 16)
        
    def collect_and_train(self):
        if self.within_robot:
            train_rollout_dataset, test_rollout_dataset = self.create_within_robot_datasets()
        else:
            train_rollout_dataset, test_rollout_dataset = self.create_across_robot_datasets()
        
        train_rollout_dataloader = DataLoader(train_rollout_dataset, shuffle=True, batch_size=16)
        test_rollout_dataloader = DataLoader(test_rollout_dataset, shuffle=True, batch_size=16)

        train_losses = []
        test_losses = []
        num_epochs = 50
        for epoch in range(num_epochs):
            train_losses.append(self._train(train_rollout_dataloader, epoch))
            test_losses.append(self._test(test_rollout_dataloader, epoch))
        plt.plot(np.arange(len(train_losses)), train_losses, label="train")
        plt.plot(np.arange(len(train_losses)), test_losses, label="test")
        plt.savefig("/home/sadhana/soft-robot-generalizeable-control-policies/examples/generalizeable_control/test.png")

trainer = GeneralizeableControllerTrainer("/home/sadhana/soft-robot-generalizeable-control-policies/data/walker-no-normalization", 10, within_robot=True)
trainer.collect_and_train()