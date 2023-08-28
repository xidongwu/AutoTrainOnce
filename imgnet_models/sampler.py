import torch
import torch.utils.data
import numpy as np


class ImbalancedAccuracySampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None, c=10):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        print(type(self.indices))
        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration

        self.num_c = c
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        self.acc_range(dataset)
        # distribution of classes in the dataset
        print(self.gap_list)

        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)

            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        print(label_to_count.values())
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

        # print(self.weights)
        
    def _get_label(self, dataset, idx):
        current_acc = dataset.acc_list[idx]
        for c in range(1, self.num_c+1):
            if self.gap_list[c-1] <= current_acc <= self.gap_list[c]:
                return c



    def acc_range(self, dataset):
        self.acc_range = dataset.acc_list[0:len(dataset)].max() - dataset.acc_list[0:len(dataset)].min()
        gap = self.acc_range/self.num_c
        self.gap_list = [dataset.acc_list[0:len(dataset)].min().item()]

        end_acc = self.gap_list[0] + gap
        for i in range(self.num_c):
            self.gap_list.append(end_acc.item())
            end_acc += gap


    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
