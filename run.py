import os
import numpy as np
import torch
from torchvision import models, datasets, transforms


def main():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	dataset_path = os.path.join(dir_path, 'dataset')

	# ref for mean and std: https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

	train_dataset = datasets.CIFAR100(dataset_path, train=True, download=True)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=4)

	# run find_out_labels_indices to find this
	vehicle_1_indices = [8, 13, 48, 58, 90]

	images = [x[0] for x in train_loader if x[1] in vehicle_1_indices]
	labels = [x[1] for x in train_loader if x[1] in vehicle_1_indices]

	resnet = models.resnet18(pretrained=True)
	print(resnet)


if __name__ == '__main__':
	main()