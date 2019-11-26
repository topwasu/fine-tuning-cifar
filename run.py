import os
import numpy as np
import torch
from torchvision import models, datasets, transforms


def get_cifar100(interested_indices, train):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	dataset_path = os.path.join(dir_path, 'dataset')

	# ref for mean and std: https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
	transform = transforms.Compose([transforms.ToTensor(),
	                                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

	train_dataset = datasets.CIFAR100(dataset_path, train=train, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=4)

	images = [x[0] for x in train_loader if x[1] in interested_indices]
	labels = [interested_indices.index(x[1]) for x in train_loader if x[1] in interested_indices]
	return torch.tensor(images), torch.tensor(labels)
	
	
def get_resnet(num_classes, frozen_layers):
	resnet = models.resnet18(pretrained=True)
	for layer_name in frozen_layers:
		layer = getattr(resnet, layer_name)
		for param in layer.parameters():
			param.requires_grad = False
	resnet.fc = torch.nn.Linear(512, num_classes, bias=True)
	return resnet


def calc_accuracy(logits, labels):
	predicted_labels = np.argmax(logits, axis=1)
	return np.mean(np.equal(predicted_labels, labels))


def train_model(train_images, train_labels, model, num_epochs, lr=1e-3, batch_size=16, verbose=False):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	classification_criterion = torch.nn.CrossEntropyLoss()
	num_len = train_labels.shape[0]
	for ep in range(num_epochs):
		ct = 0
		total_acc = 0
		while ct < num_len:
			batch_images = train_images[ct:min(ct+batch_size, num_len)]
			batch_labels = train_labels[ct:min(ct+batch_size, num_len)]
			
			optimizer.zero_grad()
			
			logits = model(batch_images)
			loss = classification_criterion(logits, batch_labels)
			
			loss.backward()
			optimizer.step()
			
			if verbose:
				total_acc += calc_accuracy(logits, batch_labels) * batch_labels.shape[0]
			
			ct += batch_size
			
		if verbose:
			print(f'Train Epoch {ep} - average accuracy = {total_acc / num_len}')


def test_model(test_images, test_labels, model, batch_size=16):
	num_len = test_labels.shape[0]
	ct = 0
	total_acc = 0
	while ct < num_len:
		batch_images = test_images[ct:min(ct + batch_size, num_len)]
		batch_labels = test_labels[ct:min(ct + batch_size, num_len)]
		
		logits = model(batch_images)
		total_acc += calc_accuracy(logits, batch_labels) * batch_labels.shape[0]
		
		ct += batch_size
	return total_acc / num_len

def main():
	# hyperparameters
	vehicle_1_indices = [8, 13, 48, 58, 90] # run find_out_labels_indices to find this
	frozen_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
	use_gpu = -1

	# get data
	train_images, train_labels = get_cifar100(vehicle_1_indices, True)

	# initialize model
	resnet = get_resnet(len(vehicle_1_indices), frozen_layers)
	
	# train
	train_model(train_images, train_labels, resnet, 100, verbose=True)
	
	# get test data
	test_images, test_labels = get_cifar100(vehicle_1_indices, False)
	
	print(f'Test accuracy = {test_model(test_images, test_labels, resnet)}')


if __name__ == '__main__':
	main()