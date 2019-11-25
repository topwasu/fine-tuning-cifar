def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def main():
	import os

	wanted_list = ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train']

	dir_path = os.path.dirname(os.path.realpath(__file__))
	dataset_path = os.path.join(dir_path, 'dataset')
	gz_path = os.path.join(dataset_path, 'cifar-100-python', 'meta')

	bdict = unpickle(gz_path)
	indices = []
	for label in wanted_list:
		indices.append(bdict['fine_label_names'.encode()].index(label.encode()))

	print(indices)


if __name__ == '__main__':
	main()


