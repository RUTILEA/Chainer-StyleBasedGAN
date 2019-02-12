import numpy as np
import argparse
import os
from PIL import Image
import chainer
from chainer import serializers
from chainer import Variable
from chainer import cuda
import dataset
import network
import utils
	
N = 5
L = 128
depth = 5

def generate():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', '-g', type=int, default=-1)
	parser.add_argument('--sgen', type=str, default=None)
	parser.add_argument('--depth', '-d', type=int, default=5)
	parser.add_argument('--out', '-o', type=str, default='img/')
#	parser.add_argument('--num', '-n', type=int, default=10)
	args = parser.parse_args()
	
	sgen = network.StyleBasedGenerator(depth=args.depth)
	print('loading generator model from ' + args.sgen)
	serializers.load_npz(args.sgen, sgen)

#	if args.gpu >= 0:
#		cuda.get_device_from_id(0).use()
#		sgen.to_gpu()

#	xp = sgen.xp

	dst = Image.new(mode='RGB', size=(L * (N + 1), L * (N + 1)))
	array_z = sgen.make_latent(N * 2)
	array_w = sgen.E(array_z)
	array_x = sgen.G(array_w)

	for i in range(N):
		dst.paste(utils.to_image(array_x[i].data), (L * (i + 1), 0))
		dst.paste(utils.to_image(array_x[i + N].data), (0, L * (i + 1)))

	for i in range(N):
		for j in range(N):
			print(i, j)
			half = depth // 2
			ws = [array_w[np.newaxis, i]] * half + [array_w[np.newaxis, j + N]] * (depth + 1 - half)

			x = sgen.G.style_mixing(ws)
			dst.paste(utils.to_image(x[0].data), (L * (i + 1), L * (j + 1)))
	
	dst.save('table.jpg')
			
if __name__ == '__main__':
	generate()
