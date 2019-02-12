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
	
def generate():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', '-g', type=int, default=-1)
	parser.add_argument('--sgen', type=str, default=None)
	parser.add_argument('--depth', '-d', type=int, default=5)
	parser.add_argument('--out', '-o', type=str, default='img/')
	parser.add_argument('--num', '-n', type=int, default=10)
	args = parser.parse_args()
	
	sgen = network.StyleBasedGenerator(depth=args.depth)
	print('loading generator model from ' + args.sgen)
	serializers.load_npz(args.sgen, sgen)

#	if args.gpu >= 0:
#		cuda.get_device_from_id(0).use()
#		sgen.to_gpu()

#	xp = sgen.xp

	imgs = []
	z1 = sgen.make_latent(1)
	for i in range(args.num):
		z2 = sgen.make_latent(1)
		
		w1 = sgen.E(z1)
		w2 = sgen.E(z2)
		
		for t in np.linspace(0, 1, 10):
			w = w1 * (1 - t) + w2 * t
			x = sgen.G(w)
			imgs.append(utils.to_image(x[0].data))
			
		z1 = z2

	imgs[0].save('analogy.gif', save_all=True, duration=100, append_images=imgs[1:], loop=True)						
			
if __name__ == '__main__':
	generate()
