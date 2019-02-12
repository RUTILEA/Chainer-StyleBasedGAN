import numpy as np
import chainer
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.initializers as init

class Const(chainer.Link):
	def __init__(self, initializer, size):
		super(Const, self).__init__()
		with self.init_scope():
			self.b = chainer.Parameter(initializer, size)

	def forward(self):
		return self.b

class NoiseAdder(Chain):
	def __init__(self, ch):
		super(NoiseAdder, self).__init__()
		with self.init_scope():
			self.s = Const(init.Zero(), (1, ch, 1, 1))
	
	def __call__(self, x):
		z = self.xp.random.normal(size=(x.shape[0], 1, x.shape[2], x.shape[3])).astype(np.float32)
		z = F.broadcast_to(z, x.shape)
		s = F.broadcast_to(self.s(), x.shape)
		return x + s * z
#		return x
		
class AdaIN(Chain):
	def __init__(self, z_dim, ch_out):
		w = init.Normal(1.0)
		super(AdaIN, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(z_dim, ch_out, initialW=w, initial_bias=init.One())
			self.l2 = L.Linear(z_dim, ch_out, initialW=w)
	
	def __call__(self, x, w):
		batch = x.shape[0]
		ch_out = x.shape[1]
		style_s = F.broadcast_to(self.l1(w).reshape(batch, ch_out, 1, 1), (batch, ch_out, x.shape[2], x.shape[3]))
		style_b = F.broadcast_to(self.l2(w).reshape(batch, ch_out, 1, 1), (batch, ch_out, x.shape[2], x.shape[3]))

		m = F.broadcast_to(F.mean(x, axis=1, keepdims=True), x.shape)
		v = F.mean((x - m) * (x - m), axis=1, keepdims=True)
		s = F.broadcast_to(F.sqrt(v + 1e-8), x.shape)
		return style_s * (x - m) / s + style_b

class ConvLayer(Chain):
	def __init__(self, ch_in, ch_out, ksize=3, stride=1, pad=1):
		w = init.Normal(1.0)
		super(ConvLayer, self).__init__()
		with self.init_scope():
			self.conv = L.Convolution2D(ch_in, ch_out, ksize, stride, pad, initialW=w)

		self.c = np.sqrt(2.0 / (ch_in * ksize * ksize))
	  
	def __call__(self, x):
		h = x * self.c
		h = self.conv(h)
		return h
		
class GFirstBlock(Chain):
	def __init__(self, ch_in, ch_out, z_dim, outsize):
		self.size = (ch_in,) + outsize
		super(GFirstBlock, self).__init__()
		with self.init_scope():
			self.x = Const(chainer.initializers.Zero(), (1,) + self.size)
			self.n1 = NoiseAdder(ch_in)
			self.n2 = NoiseAdder(ch_out)
			self.a1 = AdaIN(z_dim, ch_in)
			self.a2 = AdaIN(z_dim, ch_out)
			self.c1 = ConvLayer(ch_in, ch_out)
			self.toRGB = ConvLayer(ch_out, 3, ksize=1, pad=0)
		
	def __call__(self, w, last=False):
		batch = w.shape[0]
		x = F.broadcast_to(self.x(), (batch,) + self.size)
		l1 = self.a1(self.n1(x), w)
		l2 = F.leaky_relu(self.c1(l1))
		l3 = self.a2(self.n2(l2), w)
		if last:
			return self.toRGB(l3)
		return l3
	
class GBlock(Chain):
	def __init__(self, ch_in, ch_out, z_dim, outsize):
		super(GBlock, self).__init__()
		with self.init_scope():
			self.n1 = NoiseAdder(ch_out)
			self.n2 = NoiseAdder(ch_out)
			self.a1 = AdaIN(z_dim, ch_out)
			self.a2 = AdaIN(z_dim, ch_out)
			self.c1 = ConvLayer(ch_in, ch_out)
			self.c2 = ConvLayer(ch_out, ch_out)
			self.toRGB = ConvLayer(ch_out, 3, ksize=1, pad=0)

		self.outsize = outsize

	def __call__(self, x, w, last=False):
		l1 = F.resize_images(x, (x.shape[2] * 2, x.shape[3] * 2))
		l2 = F.leaky_relu(self.c1(l1))
		l3 = self.a1(self.n1(l2), w)
		l4 = F.leaky_relu(self.c2(l3))
		l5 = self.a2(self.n2(l4), w)
		if last:
			return self.toRGB(l5)
		return l5

class StyleEncoder(ChainList):
	def __init__(self, z_dim, n_layers):
		super(StyleEncoder, self).__init__()
		for i in range(n_layers):
			self.add_link(L.Linear(z_dim, z_dim))
		self.z_dim = z_dim

	def make_latent(self, size):
		xp = self.xp
		z = xp.random.normal(size=(size, self.z_dim)).astype(np.float32)
		z /= xp.sqrt(xp.sum(z * z, axis=1, keepdims=True) / self.z_dim + 1e-8)		
		return z

	def __call__(self, z):
		h = z
		for link in self.children():
			h = F.leaky_relu(link(h))
		return h
			
class Generator(Chain):
	def __init__(self, depth, z_dim):
		super(Generator, self).__init__()
		with self.init_scope():
			self.b0 = GFirstBlock(512, 512, z_dim, (4, 4))
			self.b1 = GBlock(512, 512, z_dim, (8, 8))
			self.b2 = GBlock(512, 256, z_dim, (16, 16))
			self.b3 = GBlock(256, 128, z_dim, (32, 32))
			self.b4 = GBlock(128, 64, z_dim, (64, 64))
			self.b5 = GBlock(64, 32, z_dim, (128, 128))
#			self.b6 = GBlock(32, 16, z_dim, (256, 256))

		self.depth = depth

	def __call__(self, w, alpha=1.0):
		if self.depth > 0 and alpha < 1.0:
			h = self.b0(w)
			for i in range(1, self.depth):
				h = self['b%d' % i](h, w)

			h1 = F.resize_images(h, (h.shape[2] * 2, h.shape[3] * 2))
			h2 = self['b%d' % (self.depth - 1)].toRGB(h1)
			h3 = self['b%d' % self.depth](h, w, True)
			
			h = h2 * (1 - alpha) + h3 * alpha

		elif self.depth > 0:
			h = self.b0(w)
			for i in range(1, self.depth):
				h = self['b%d' % i](h, w)
				
			h = self['b%d' % self.depth](h, w, True)
		else:
			h = self.b0(w, True)
					
		return h

	def style_mixing(self, ws):
		# --- assertion ---
		# alpha = 1.0
		# len(ws) = depth + 1
		# -----------------
		
		h = self.b0(ws[0])
		for i in range(1, self.depth):
			h = self['b%d' % i](h, ws[i])
			
		h = self['b%d' % self.depth](h, ws[self.depth], True)
		return h
		
class StyleBasedGenerator(Chain):
	def __init__(self, depth):
		z_dim = 512
		super(StyleBasedGenerator, self).__init__()
		with self.init_scope():
			self.G = Generator(depth, z_dim)
			self.E = StyleEncoder(z_dim, 4)

	def make_latent(self, size):
		return self.E.make_latent(size)

	def __call__(self, z, alpha):
		w = self.E(z)
		return self.G(w, alpha)

class DBlock(Chain):
	def __init__(self, ch_in, ch_out):
		super(DBlock, self).__init__()
		with self.init_scope():
			self.fromRGB = ConvLayer(3, ch_in, ksize=1, pad=0)
			self.c1 = ConvLayer(ch_in, ch_in)
			self.c2 = ConvLayer(ch_in, ch_out)

	def __call__(self, x, first=False):
		if first:
			l0 = F.leaky_relu(self.fromRGB(x))
		else:
			l0 = x
		l1 = F.leaky_relu(self.c1(l0))
		l2 = F.leaky_relu(self.c2(l1))
		l3 = F.average_pooling_2d(l2, 2, 2)
		return l3

class MinibatchStddev(Link):
	def __call__(self, x):
		m = F.broadcast_to(F.mean(x, axis=0, keepdims=True), x.shape)
		v = F.mean((x - m) * (x - m), axis=0, keepdims=True)
		std = F.mean(F.sqrt(v + 1e-8), keepdims=True)
		new_channel = F.broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
		return F.concat((x, new_channel), axis=1)

class DLastBlock(Chain):
	def __init__(self, ch_in, ch_out):
		super(DLastBlock, self).__init__()
		with self.init_scope():
			self.fromRGB = ConvLayer(3, ch_in, ksize=1, pad=0)
			self.stddev = MinibatchStddev()
			self.c1 = ConvLayer(ch_in + 1, ch_out)
			self.c2 = ConvLayer(ch_out, ch_out, 4, 1, 0)

	def __call__(self, x, first=False):
		if first:
			l0 = F.leaky_relu(self.fromRGB(x))
		else:
			l0 = x
		l1 = self.stddev(l0)
		l2 = F.leaky_relu(self.c1(l1))
		l3 = F.leaky_relu(self.c2(l2))
		return l3
		
class Discriminator(Chain):
	def __init__(self, depth):
		w = init.Normal(1.0)
		super(Discriminator, self).__init__()
		with self.init_scope():
#			self.b1 = DBlock(16, 32)
			self.b1 = DBlock(32, 64)
			self.b2 = DBlock(64, 128)
			self.b3 = DBlock(128, 256)
			self.b4 = DBlock(256, 512)
			self.b5 = DBlock(512, 512)
			self.b6 = DLastBlock(512, 512)
			self.l = L.Linear(512, 1, initialW=w)

		self.depth = depth

	def __call__(self, x, alpha=1.0):
		if self.depth > 0 and alpha < 1:
			h1 = self['b%d' % (6 - self.depth)](x, True)
			x2 = F.average_pooling_2d(x, 2, 2)
			h2 = F.leaky_relu(self['b%d' % (7 - self.depth)].fromRGB(x2))
			h = h2 * (1 - alpha) + h1 * alpha
		else:
			h = self['b%d' % (6 - self.depth)](x, True)
				
		for i in range(self.depth):
			h = self['b%d' % (7 - self.depth + i)](h)

		h = self.l(h)
		h = F.flatten(h)
		return h
