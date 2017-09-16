import math
from math import exp
from random import random
import numpy as np

from random import randrange

import os
import sys

learning_rate = 0.001

supported_chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m'\
,'n','o','p','q','r','x','y','z','.',' ','!','-', '1', '2', '3', '4', '5'\
'6','7','8','9', '[', ']', ',', ';', ':']

def tanh(x):
	return np.tanh(x)

def hipster_softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T

def dtanh(x):
	return 1.0 - x**2

def sigmoid(X):
	return 1. / (1. + np.exp(-X))

def dsigmoid(x):
	return (1.0 - x) * x

def tanh_backward(dout, cache):
	dX = (1 - cache**2) * dout
	return dX

def sigmoid_backward(dout, cache):
	return cache * (1. - cache) * dout

def exp_running_avg(running, new, gamma=.9):
	return gamma * running + (1. - gamma) * new

def fc_backward(dout, cache):
	W, h = cache

	dW = np.dot(h.T, dout)
	db = np.sum(dout, axis=0)
	dX = np.dot(dout, W.T)

	return dX, dW, db

def fc_forward(W, x, bf):
	return np.dot(x,W) + bf

def softmax(y):
	y = np.exp(y)
	s = np.sum(y)
	y /= s
	return y

def cross_entropy_loss(y,y_correct):
	return ( - math.log( y[0,y_correct]) )

class lstm:

	def __init__(self, N, F, name="LSTM_NeuralNet.txt", alpha=1e-3, momentum=0.95, decay=1e-3):
		# N = number of neurons
		# F = number of features (letters shown up in the text, D > H )

		S = N + F # The sum of the two

		# Note: In order for BLAS to kick in when computing 
		# matrix-vector multiplications, it is necessary that
		# the computation Ax = y , where x is column vector
		# is instead computed as xT AT = yT, for some reason
		# it i MUCH faster this way on my machine.

		self.model = dict(
			Wf=np.random.randn(S,N) / np.sqrt(S / 2),
			Wi=np.random.randn(S,N) / np.sqrt(S / 2),
			Wc=np.random.randn(S,N) / np.sqrt(S / 2),
			Wo=np.random.randn(S,N) / np.sqrt(S / 2),
			Wy=np.random.randn(N,F) / np.sqrt(N / 2),
			bf=np.zeros((1,N)),
			bi=np.zeros((1,N)),
			bc=np.zeros((1,N)),
			bo=np.zeros((1,N)),
			by=np.zeros((1,F))
		)

		self.N = N
		self.F = F
		self.S = S

		self.name = name

		# parameters for gradient descent with momentum update
		self.momentum = momentum
		self.decay = decay
		self.alpha = alpha

		self.M = {k: np.zeros_like(v) for k, v in self.model.items()}
		self.R = {k: np.zeros_like(v) for k, v in self.model.items()}
		self.beta1 = .9
		self.beta2 = .999

	def get_initial_state(self):
		return ( np.zeros_like(self.model["bi"]), np.zeros_like(self.model["bi"]) )

	def initial_gradients(self):
		F = self.F
		N = self.N
		S = self.S
		return dict(
			Wf=np.zeros((S,N)),
			Wi=np.zeros((S,N)),
			Wc=np.zeros((S,N)),
			Wo=np.zeros((S,N)),
			Wy=np.zeros((N,F)),
			bf=np.zeros((1,N)),
			bi=np.zeros((1,N)),
			bc=np.zeros((1,N)),
			bo=np.zeros((1,N)),
			by=np.zeros((1,F))
		)

	def get_model(self):
		return self.model

	def read_net_from_ascii(self, name_to_read):
		m = self.model
		name = name_to_read
		for k in m:
			with open(name + "/" + k, "rb") as f:
				m[k] = np.load(f)
		self.name = name

	def store_net_in_ascii(self):
		try:
			os.makedirs(self.name)
		except OSError as e:
			print("error")
		m = self.model
		for k in m:
			with open(self.name + "/" + k, "wb") as f:
				np.save(f, m[k])

	def forward_propagate(self, X, state):
		# X should be an index value in the interval 0 <= X < D

		m = self.model
		N = self.N
		F = self.F

		Wf, Wi, Wc, Wo, Wy = m['Wf'], m['Wi'], m['Wc'], m['Wo'], m['Wy']
		bf, bi, bc, bo, by = m['bf'], m['bi'], m['bc'], m['bo'], m['by']

		h_old, c_old = state

#		h_old = np.array([h_old])
		X_one_hot = np.zeros(F)
		X_one_hot[X] = 1.
		X_one_hot = X_one_hot.reshape(1, -1)
		
#		print(h_old)
#		print(X_one_hot)
		X = np.column_stack((h_old, X_one_hot))

		hf = np.dot(X, Wf) + bf

		hi = np.dot(X, Wi) + bi
		ho = np.dot(X, Wo) + bo
		hc = np.dot(X, Wc) + bc

		hf = sigmoid( hf )
		hi = sigmoid( hi )	
		ho = sigmoid( ho )	
		hc = tanh( hc )	

		c = hf * c_old + hi * hc

		tanh_c_cache = tanh(c)

		h = ho * tanh_c_cache

		y = np.dot(h, Wy) + by

		prob = softmax(y)

		state = (h, c)

		cache = (X, h_old, c_old, hf, hi, ho, hc, tanh_c_cache )

		return prob, state, cache

	def backward_propagate(self, y_prob, y_train, state, d_next, cache):
			
		h, c = state 
		dldh_next, dldc_next = d_next

		X, h_old, c_old, hf, hi, ho, hc, tanh_c_cache = cache
		m = self.model

		dldh = y_prob
		dldh[0,y_train] -= 1.

		dh, dWy, dby = fc_backward(dldh, (m["Wy"], h) )
#		print("dh: " + str(np.shape(dh)) + " dldh_next: " + str(np.shape(dldh_next)))
		dh += dldh_next

		dldho = dh * tanh_c_cache
		dldho = dsigmoid(ho) * dldho

#		print("dldh: " + str(np.shape(dldh)) + " ho: " + str(np.shape(ho)) + " cache_tmp: " + str(np.shape(cache_tmp)))
		dldc = dh * ho * dtanh(tanh_c_cache)
		dldc += dldc_next

		dldhf = dldc * c_old
		dldhf = dsigmoid(hf) * dldhf

		dldhi = hc * dldc
		dldhi = dsigmoid(hi) * dldhi

		dldhc = hi * dldc
		dldhc = dtanh(hc) * dldhc

		dXi, dWi, dbi = fc_backward( dldhi ,(m["Wi"], X) )
		dXo, dWo, dbo = fc_backward( dldho ,(m["Wo"], X) )
		dXf, dWf, dbf = fc_backward( dldhf ,(m["Wf"], X) )
		dXc, dWc, dbc = fc_backward( dldhc ,(m["Wc"], X) )

		dX = dXi + dXo + dXf + dXc

		dh_next = dX[:, :self.N]

		dc_next = hf * dldc

#		print(dh_next.flags)

		grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)

		return grad, (dh_next, dc_next)

	def gradient_descent(self, grads, t):
		m = self.model
		learn = self.alpha

		for k in grads:
			m[k] -= learn * grads[k]

	def gradient_update(self, grads, t):
		# grads must have the keys and values like those returned from backward_propagate
		self.gradient_descent(grads, t)

def get_minibatch(X, y, minibatch_size, shuffle=True):
	minibatches = []

	for i in range(0, X.shape[0], minibatch_size):
		X_mini = X[i:i + minibatch_size]
		y_mini = y[i:i + minibatch_size]

		minibatches.append((X_mini, y_mini))

	return minibatches


def train_lstm(filename, netname="LSTM_NeuralNet.txt", read_net=False, filename_to_read="LSTM_NeuralNet.txt", iterations=10000000, store_every=50):

	txt = ""
	with open(filename, 'r') as f:
		txt = f.read()
		f.close()

	X = []
	y = []

	char_to_idx = {char: i for i, char in enumerate(set(txt))} # Map for char to index
	idx_to_char = {i: char for i, char in enumerate(set(txt))} # Map for index to char

	X = np.array([char_to_idx[x] for x in txt])
	y = [char_to_idx[x] for x in txt[1:]]
	y.append(char_to_idx['.'])
	y = np.array(y)

	D = 128 # the number of neurons to be used
	H = len(char_to_idx) # number of features to classify (chars appearing)

	l = lstm(D,H,netname)
	if read_net:
		l.read_net_from_ascii(filename_to_read)

	state = l.get_initial_state()

	idx_list = list(range(l.F))

	batches = get_minibatch(X,y,10)

	# Training on the mini batches!

	best_loss = 100.0
	indx = 0
	for i in range(1, iterations+1):
		probs = []
		caches = []
		states = []
		loss = 0.
 
		(x_mini, y_mini) = batches[indx]

		indx += 1
		if ( indx >= len(batches)):
			indx = 0
#		print(x_mini)
		# Forward
		n = -1
		for x, y_true in zip(x_mini, y_mini):
			prob, state, cache = l.forward_propagate(x, state)
			loss += cross_entropy_loss(prob, y_true)

			states.append(state)
			probs.append(prob)
			caches.append(cache)
			n += 1

		loss /= len(x_mini)

		if ( loss < best_loss):
			best_loss = loss

		dh_next = np.zeros((1, l.N))
		dc_next = np.zeros((1, l.N))
		d_next = (dh_next, dc_next)

		grds = {k: np.zeros_like(v) for k, v in l.model.items()}

		# Backward
		for x, y_true in reversed(list(zip(x_mini, y_mini))):
			cache = caches[n]
			prob = probs[n]
			state = states[n]

			grads, d_next = l.backward_propagate(prob, y_true, state, d_next, cache)

			for k in grads.keys():
	#			print(grads[k])
	#			print(grds[k])
				grds[k] += grads[k]
			n -= 1

		for k, v in grds.items():
			grds[k] = np.clip(v, -5,5)

		# Gradient decend
#		print(grds)
		l.gradient_update(grds, i)

		test_state = l.get_initial_state() # starting state

		if ( i % 200 == 0 ):
			print("Iteration: " + str(i) + ". Loss: " + str(loss) + ", best loss: " + str(best_loss))
			print("===================")
			test_input = x_mini[0]
			st = ""

			for q in range(0,50):
				pr, test_state, ch = l.forward_propagate(test_input, test_state)
		#		print(pr)
		#		print(idx_list)
				test_input = np.random.choice(idx_list,p=pr.ravel())
				st += idx_to_char[test_input]
				#print(pr) <- changes over time
			print(st)
			print("===================")

if __name__=="__main__":

	stop_count = 40

	iterations = 100

	if len(sys.argv) < 3:
		print("I need at least 2 argument as input: python(what ever version) the_program filename netname [name of neural net to read from]]")

	filename = sys.argv[1]
	netname = sys.argv[2]

	if len(sys.argv) > 3:
		# Read from file
		to_read_name = sys.argv[3]

		train_lstm(filename, netname, True, to_read_name)	
	
	else:
	
		train_lstm(filename, netname)	
