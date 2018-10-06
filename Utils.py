from scipy import linalg
import numpy as np


def SVD(bow_with_kernal,dimension):
	u, s, vh = linalg.svd(bow_with_kernal, full_matrices=False)
	#print(u.shape)
	#print(vh.shape)
	sigma = s.tolist()
	#print(sigma)

	while (len(sigma)!=dimension):
		a = min(sigma)
		ind = sigma.index(a)
		u = np.delete(u,ind,1)
		#print(u.shape)
		vh = np.delete(vh,ind,0)
		#print(vh.shape)
		sigma.remove(a)
	sigma = np.asarray(sigma)
	#print(sigma)

	s = np.diag(sigma)
	#print(s.shape)

	A = np.dot(u,np.dot(s,vh))
	return A