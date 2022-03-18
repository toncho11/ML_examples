
from scipy.linalg import sqrtm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pyriemann.utils.distance import distance
from pyriemann.spatialfilters import Xdawn, CSP, BilinearFilter, SimplifiedSTCP
from pyriemann.classification import MDM, QuanticSVM, QuanticVQC
from pyriemann.estimation import ERPCovariances, Covariances, XdawnCovariances
from braininvaders2012.dataset import BrainInvaders2012
import numpy as np
import mne
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
from sklearn.svm import SVC

from qiskit import BasicAer, IBMQ
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.algorithms import SklearnSVM
from qiskit.ml.datasets import ad_hoc_data, sample_ad_hoc_data
from qiskit.providers.ibmq import least_busy
seed = 10599
aqua_globals.random_seed = seed

from pyriemann.hcbr import HCBRClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from pyriemann.tangentspace import TangentSpace
from sklearn.decomposition import PCA

"""
=============================
Classification of the trials
=============================

This example shows how to extract the epochs from the dataset of a given
subject and then classify them using Machine Learning techniques using
Riemannian Geometry. 

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

import warnings
warnings.filterwarnings("ignore")

def compressed_sensing(X, y, clf, tolerance_percent = 0.15, step=0.4):
  print("Initializing compression...")
  # get score of non-compressed signal
  skf = StratifiedKFold(n_splits=5)
  initial_score = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc').mean()
  print("Score without compression:", initial_score)
  #init convergence
  score = 0
  sample_size = 0
  ri = 0
  n = len(X[0][0])
  while score < initial_score * (1 - tolerance_percent):
    sample_size = sample_size + step
    print("sample size:", sample_size)
    # extract samples of signal
    m = int(n * sample_size)
    ri = np.random.choice(n, m, replace=False) # random sample of indices
    X2 = []
    Xtemp = []
    for epoch in X:
      epoch2 = []
      epochTemp = []
      vx = None
      for sensor in epoch:
        
        sensor2 = sensor[ri]
        # create idct matrix operator
        A = spfft.idct(np.identity(n), norm='ortho', axis=0)
        A = A[ri]
        # do L1 optimization
        vx = cvx.Variable(n)
        objective = cvx.Minimize(cvx.norm(vx, 1))
        constraints = [A*vx == sensor2]
        prob = cvx.Problem(objective, constraints)
        prob.solve(verbose=False)
        epoch2.append(sensor2)
        epochTemp.append(vx.value)
      
      X2.append(epoch2)
      Xtemp.append(epochTemp)

    score = cross_val_score(clf, np.array(np.array(Xtemp)), y, cv=skf, scoring='roc_auc').mean()
    print("Sample score:", score)
  n2 = len(X2[0][0])
  print("Compression:", n2, "/", n, "=", (n - n2)/n*100)
  return np.array(np.array(X2))
  
def serialize(cov):
	ret = ''
	dim = len(cov)
	print("dim ", dim)
	dim2 = len(cov[0])
	for i in range(0, dim):
		for j in range(0, dim2):
			# ret = ret + ' {:.12f}'.format(cov[i][j])
			ret = ret + ' ' + str(cov[i][j])
	return ret[1:-1]

def serializes(covs):
	ret = ''
	print(len(covs))
	for cov in covs:
		ret = ret + serialize(cov) + '\n'
	return ret

# define the dataset instance
dataset = BrainInvaders2012(Training=True)

scr = {}
# get the data from subject of interest
# for subject in dataset.subject_list:
for subject in range(1, 2):

	data = dataset._get_single_subject_data(subject)
	raw = data['session_1']['run_training']

	# filter data and resample
	fmin = 1
	fmax = 24
	raw.filter(fmin, fmax, verbose=False)
	# raw.resample(64)

	# detect the events and cut the signal into epochs
	events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
	event_id = {'NonTarget': 1, 'Target': 2}
	epochs = mne.Epochs(raw, events, event_id, tmin=0.1, tmax=0.7, baseline=None, verbose=False, preload=True)
	epochs.pick_types(eeg=True)

	# get trials and labels
	X = epochs.get_data()
	y = events[:, -1]
	y = LabelEncoder().fit_transform(y)
  

	# covariance matrices
	sf = XdawnCovariances(nfilter=1, estimator="corr", xdawn_estimator="cov")
	sf2 = SimplifiedSTCP(target = 1, nbComponents=8)
	sf3 = Xdawn(nfilter=1, estimator='cov')
	erpc = ERPCovariances(classes=[1], estimator='corr')
	qsvm = QuanticSVM(target=1, processVector=lambda v:v, verbose=True, qAccountToken="b379c7b98b59c7096891400f930727a20a823e54ee99def836375a64c2290f5ed571a11d996eb38e4dc9bff97ebceddda2e56fb1d2e6fe94c358fca035d3dc4e")
	pca = PCA(n_components = 8)
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	# clf = make_pipeline(XdawnCovariances(nfilter=1, estimator='corr', xdawn_estimator='cov'))
	clf = make_pipeline(sf, TangentSpace())
	clf.fit(X, y)
	Xp = clf.transform(X)
	def tm (N):
		tot = N*N
		ret = []
		for n in range(tot):
			i = int(n / N)
			j = n % N
			if( not i == j and (i >= N/2 or j >= N/2)):
				ret.append(n)
		return ret
	
	


	def processV(v, nbChannels, nbSamples):
		
		return (v*10**4).astype(int) #[tm(nbChannels)]
		#[[2, 3, 6, 7, 8, 9, 11, 12, 13, 14]]
		x = np.reshape(v, (nbChannels, nbSamples))
		# return pca.fit_transform((x*10**4).astype(int))
		I = np.identity(nbChannels)
		dist = round(distance(x, I, "riemann")*10)
		# dist = np.round(np.trace(x), 4)
		return [dist]
	hcbr = HCBRClassifier("C:\\Users\\GregoireCattan\\Documents\\py.BI.EEG.2012-GIPSA\\params.json", 1, Xp, y, processV)
	svc = SVC(kernel='linear')
	
	clf.steps.append(("qsvm", qsvm))

	# X[y == 0] = np.clip(X[y == 0], 0.5, 0.5)
	# X[y == 1] = np.clip(X[y == 1], 1, 1)
	# hcbr.X = X

  # cross validation
	# clf.fit(X, y)
	# exit()


	
	skf = StratifiedKFold(n_splits=5)

	def scorer(clf, X_test, y_test):
		y_pred = clf.predict(X_test)
		# y_pred = clf.predict(X_test)
		ba = balanced_accuracy_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred,)
		return {'ba': ba, 'f1': f1}

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	ba = balanced_accuracy_score(y_test, y_pred)
	print("subject", subject, "ba = ", ba)
	exit()

	cv_results = cross_validate(clf, X, y, cv=skf, scoring=scorer, verbose=True, error_score="raise")
	scr[subject] = { 'ba': cv_results['test_ba'].mean(),
	'f1': cv_results['test_f1'].mean(),
	'fit_time': cv_results['fit_time'].mean(),
	'score_time': cv_results['score_time'].mean()
  }
	# exit()

  

	# with open('test.txt', 'w') as the_file:
	# 	the_file.write(serializes(covs))
	# y_string = ''
	# for i in y:
	# 	y_string = y_string + str(i) + '\n'
	# with open('test_y.txt', 'w') as the_file:
	# 	the_file.write(y_string)
	# exit()

	# print results of classification
	print('subject', subject)
	print('mean AUC :', scr[subject])

	#####

print(scr)
filename = './hcbr_xdawn.pkl'
joblib.dump(scr, filename)	




