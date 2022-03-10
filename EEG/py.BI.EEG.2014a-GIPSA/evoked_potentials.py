
import mne

from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances, ERPCovariances
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import joblib

import sys
sys.path.append('.')
from braininvaders2014a.dataset import BrainInvaders2014a

filename = 'classification_scores.pkl'
scores = joblib.load(filename)
dataset = BrainInvaders2014a()

# loop on list of subjects
for subject in dataset.subject_list:

	print('treating subject', str(subject).zfill(2))

	# get the raw object
	sessions = dataset._get_single_subject_data(subject)
	raw = sessions['session_1']['run_1']
	chname2idx = {}
	for i, chn in enumerate(raw.ch_names):
		chname2idx[chn] = i

	# filter data and resample
	fmin = 1
	fmax = 24
	raw.filter(fmin, fmax, verbose=False)

	# detect the events and cut the signal into epochs
	events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
	event_id = {'NonTarget': 1, 'Target': 2}
	epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.8, baseline=None, verbose=False, preload=True)
	epochs.pick_types(eeg=True)

	fig, ax = plt.subplots(facecolor='white', figsize=(10.9,  7.6))
	evkTarget = epochs['Target'].average().data[chname2idx['Cz'],:]
	evkNonTarget = epochs['NonTarget'].average().data[chname2idx['Cz'],:]
	t = np.arange(len(evkTarget)) / epochs.info['sfreq']
	ax.plot(t, evkTarget, c='#2166ac', lw=3.0, label='Target (' + str(len(epochs['Target'])) + ' trials)')
	ax.plot(t, evkNonTarget, c='#b2182b', lw=3.0, label='NonTarget (' + str(len(epochs['NonTarget'])) + ' trials)')
	ax.plot([0, 0.8], [0, 0], c='#CDCDCD', lw=2.0, ls='--')	
	ax.set_xlim(0, 0.8)
	ax.set_title('Average evoked potentials at electrode Cz for subject ' + str(subject) + ' (AUC : ' + '{:.2f}'.format(scores[subject]) + ')', fontsize=16)
	ax.set_ylabel(r'amplitude ($\mu$V)', fontsize=12)
	ax.set_xlabel('time after stimulus (s)', fontsize=12)
	ax.legend()

	filename = './evoked_potentials/evoked_potentials_subject_' + str(subject).zfill(2) + '.pdf'
	fig.savefig(filename, format='pdf')

