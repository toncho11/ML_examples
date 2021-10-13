#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import mne
import numpy as np
from . import download as dl
import os
import glob
import zipfile
import yaml
from scipy.io import loadmat
from distutils.dir_util import copy_tree
import shutil
import pandas as pd

BI2014a_URL = 'https://zenodo.org/record/3266223/files/'


class BrainInvaders2014a():
    '''
    This dataset contains electroencephalographic (EEG) recordings of 71 subjects 
    playing to a visual P300 Brain-Computer Interface (BCI) videogame named Brain Invaders. 
    The interface uses the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target) 
    that are flashed pseudo-randomly to elicit the P300 response. EEG data were recorded 
    using 16 active dry electrodes with up to three game sessions. The experiment took place 
    at GIPSA-lab, Grenoble, France, in 2014. A full description of the experiment is available 
    at https://hal.archives-ouvertes.fr/hal-02171575. Python code for manipulating the data 
    is available at https://github.com/plcrodrigues/py.BI.EEG.2014a-GIPSA. The ID of this 
    dataset is bi2014a.
    '''

    def __init__(self):

        self.subject_list = list(range(1, 65))

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)

        sessions = {}
        session_name = 'session_1'
        sessions[session_name] = {}
        run_name = 'run_1'

        chnames = ['Fp1',
                   'Fp2',
                   'F3',
                   'AFz',
                   'F4',
                   'T7',
                   'Cz',
                   'T8',
                   'P7',
                   'P3',
                   'Pz',
                   'P4',
                   'P8',
                   'O1',
                   'Oz',
                   'O2',
                   'STI 014']
        chtypes = ['eeg'] * 16 + ['stim']

        file_path = file_path_list[0]
        D = loadmat(file_path)['samples'].T

        S = D[1:17, :]
        stim = D[-1, :]
        X = np.concatenate([S, stim[None, :]])

        info = mne.create_info(ch_names=chnames, sfreq=512,
                               ch_types=chtypes,
                               verbose=False)
        raw = mne.io.RawArray(data=X, info=info, verbose=False)

        sessions[session_name][run_name] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = BI2014a_URL + 'subject_' + str(subject).zfill(2) + '.zip'
        path_zip = dl.data_path(url, 'BRAININVADERS2014A')
        path_folder = path_zip.strip(
            'subject_' + str(subject).zfill(2) + '.zip')

        # check if has to unzip
        path_folder_subject = path_folder + \
            'subject_' + str(subject).zfill(2) + os.sep
        if not(os.path.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        subject_paths = []

        # filter the data regarding the experimental conditions
        subject_paths.append(path_folder_subject +
                             'subject_' + str(subject).zfill(2) + '.mat')

        return subject_paths
