"""

Requires moabb > 0.5.0
Requires the still unmerged version of EEGConformer pip install git+https://github.com/braindecode/braindecode.git@refs/pull/454/head
EEGConformer is loaded through braindecode.

Resampler_Epoch is available only after 0.5.0 version of MOABB

Force MOABB version from git: pip install git+https://github.com/NeuroTechX/moabb.git#egg=moabb 

"""

from moabb.datasets.bnci import BNCI2014001
from moabb.datasets.braininvaders import VirtualReality, bi2012
from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP

#from moabb.datasets.compound_dataset.base import CompoundDataset
#from moabb.datasets.utils import blocks_reps

from moabb.evaluations.evaluations import CrossSubjectEvaluation
from moabb.paradigms.p300 import P300
import torch
import seaborn as sns
import pandas as pd

from braindecode import EEGClassifier

#from braindecode.models import EEGNetv4
from braindecode.models import ( EEGNetv4, EEGConformer)


from sklearn.pipeline import Pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from moabb.pipelines.features import Resampler_Epoch


from moabb.pipelines.utils_pytorch import BraindecodeDatasetLoader, InputShapeSetterEEG
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt

# Set up GPU if it is there
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"

# Hyperparameter
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0
BATCH_SIZE = 64
SEED = 42
VERBOSE = 1
EPOCH = 10
PATIENCE = 3

# Create the dataset
create_dataset = BraindecodeDatasetLoader()

#name, electrodes, subjects
#bi2013a	    16	24 (normal)
#bi2014a    	16	64 (usually low performance)
#BNCI2014009	16	10 (usually high performance)
#BNCI2014008	 8	 8
#BNCI2015003	 8	10
#bi2015a        32  43
#bi2015b        32  44
datasets = [bi2013a()] #BNCI2014009()

# Set random Model
model = EEGNetv4(in_chans=16, n_classes=2, input_window_samples=100)
#model = EEGConformer(n_classes=2, n_channels=16)

# Define a Skorch classifier
clf = EEGClassifier(
    module=model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=EPOCH,
    train_split=ValidSplit(0.2, random_state=SEED),
    device=device,
    callbacks=[
        EarlyStopping(monitor="valid_loss", patience=PATIENCE),
        EpochScoring(
            scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
        ),
        EpochScoring(
            scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
        ),
        InputShapeSetterEEG(
            params_list=["in_chans", "input_window_samples", "n_classes"],
        ),
    ],
    verbose=VERBOSE,  # Not printing the results for each epoch
)

# Create Pipelines

pipelines_withEpochs = {}
pipelines_withArray  = {}

pipelines_withEpochs["braindecode_" + model.__class__.__name__] = Pipeline(
    [
        ("resample", Resampler_Epoch(128)),
        ("braindecode_dataset", create_dataset),
        (model.__class__.__name__ + "_clf", clf),
    ]
)

pipelines_withArray["MDM"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(),
    MDM()
)

# Run evaluation

paradigm = P300()
# paradigm.resample = 128

# class CustomDataset1(CompoundDataset):
#     def __init__(self):
#         biVR = VirtualReality(virtual_reality=True, screen_display=True)
#         runs = blocks_reps([1, 3], [1, 2, 3, 4, 5])
#         subjects_list = [
#             (biVR, 1, "VR", runs),
#             (biVR, 2, "VR", runs),
#         ]
#         CompoundDataset.__init__(
#             self,
#             subjects_list=subjects_list,
#             events=dict(Target=2, NonTarget=1),
#             code="D1",
#             interval=[0, 1.0],
#             paradigm="p300",
#         )

# redeuce to two subjects for testing
n_subjects = 2
for dataset in datasets:
    dataset.subject_list = dataset.subject_list[0:n_subjects]

evaluation1 = CrossSubjectEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
    return_epochs=True
)

results1 = evaluation1.process(pipelines_withEpochs)

evaluation2 = CrossSubjectEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
    return_epochs=False
)

results2 = evaluation2.process(pipelines_withArray)

results = pd.concat([results1, results2])
print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])



# Here we plot the results to compare two pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])
title = "TODO"
sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1").set(
    title=title
)

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()