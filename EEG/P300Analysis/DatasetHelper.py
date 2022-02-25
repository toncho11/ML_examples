def GetCzChannel(dataset):
    if (dataset == "BNCI2014008"): #8 electrodes
        return 1
    else:
        print("Error: Could not set Cz for this dataset!")
        
def GetElectrodeCount(dataset):
    if (dataset == "BNCI2014008"): #8 electrodes
        return len(GetChannelNames(dataset))
    else:
        print("Error: Could not get the electrode count!")

def GetChannelNames(dataset):
    if (dataset == "BNCI2014008"): #8 electrodes
        return ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7","PO8"]
    else:
        print("Error: Could not get the channel names!")
        
def GetSubjectsCount(dataset):
    if (dataset == "BNCI2014008"): #8 electrodes
        return 8
    else:
        print("Error: Could not get subjects count!")

#if "Cz" is provided then it will return an integer for this electrode
def ElectrodeByName(dataset, electrode_name):   
    i = 0
    for e in GetChannelNames(dataset):
        if e.lower() == electrode_name.lower():
            return i
        else:
            i = i + 1