from __future__ import division
from __future__ import print_function
import crnn
from PIL import Image
import string
import numpy as np
from scipy.special import softmax
import torch 
from torch.autograd import Variable
import torchvision.transforms as transforms

import pickle
with open('weights/prior.pkl', 'rb') as f:
    prior = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ctcBestPath(mat, classes):
    "implements best path decoding as shown by Graves (Dissertation, p63)"

    # dim0=t, dim1=c
    maxT, maxC = mat.shape
    label = ''
    blankIdx = len(classes)
    lastMaxIdx = maxC # init with invalid label

    for t in range(maxT):
        maxIdx = np.argmax(mat[t, :])

        if maxIdx != lastMaxIdx and maxIdx != blankIdx:
            label += classes[maxIdx]

        lastMaxIdx = maxIdx

    return label

def ctcBeamSearch(mat, classes, lm, beamWidth=25):
    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."

    blankIdx = len(classes)
    maxT, maxC = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    # go over all time-steps
    for t in range(maxT):
        curr = BeamState()

        # get beam-labelings of best beams
        bestLabelings = last.sort()[0:beamWidth]

        # go over best beams
        for labeling in bestLabelings:

            # probability of paths ending with a non-blank
            prNonBlank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

            # add beam at current time-step if needed
            addBeam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
            curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

            # extend current beam-labeling
            for c in range(maxC - 1):
                # add new char to current beam-labeling
                newLabeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[labeling].prTotal

                # add beam at current time-step if needed
                addBeam(curr, newLabeling)
                
                # fill in data
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank
                
                # apply LM
                applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

        # set new beam state
        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

     # sort by probability
    bestLabeling = last.sort()[0] # get most probable labeling

    # map labels to chars
    res = ''
    for l in bestLabeling:
        res += classes[l]

    return res

def crnn_predict(crnn, img, transformer, decoder='bestPath', normalise=False):
    """
    Params
    ------
    crnn: torch.nn
        Neural network architecture
    transformer: torchvision.transform
        Image transformer
    decoder: string, 'bestPath' or 'beamSearch'
        CTC decoder method.
    
    Returns
    ------
    out: a list of tuples (predicted alphanumeric sequence, confidence level)
    """
    
    classes = string.ascii_uppercase + string.digits
    image = img.copy()
    
    image = transformer(image).to(device)
    image = image.view(1, *image.size())
    
    # forward pass (convert to numpy array)
    preds_np = crnn(image).data.cpu().numpy().squeeze()
    
    # move first column to last (so that we can use CTCDecoder as it is)
    preds_np = np.hstack([preds_np[:, 1:], preds_np[:, [0]]])
    
    preds_sm = softmax(preds_np, axis=1)
#     preds_sm = np.divide(preds_sm, prior)
    
    # normalise is only suitable for best path
    if normalise == True:
        preds_sm = np.divide(preds_sm, prior)
            
    if decoder == 'bestPath':
        output = ctcBestPath(preds_sm, classes)
        
    elif decoder == 'beamSearch':
        output = ctcBeamSearch(preds_sm, classes, None)
    else:
        raise Exception("Invalid decoder method. \
                        Choose either 'bestPath' or 'beamSearch'")
        
    return output

class AutoLPR:
    
    def __init__(self, decoder='bestPath', normalise=False):
        
        # crnn parameters
        self.IMGH = 32
        self.nc = 1 
        alphabet = string.ascii_uppercase + string.digits
        self.nclass = len(alphabet) + 1
        self.transformer = transforms.Compose([
            transforms.Grayscale(),  
            transforms.Resize(self.IMGH),
            transforms.ToTensor()])
        self.decoder = decoder
        self.normalise = normalise
        
                
    def load(self, crnn_path):

        # load CRNN
        self.crnn = crnn.CRNN(self.IMGH, self.nc, self.nclass, nh=256).to(device)
        self.crnn.load_state_dict(torch.load(crnn_path, map_location=device))
            
        # remember to set to test mode (otherwise some layers might behave differently)
        self.crnn.eval()
        
    def predict(self, img_path):
        
        # image processing for crnn
        self.image = Image.open(img_path)
        return crnn_predict(self.crnn, self.image, self.transformer, self.decoder, self.normalise)
