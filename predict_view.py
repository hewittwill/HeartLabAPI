import os
import cv2

import numpy as np

from util import Tile

from keras.models import Model, load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

class PredictView():

    viewmodel = Model

    MODEL_NAME = 'predict_viewclass_echocv.h5'

    def __init__(self):

        print('INFO: Loading viewclass model')

        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
                self.viewmodel = load_model( os.path.join('..', 'models', self.MODEL_NAME) , custom_objects={"Tile":Tile})

        print('INFO: Viewclass model loaded')

    def classify_view(self, frames, num_frames):

        processed_frames = []

        for frame in frames:
            processed_frames.append(np.array(cv2.cvtColor(cv2.resize( frame, (224, 224) ), cv2.COLOR_RGB2GRAY).reshape(224,224, 1)))

        processed_frames = np.array(processed_frames)

        result = self.viewmodel.predict(x=processed_frames, batch_size=num_frames)

        return result[0]
