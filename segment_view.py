import os
import cv2

import numpy as np

from keras.models import Model, load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

class SegmentView():

    model = Model

    PLAX_NAME = 'segment_plax.h5'
    PSAX_NAME = 'segment_psax.h5'
    A4C_NAME = 'segment_a4c.h5'
    A3C_NAME = 'segment_a3c.h5'
    A2C_NAME = 'segment_a2c.h5'

    MODELS = {'plax': PLAX_NAME, 'psax': PSAX_NAME, 'a4c': A4C_NAME, 'a3c': A3C_NAME, 'a2c': A2C_NAME}

    def __init__(self, view):

        print('INFO: Loading segmentation model')

        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
                self.model = load_model( os.path.join('..','models', self.MODELS[view]))

        print('INFO: Segmentation model loaded')

    def segment_view(self, frames, num_frames):

        processed_frames = []
        result = None

        for frame in frames:
            processed_frames.append(np.array([cv2.cvtColor(cv2.resize(frame, (384,384)), cv2.COLOR_RGB2GRAY)]).reshape((384,384,1)))

        processed_frames = np.array(processed_frames)

        result = self.model.predict(x=processed_frames, batch_size=num_frames)

        result = np.argmax(result, axis=-1)

        return result
