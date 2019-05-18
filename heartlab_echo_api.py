import os
import cv2
import uuid
import skvideo.io
import numpy as np
import tensorflow as tf

from pydicom import dcmread
from pydicom.filebase import DicomBytesIO

from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from util import CLASS_NAMES, SEGMENT_VIEWS, PLAX_VIEWS, PSAX_VIEWS, A4C_VIEWS, A3C_VIEWS, A2C_VIEWS

from predict_view import PredictView
from segment_view import SegmentView

app = Flask(__name__)
CORS(app)

graph = tf.get_default_graph()
tf.Session()

view_model = PredictView()
plax_model = SegmentView('plax')
psax_model = SegmentView('psax')

a4c_model = SegmentView('a4c')
a3c_model = SegmentView('a3c')
a2c_model = SegmentView('a2c')

SEG_BATCH_SIZE = 32

@app.route('/segments/<id>')
def segments(id):
    return send_from_directory('segments', id + '.mp4')

@app.route('/videos/<id>')
def videos(id):
    return send_from_directory('videos', id + '.mp4')

@app.route('/upload_dcm', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(url_for('home'))
        #
        # file = request.files['file']
        #
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(url_for('home'))
        #
        # if file and file.filename.split('.')[1] == 'dcm':
        #
        #     print(file.read())

            # ds = dcmread( DicomBytesIO(file.read()) )
            # ds.decompress()
            #
            # print(ds)
        #
        #     frames = ds.pixel_array
        #
        #     id = uuid.uuid1()
        #
        #     skvideo.io.vwrite('videos/' + str(id) +  ".mp4", ds.pixel_array)
        #
        #     view = 'other'
        #     segmented = False
        #
        #     global graph
        #     with graph.as_default():
        #         result = view_model.classify_view(ds.pixel_array[:10], 10)
        #
        #         maxval = np.amax(result)
        #         maxidx = np.where(result == maxval)[0][0]
        #
        #         view = CLASS_NAMES[maxidx]
        #
        #         if view in SEGMENT_VIEWS:
        #
        #             segmented = True
        #
        #             frames = ds.pixel_array
        #             nframes = ds.NumberOfFrames
        #             nbatch = nframes // SEG_BATCH_SIZE
        #             rbatch = nframes % SEG_BATCH_SIZE
        #
        #             print('INFO: ' + str(nbatch) + ' batches to segment.')
        #
        #             results = np.zeros((nframes, 384, 384))
        #
        #             model = None
        #
        #             if view in A4C_VIEWS:
        #                 model = a4c_model
        #             if view in A3C_VIEWS:
        #                 model = a3c_model
        #             if view in A2C_VIEWS:
        #                 model = a2c_model
        #             if view in PLAX_VIEWS:
        #                 model = plax_model
        #             if view in PSAX_VIEWS:
        #                 model = psax_model
        #
        #             for i in range(0,nbatch):
        #                 print('INFO: Segmenting batch #' + str(i))
        #                 results[i*SEG_BATCH_SIZE:(i*SEG_BATCH_SIZE)+SEG_BATCH_SIZE] = model.segment_view(frames[i*SEG_BATCH_SIZE:(i*SEG_BATCH_SIZE)+SEG_BATCH_SIZE], SEG_BATCH_SIZE )
        #
        #             if rbatch != 0: results[(-1*rbatch):] = model.segment_view(frames[(-1*rbatch):], rbatch)
        #
        #             downsized_orig = np.array([cv2.resize(i, (384, 384)) for i in ds.pixel_array])
        #             results = results.reshape((results.shape[0],384,384,1))
        #
        #             overlay = np.concatenate((results, results, results), axis=-1)
        #
        #             if view in A4C_VIEWS:
        #
        #                 overlay[ np.where((overlay == [1, 1, 1]).all(axis=-1)) ] = [255, 0, 0]
        #                 overlay[ np.where((overlay == [2, 2, 2]).all(axis=-1)) ] = [0, 255, 0]
        #                 overlay[ np.where((overlay == [3, 3, 3]).all(axis=-1)) ] = [0, 0, 0]
        #                 overlay[ np.where((overlay == [4, 4, 4]).all(axis=-1)) ] = [0, 0, 255]
        #                 overlay[ np.where((overlay == [5, 5, 5]).all(axis=-1)) ] = [0, 0, 0]
        #
        #             if view in A3C_VIEWS:
        #
        #                 overlay[ np.where((overlay == [1, 1, 1]).all(axis=-1)) ] = [255, 0, 0]
        #                 overlay[ np.where((overlay == [2, 2, 2]).all(axis=-1)) ] = [0, 255, 0]
        #                 overlay[ np.where((overlay == [3, 3, 3]).all(axis=-1)) ] = [0, 0, 0]
        #                 overlay[ np.where((overlay == [4, 4, 4]).all(axis=-1)) ] = [0, 0, 255]
        #                 overlay[ np.where((overlay == [5, 5, 5]).all(axis=-1)) ] = [0, 0, 0]
        #
        #             if view in A2C_VIEWS:
        #
        #                 overlay[ np.where((overlay == [1, 1, 1]).all(axis=-1)) ] = [255, 0, 0]
        #                 overlay[ np.where((overlay == [2, 2, 2]).all(axis=-1)) ] = [0, 255, 0]
        #                 overlay[ np.where((overlay == [3, 3, 3]).all(axis=-1)) ] = [0, 0, 0]
        #                 overlay[ np.where((overlay == [4, 4, 4]).all(axis=-1)) ] = [0, 0, 255]
        #                 overlay[ np.where((overlay == [5, 5, 5]).all(axis=-1)) ] = [0, 0, 0]
        #
        #             if view in PLAX_VIEWS:
        #
        #                 overlay[ np.where((overlay == [1, 1, 1]).all(axis=-1)) ] = [255, 0, 0]
        #                 overlay[ np.where((overlay == [2, 2, 2]).all(axis=-1)) ] = [0, 255, 0]
        #                 overlay[ np.where((overlay == [3, 3, 3]).all(axis=-1)) ] = [0, 0, 0]
        #                 overlay[ np.where((overlay == [4, 4, 4]).all(axis=-1)) ] = [0, 0, 255]
        #                 overlay[ np.where((overlay == [5, 5, 5]).all(axis=-1)) ] = [0, 0, 0]
        #
        #             if view in PSAX_VIEWS:
        #
        #                 overlay[ np.where((overlay == [1, 1, 1]).all(axis=-1)) ] = [255, 0, 0]
        #                 overlay[ np.where((overlay == [2, 2, 2]).all(axis=-1)) ] = [0, 255, 0]
        #                 overlay[ np.where((overlay == [3, 3, 3]).all(axis=-1)) ] = [0, 0, 0]
        #                 overlay[ np.where((overlay == [4, 4, 4]).all(axis=-1)) ] = [0, 0, 255]
        #                 overlay[ np.where((overlay == [5, 5, 5]).all(axis=-1)) ] = [0, 0, 0]
        #
        #         overlay_vid = (0.7*overlay) + downsized_orig
        #
        #         skvideo.io.vwrite('segments/' + str(id) + '.mp4', overlay_vid)

            return jsonify(id='13', view='a4c', segmented='yep')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
