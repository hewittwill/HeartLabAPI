from keras.layers import Layer, concatenate

CLASS_NAMES = [
    'plax_far',
    'plax_plax',
    'plax_laz',
    'psax_az',
    'psax_mv',
    'psax_pap',
    'a2c_lvocc_s',
    'a2c_laocc',
    'a2c',
    'a3c_lvocc_s',
    'a3c_laocc',
    'a3c',
    'a4c_lvocc_s',
    'a4c_laocc',
    'a4c',
    'a5c',
    'other',
    'rvinf',
    'psax_avz',
    'suprasternal',
    'subcostal',
    'plax_lac',
    'psax_apex'
]

PLAX_VIEWS = [
    'plax_plax'
]

PSAX_VIEWS = [
    'psax_pap'
]

A4C_VIEWS = [
    'a4c_laocc',
    'a4c'
]

A3C_VIEWS = [
    'a3c_laocc',
    'a3c'
]

A2C_VIEWS = [
    'a2c_laocc',
    'a2c'
]

SEGMENT_VIEWS = PLAX_VIEWS + PSAX_VIEWS + A4C_VIEWS + A3C_VIEWS + A2C_VIEWS

class Tile(Layer):

    def __init__(self, **kwargs):
        super(Tile, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Tile, self).build(input_shape)

    def call(self, x):
        return concatenate([x-24, x-24, x-24], axis=3)

    def compute_output_shape(self, input_shape):
        return (None, 224, 224, 3)
