import pytest
import numpy as np

import bgb_transport

def hardcoded_sample_input_data():
    input = {
        'lref': [0.5, 1.0],
        'bref': [12.0, 3.0],
        'betae': [0.001, 0.005],
        'nuei': [0.05, 0.2],
        'ate': [2.0, 3.0],
        'ati': [2.0, 3.0],
        'ane': [0.75, 1.0],
        'tinorm': [0.98, 1.02],
        'ninorm': [0.85, 0.9],
        'q': [1.7, 2.0],
        'zeff': [1.4, 1.7],
        'eps': [0.3, 0.4],
        'x': [0.7, 0.8],
    }
    return input

def hardcoded_target_output_data():
    output = {
        'chie_bgb': [0.2973590088482835, 1.6216836277498352],
        'chii_bgb': [0.7828508510577137, 1.394458391045369],
        'de_bgb': [0.17024684907225685, 0.5698145012622337],
        'di_bgb': [0.17024684907225685, 0.5698145012622337],
        'qe_bgb': [0.594718017696567, 4.865050883249506],
        'qi_bgb': [1.5657017021154274, 4.183375173136107],
        'gammae_bgb': [0.12768513680419263, 0.5698145012622337],
        'gammai_bgb': [0.12768513680419263, 0.5698145012622337],
    }
    return output

@pytest.fixture(scope='module')
def sample_input_data():
    return hardcoded_sample_input_data()

@pytest.fixture(scope='module')
def target_output_data():
    return hardcoded_target_output_data()

