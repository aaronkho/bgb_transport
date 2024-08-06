import pytest
import numpy as np
import pandas as pd

import bgb_transport

@pytest.mark.bgb
@pytest.mark.usefixtures('sample_input_data', 'target_output_data')
class TestBgBOperation(object):

    debug_flag = False

    @pytest.fixture(scope='class')
    def sample_input_pandas(self, sample_input_data):
        return pd.DataFrame(data=sample_input_data)

    @pytest.fixture(scope='class')
    def sample_output_pandas(self, sample_input_pandas):
        return bgb_transport.model.predict(sample_input_pandas)

    def test_bgb_output_is_present(self, sample_output_pandas):
        assert isinstance(sample_output_pandas, pd.DataFrame)

    def test_bgb_variable_is_present(self, sample_output_pandas, target_output_data):
        for key in target_output_data.keys():
            assert key in sample_output_pandas

    def test_bgb_value_is_close(self, sample_output_pandas, target_output_data):
        for key, val in target_output_data.items():
            assert np.all(np.isclose(sample_output_pandas.loc[:, key], val, equal_nan=True))

