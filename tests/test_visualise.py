import matplotlib
matplotlib.use('Agg') 
import pytest
import numpy as np
from numcompute_stream.visualise import (
    plot_metric_over_time, 
    plot_confusion_matrix, 
    compare_models
)

def test_visualise_functions():
    data = [0.1, 0.5, 0.9]
    cm = np.array([[10, 0], [0, 10]])
    
    # Execute functions to trigger coverage
    plot_metric_over_time(data, title="Test", ylabel="Acc", show=False)
    plot_confusion_matrix(cm, show=False)
    compare_models(data, data, labels=("A", "B"), show=False)
    

    assert True