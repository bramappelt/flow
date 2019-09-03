import unittest
import pandas as pd
import numpy as np

from waterflow.utility.statistics import RMSE, MAE

class TestStatisctics(unittest.TestCase):
    ''' Test all functions in this package '''
    def test_RMSE_MAE(self):
        arr1 = np.array([1,2,3,1,2,3,4,5,2,8,5,3])
        arr2 = np.array([2,2,1,3,4,6,2,3,4,5,6,7])
        
        dataframe = pd.DataFrame()
        dataframe["col1"] = pd.Series(arr1)
        dataframe["col2"] = pd.Series(arr2)
        
        rmse_df = RMSE(dataframe)
        self.assertAlmostEqual(2.23606797749979, rmse_df)
        
        rmse_arr = RMSE(arr1, arr2)
        self.assertAlmostEqual(2.23606797749979, rmse_arr)
        
        mae_df = MAE(dataframe)
        self.assertAlmostEqual(2.0, mae_df)
        
        mae_arr = MAE(arr1, arr2)
        self.assertAlmostEqual(2.0, mae_arr)
    
if __name__ == "__main__":
    unittest.main()