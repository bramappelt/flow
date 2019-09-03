""" This package contains some functions which allow for the calculation of
certain statistical measures. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def RMSE(*args):
    """ The Root Mean Square Error.
    
    Returns the Root Mean Square Error based on two individual arrays or the
    last two entries of a dataframe.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        A dataframe with 2 columns and at least 2 rows. The columns in the 
        dataframe should be of equal length.
        
    arr1 : list/numpy.ndarray
        An 1-dimensional array.
        
    arr2 : list/numpy.ndarray
        An 1-dimensional array.
        
    Returns
    -------
    numpy.float64
        Floating point value of the Root Mean Square Error.
        
    Notes
    -----
    Notes about the implementation algorithm.
    
    The formula implemented:
    
    .. math:: RMSE=\\sqrt{\\frac{\\sum{(X_i -X_j)^2}}{N}}

    Examples
    --------
    Create a dataframe and populate it with random values. Note: The data
    arrays can also be passed to the function as separate arguments.
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> dataframe = pd.DataFrame()
    >>> dataframe["col1"] = pd.Series(np.array([1,2,3,1,2,3,4,5,2,8,5,3]))
    >>> dataframe["col2"] = pd.Series(np.array([2,2,1,3,4,6,2,3,4,5,6,7]))
    >>> rmse = RMSE(dataframe)
    >>> print(rmse)
    2.23606797749979
    
    """
    
    # If the single argument is of type pandas.core.frame.DataFrame
    if isinstance(args[0], pd.core.frame.DataFrame):
        df = args[0]
        # two series needed for RMSE calculation
        if len(df.columns) < 2:
            return np.nan
        # return RMSE
        return(np.sqrt(((df.iloc[::,-2] - df.iloc[::,-1])**2).mean()))
    
    # If the first argument is of type list or np.ndarray
    elif isinstance(args[0], list) or isinstance(args[0], np.ndarray):
        arr1 = np.array(args[0])
        arr2 = np.array(args[1])
        return(np.sqrt(((arr1 - arr2)**2).mean()))
    
def MAE(*args):
    """ The Mean Absolute Error.
    
    Returns the Mean Absolute Error based on two individual arrays or the
    last two entries of a dataframe.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        A dataframe with 2 columns and at least 2 rows. The columns in the 
        daraframe should be of equal length.

    arr1 : list/numpy.ndarray
        An 1-dimensional array.
        
    arr2 : list/numpy.ndarray
        An 1-dimensional array.
        
    Returns
    -------
    numpy.float64
        Floating point value of the Mean Absolute Error.
        
    Notes
    -----
    Notes about the implementation algorithm.
    
    The formula implemented:
    
    .. math:: MAE=\\sqrt{\\frac{\\lvert X_i -X_j\\rvert}{N}}
    
    Examples
    --------
    Create a dataframe and populate with random values. Note: The data
    arrays can also be passed to the function as separate arguments.
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> dataframe = pd.DataFrame()
    >>> dataframe["col1"] = pd.Series(np.array([1,2,3,1,2,3,4,5,2,8,5,3]))
    >>> dataframe["col2"] = pd.Series(np.array([2,2,1,3,4,6,2,3,4,5,6,7]))
    >>> mae = MAE(dataframe)
    >>> print(mae)
    2.0
    
    """
    
    # If the single argument is of type pandas.core.frame.DataFrame
    if isinstance(args[0], pd.core.frame.DataFrame):
        df = args[0]
        # two series needed for RMSE calculation
        if len(df.columns) < 2:
            return np.nan
        # return RMSE
        return(abs(df.iloc[::,-2] - df.iloc[::,-1]).mean())
    
    # If the first argument is of type list or np.ndarray
    elif isinstance(args[0], list) or isinstance(args[0], np.ndarray):
        arr1 = np.array(args[0])
        arr2 = np.array(args[1])
        return(abs(arr1 - arr2).mean())  

                      
if __name__ == "__main__":
    # check code in the docstrings
    import doctest
    doctest.testmod()
    
    df = pd.DataFrame()
    all_rmse = []
    all_mae = []
    for i in range(20):
        df[str(i)] = pd.Series(np.random.randn(10))
        all_rmse.append(RMSE(df))
        all_mae.append(MAE(df))
    all_rmse = np.array(all_rmse)    
            
    rmse = RMSE(df)
    mae = MAE(df)
    plt.plot(all_rmse, color = "red", label = "RMSE")
    plt.plot(all_mae, color = "blue", label = "MAE")
    plt.legend()
    plt.grid()
    plt.show()
