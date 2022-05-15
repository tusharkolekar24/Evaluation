class Evaluate:
      def __init__(self):
          pass

      def mean_absolute_error(self,true,pred):
          """
          Mean absolute error regression loss

          Parameters
          ----------
          y_true        : array-like of shape (n_samples,) or (n_samples, n_outputs)
                            Ground truth (correct) target values.

          y_pred        : array-like of shape (n_samples,) or (n_samples, n_outputs)
                            Estimated target values.

          Returns
          -------
          loss          : float or ndarray of floats
                            If multioutput is 'raw_values', then mean absolute error is returned
                            for each output separately.
                            If multioutput is 'uniform_average' or an ndarray of weights, then the
                            weighted average of all output errors is returned.

          MAE output is non-negative floating point. The best value is 0.0.

          where, ytrue is true value and ypred is predicted values.

          Examples
          --------
                    >>> y_true = [3, -0.5, 2, 7]
                    >>> y_pred = [2.5, 0.0, 2, 8]
                    >>> mean_absolute_error(y_true, y_pred)
          """
          import numpy as np          
          true = np.array(true).reshape(-1)
          pred = np.array(pred).reshape(-1)
          mae  = np.mean(np.abs(true-pred))
          return mae

      def mean_square_error(self,true,pred):
          """
            Mean squared error regression loss

            Parameters
            ----------
            y_true      : array-like of shape (n_samples,) or (n_samples, n_outputs)
                          Ground truth (correct) target values.

            y_pred      : array-like of shape (n_samples,) or (n_samples, n_outputs)
                          Estimated target values.

            Returns
            -------
            loss        : float or ndarray of floats
                          A non-negative floating point value (the best value is 0.0), or an
                          array of floating point values, one for each individual target.

            where, ytrue is true value and ypred is predicted values.

            Examples
            --------
                    >>> y_true = [3, -0.5, 2, 7]
                    >>> y_pred = [2.5, 0.0, 2, 8]
                    >>> mean_squared_error(y_true, y_pred)
          """
          import numpy as np          
          true = np.array(true).reshape(-1)
          pred = np.array(pred).reshape(-1)
          mse  = np.mean(np.square(true-pred))
          return mse
    
      def root_mean_square(self,true,pred):
          """
            Root Mean squared error regression loss

            Parameters
            ----------
            y_true      : array-like of shape (n_samples,) or (n_samples, n_outputs)
                          Ground truth (correct) target values.

            y_pred      : array-like of shape (n_samples,) or (n_samples, n_outputs)
                          Estimated target values.

            Returns
            -------
            loss        : float or ndarray of floats
                          A non-negative floating point value (the best value is 0.0), or an
                          array of floating point values, one for each individual target.

            where, ytrue is true value and ypred is predicted values.

            Examples
            --------
                    >>> y_true = [3, -0.5, 2, 7]
                    >>> y_pred = [2.5, 0.0, 2, 8]
                    >>> root_mean_squared(y_true, y_pred)
          """
          import numpy as np          
          true = np.array(true).reshape(-1)
          pred = np.array(pred).reshape(-1)
          mse  = np.mean(np.square(true-pred))
          rmse = np.sqrt(mse)  
          return rmse
        
      def mean_absolute_percentage_error(self,true,pred):
          """
            Mean absolute percentage error regression loss

            Parameters
            ----------
            y_true      : array-like of shape (n_samples,) or (n_samples, n_outputs)
                          Ground truth (correct) target values.

            y_pred      : array-like of shape (n_samples,) or (n_samples, n_outputs)
                          Estimated target values.

            Returns
            -------
            loss        : float or ndarray of floats
                          A non-negative floating point value (the best value is 0.0), or an
                          array of floating point values, one for each individual target.

            where, ytrue is true value and ypred is predicted values.

            Examples
            --------
                    >>> y_true = [3, -0.5, 2, 7]
                    >>> y_pred = [2.5, 0.0, 2, 8]
                    >>> mean_absolute_percentage_error(y_true, y_pred)
          """
          import numpy as np          
          true = np.array(true).reshape(-1)
          pred = np.array(pred).reshape(-1)          
          mape = np.mean(np.abs(true-pred)/true)*100
          return mape

Evaluate.__doc__
