# Multiple Linear Regression

`Y` - target varible    
`X` - 1xn matrix containing feature values    
`Beta` -  nxm matrix containing regression coefficients    
`E` - nx1 matrix containing error values    
    
`SST` - Total Sum of Squared    
`SSR` - Residual Sum of Squares    

## Formulae used in regression:

* Formula used to calculate the regression coefficients -
    
$$
\beta = (X^T X)^{-1} X^T Y
$$

* Formula used to calculate the error matrix - 
  
$$
E = Y - X \beta
$$

## Formulae used to calculate scores and error metrics: 
* Formulae used to calculate R^2 score - 
      
$$
SST = \sum (Y_i - \bar{Y})^2
$$

$$
SSR = \sum E_i^2     
$$

$$
R^2 = 1 - \frac{SSR}{SST}
$$

* Formulae used to calculate different error values - 
  
$$
MSE = \frac{1}{n} \sum (Y_{\text{gen}} - Y_{\text{pred}})^2
$$

$$
RMSE = \sqrt{MSE}
$$

$$
MAE = \frac{1}{n} \sum |Y_{\text{gen}} - Y_{\text{pred}}|
$$

$$
MAPE (in percentage) = \frac{100}{n} \sum \left| \frac{Y_{\text{gen}} - Y_{\text{pred}}}{Y_{\text{pred}}} \right| \% 
$$
