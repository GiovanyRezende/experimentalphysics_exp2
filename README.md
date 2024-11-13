# Refractive Index Experiment
*This project is the data analysis of an refractive index experiment developed in the Experimental Physics class*. The libraries were Pandas (for dataframe creation), Numpy (for function operation in Pandas Series), Matplotlib (for data visualization) and Scikit-Learn (for R-squared metrics).

## Context
The main objective of the experiment is to determinate the refractive index of acrylic using Snell-Descartes Law.

### Materials
|Material|Duty|
|-|-|
|Ne-He Laser|Light beam source|
|Protactor|Angle measurement|
|Acrylic len|Refraction study object|

# DataFrame Creation
```
import pandas as pd
import numpy as np

n_acry = 1.49

dict = {'angle_1':[32,90,13,21,82.5],
        'angle_1_error':[1.5,1.5,1,1,0.5],
        'angle_2':[19.5,37,10.5,15,39.5],
        'angle_2_error':[0.5,1,0.5,1,0.5]}

df = pd.DataFrame(dict)

df['sin_1'] = round(np.sin(np.radians(df['angle_1'])),2)
df['sin_2'] = round(np.sin(np.radians(df['angle_2'])),2)
df['n'] = round(df['angle_1']/df['angle_2'],2)
df['sq_error'] = round((df['n'] - n_acry)**2,2)
df['angle_1_relative_error'] = round(df['angle_1_error']/df['angle_1'],2)
df['angle_2_relative_error'] = round(df['angle_2_error']/df['angle_2'],2)
df['n_error'] = round(df['n']*(df['angle_1_relative_error'] + df['angle_2_relative_error']),2)

df['angle_1'] = df['angle_1'].apply(lambda x: f"{x:.2f}")
df['angle_1_error'] = df['angle_1_error'].apply(lambda x: f"{x:.2f}")
df['angle_2'] = df['angle_2'].apply(lambda x: f"{x:.2f}")
df['angle_2_error'] = df['angle_2_error'].apply(lambda x: f"{x:.2f}")
```

The ```dict``` has the data extracted at the lab. The dataframe is the previous one:
![image](https://github.com/user-attachments/assets/8acc074a-e278-4965-873a-fe548a345a08)


# Data Visualization and Experiment Validation
```
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def sin_2_pred(angle_1):
  return np.sin(angle_1)/n_acry

x = np.arange(0,1.01,0.1)
y = sin_2_pred(x)

y_pred = df['sin_1']/n_acry
r2 = r2_score(df['sin_2'],y_pred)

plt.scatter(df['sin_1'],df['sin_2'],label='Extracted data')
plt.plot(x,y,label='Predicted data')
plt.xlabel("Sin 1")
plt.ylabel("Sin 2")
plt.legend()
plt.text(0,0.5,f'R^2: {round(r2*100,2)}%')
plt.show()
```
![image](https://github.com/user-attachments/assets/8fff424d-c32d-46dd-9ff1-5d1c74e869f2)

Overall, the experiment had a good R-squared with reasonable result and efficiency.

# Refractive Index Error
```
df['n_measure'] = pd.Series([f"{df['n'][x]} ± {df['n_error'][x]}" for x in range(len(df['n']))])
```
![image](https://github.com/user-attachments/assets/8258123e-2d97-4c32-8114-d868e03eea46)

# Experiment Accuracy
*Warning*: don't confuse "accuracy" here with the Machine Learning concept (the ratio between the number of correct classifications and the number of records in the training process). At Experimental Physics, accuracy is a boolean variable that means if the measurement has the accepted value (in the experiment, n = 1.49 for acrylic) in the error range of the measurement.

```
df['accuracy'] = (df['n'] - df['n_error'] < n_acry) & (df['n'] + df['n_error'] > n_acry)
```
![image](https://github.com/user-attachments/assets/0ac398c2-27aa-4bea-8272-c5a18bf71827)

It means the best measurement was the Number 4 (index 3 in Pandas dataframe).

<div align= center>

# Contact



[![logo](https://cdn-icons-png.flaticon.com/256/174/174857.png)](https://br.linkedin.com/in/giovanyrezende)
[![logo](https://images.crunchbase.com/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/v1426048404/y4lxnqcngh5dvoaz06as.png)](https://github.com/GiovanyRezende)[
![logo](https://logospng.org/download/gmail/logo-gmail-256.png)](mailto:giovanyrmedeiros@gmail.com)

</div>
