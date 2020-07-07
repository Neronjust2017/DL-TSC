import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import parallel_coordinates

data = pd.read_csv('C:\Users\Administrator\Desktop\nas\inceptiontime-greedy.csv')
data_1 = data[['Species', 'Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width']]

parallel_coordinates(data_1, 'Species')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fancybox=True, shadow=True)
plt.show()
