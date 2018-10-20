import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf as plt_pdf
import pandas as pd


df = pd.read_excel('~/Documents/UTSW/AI/Lung-SBRT-Data.xlsx')
df = df.values

print(df)
