#!/usr/bin/env python
# coding: utf-8

# **Universidad Privada Boliviana**
# 
# **Extensión Universitaria**
# 
# **Experto en Análisis estadístico de datos con Python**
# 
# **Docente: Mauro Delboy Ph.D(c)**
# 
# ## <div align="center">Trabajo Final - Experto en análisis estadístico de datos con Python

# GRUPO: **GPD**

# *Integrantes:*
# - Carlos Andrés Pérez Guzmán
# - Gerson Flores
# - José Esponoza
# - Jhonny Mammani

# ---

# ### 1. Librerías

# In[42]:


# Librerías de trabajo y cálculo
import pandas as pd, numpy as np

# Cálculo de la moda
from statistics import multimode

# Gráficos
import matplotlib.pyplot as plt, seaborn as sns

# Resolución de gráficos
plt.rcParams['figure.dpi'] = 90


# ### 2. Carga de datos

# In[2]:


df = pd.read_csv(filepath_or_buffer = 'Data/zapatos.csv', sep = ',', decimal = '.', parse_dates = ['Date'])
df.head()


# ### 3. Generalidades del origen de datos

# Existe un solo archivo de datos consolidado con 14967 registros y 12 variables bien definidas de tipo cualitativo y cuantitativo, las cuales no presentan valores faltantes (NaN). Bajo un vistazo general, las filas representan la venta de zapatos identificada por un número de factura. Las columnas definen las características y condiciones de dicha venta.
# 
# Por otro lado, se debe observar y validar la integridad de los datos para asignar el tipo correcto de vairable.

# In[3]:


df.info()


# In[4]:


df.isnull().values.any()


# El número de factura no representa a una fila exclusivamente. Por ejemplo, existe una factura N° 62453 está presente en 5 filas del dataset. Esto quiere decir que cada fila representa la venta de un producto en cierto número de factura y según las características otorgadas por las demás variables. De esta manera, no se necesita de otro índice del que ya se dispone en el Dataframe.

# In[5]:


df['InvoiceNo'].value_counts().value_counts()


# In[6]:


df['InvoiceNo'].value_counts().head(1)


# In[7]:


df.loc[df['InvoiceNo'] == 62453]


# ### 4. Transformaciones preliminares

# In[8]:


df_01 = df.copy()


# In[9]:


df_01.dtypes.index


# In[10]:


# Eliminación de los caracteres '%' y '$'
df_01.replace(to_replace = ['%','\$'], value = '', regex = True, inplace = True)

# Asignación de variables numéricas
var = ['InvoiceNo','ProductID','Size (US)', 'Size (UK)', 'UnitPrice', 'Discount','SalePrice']
df_01[var] = df_01[var].apply(pd.to_numeric, errors = 'coerce')

# Asignación de variables cualitativas
df_01 = df_01.astype({'Gender':'category','Country':'category'})

df_01.info()
print(df_01.isnull().values.any())


# ### 5. Consignas de trabajo
# Ahora que los datos fueron revisados, se pueden ejecutar análisis y estraer información útil

# #### 5.1. Utilice la base de datos zapatos.csv y realice un análisis estadístico obteniendo media, mediana, moda y demás medidas de tendencia central y de dispersión.

# In[11]:


# Formato de visualización
pd.options.display.float_format = '{:.2f}'.format

# Regresar a los valroes por defecto
#pd.reset_option('all', silent = True)


# Análisis de vairables cuantitativas

# In[43]:


df_01[['Date','Size (US)','Size (UK)','UnitPrice','Discount','SalePrice']].describe(datetime_is_numeric = True)


# Análisis de variables cualitativas

# In[13]:


df_01.describe(exclude = ['float64','int64','datetime64[ns]'])


# Análisis de modas en el Dataset

# In[14]:


df_01.apply(multimode)


# #### 5.2. Genere una variable llamada "Dif_Price" que muestre la diferencia entre el precio de venta (SalePrice) y el precio unitario (UnitPrice).

# In[15]:


df_01['Dif_Price'] = df_01['SalePrice'] - df_01['UnitPrice']
df_01.head()


# #### 5.3. Si un zapato de talla 9 en Estados Unidos (SizeUS=9.0), equivale a un zapato talla 40 en Latinoamérica, genere una columna llamada Size(LATAM) que muestre la talla de zapatos en Latinoamérica.

# Se asume que el cambio de talla en un zapato es lineal, por tanto la operación que contendrá la columna "Size(LATAM)" será:
# 
# $$\text{Size (LATAM)} =  \text{Size (US)} \cdot \frac{40}{9}$$

# In[16]:


df_01.insert(loc = 7, column = 'Size(LATAM)', value = df_01['Size (US)'] * (40/9))
df_01.head()


# #### 5.4. Realice un análisis de visualización de datos de la base zapatos.csv, utilizando todas las gráficas vistas en clase. Es importante que realice la interpretación de cada uno de estos gráficos

# Las características de los datos muestran que la información se venera a partir de una venta. Por lo cual las variables de agregación serán aquellas dependientes del precio, el descuento y el tipo de producto. 

# ##### a) Facturas con más registros de productos

# La cantidad de productos adquiridos por venta (factura) se caracteriza por incluir un solo para de zapatos. Los casos que incluyen más productos por factura son muy poco frecuentes. 

# In[17]:


sns.set_theme(style = 'ticks',palette = 'tab20c', font = 'Arial')

aux_a = df_01['InvoiceNo'].value_counts().value_counts()

sns.barplot(x = aux_a.index, y = aux_a.values)
plt.title(label = 'Productos vendidos por factura', loc = 'left', size = 12, fontstyle = 'italic', fontweight = 'bold')
plt.xlabel(xlabel = 'Unidades vendidas', size = 12)
plt.ylabel(ylabel = 'Cantidad de Facturas', size = 12)
sns.despine(offset = 5, trim = False, bottom = False)


# In[18]:


aux_a


# ##### b) Ventas por país

# EEUU es el país que reporta más ingresos por calzados con un 39%, seguido por Alemania, luego Canadá, y, en último lugar con un 12%, El Reino Unido

# In[19]:


plt.title(label = 'Proporción de ingresos por país', loc = 'left', size = 12, fontstyle = 'italic', fontweight = 'bold')
aux_b = df_01.groupby(by = ['Country']).sum(numeric_only = True)['SalePrice']
plt.pie(aux_b.values, labels = aux_b.index, autopct = '%.0f%%')
plt.show()


# Se puede notar que los zapatos de hombres proporcionan un ingreso más alto respecto que los de mujeres. Asimismo, los ingresos en EEUU y Alemnaia son las más altos independientemente del género.

# In[20]:


sns.set_theme(style = 'ticks',palette = 'GnBu_r', font = 'Arial')

bar_order = df_01['Country'].value_counts(ascending = False).index.tolist()

g = sns.catplot(data = df_01, x = 'Country', y = 'SalePrice', order = bar_order,
            kind = 'bar', estimator = 'sum', col = 'Gender',
            errorbar = None)
g.set_axis_labels(x_var = '', y_var = 'Ingresos [$]')
g.set_titles('{col_name}')

plt.suptitle(t = 'Ingresos por país y género', y = 1.05, x = 0.15,
             fontstyle = 'italic', fontweight = 'bold', size = 12)

sns.despine(offset = 5, trim = False, bottom = False)


# In[21]:


(df_01.groupby(by = ['Country','Gender']).
 sum(numeric_only = True)['SalePrice'].
 reset_index(name = 'Ingresos').
 sort_values(by = ['Country','Ingresos'], ascending = False))


# ##### c) Distribución de la talla por género

# Casi la mitad de los zapatos de hombre se encuentran por encima de la talla 40 , mientras que 75% de los datos de los zapatos de mujeres se encuentra por debajo de la talla 40. La dispersión de los datos es similar aunque los hombres tienen más valores atípicos.

# In[22]:


sns.set_theme(style = 'ticks',palette = 'GnBu_r', font = 'Arial')

sns.catplot(data = df_01, x = 'Gender', y = 'Size(LATAM)',kind = 'box')

plt.title(label = 'Distribución de tallas', loc = 'left', size = 12, fontstyle = 'italic', fontweight = 'bold')
plt.xlabel(xlabel = None)
plt.ylabel(ylabel = 'Talla', size = 12)
sns.despine(offset = 5, trim = False, bottom = False)


# ##### d) Relación entre precio unitario y talla

# A medida que la talla de los zapatos crece, el precio cambia en proporciones parecidas con una tendencia lineal. 

# In[23]:


sns.set_theme(style = 'whitegrid', font = 'Arial')

sns.relplot(data = df_01, x = 'Size(LATAM)', y = 'Size (UK)', hue = 'Gender',
            facet_kws={'sharey': True, 'sharex': True})

plt.suptitle(t = 'Size(UK) vs Size(LATAM)', y = 1.05, x = 0.15,
             fontstyle = 'italic', fontweight = 'bold', size = 12)

plt.subplots_adjust(hspace = 0.2, wspace = 0.1)
plt.xlim(10,80)
plt.ylim(0,16);


# #### 5.5. Habiendo generado las dos anteriores columnas y utilizando funciones condicionales responda a las siguientes preguntas:

# **¿Cuánto cuesta en promedio un zapato para hombre en Estados Unidos?**
# 
# El precio nominal de un par de zapatos en EEUU es de 163.03 dólares, pero si se adquieren con descuento el precio promedio sería 143.58 dólares.

# In[24]:


aux = df_01.groupby(by = ['Country','Gender']).mean(numeric_only = True)[['UnitPrice','Discount','SalePrice']]
aux


# **¿En qué país se venden los zapatos más caros?**
# 
# Los zapatos más caros tienen un precio de 199 $. Este precio de venta, no tinene descuento y se puede encontrar en los cuatro paises del dataset, sin embargo, EEUU tiene más registros de ventas respecto a los demás paises.

# In[25]:


df_01['SalePrice'].max()


# In[26]:


df_01['Country'].unique()


# In[50]:


df_01.loc[df_01['SalePrice'] == 199, 'Country'].value_counts()


# **¿En qué tienda (Variable Shop) se venden los zapatos más baratos?**

# Las tiendas cuyo precio es mínimo corresponde a GRE1 y GER2 establecidas en Alemania con un valor de 33 y 30 dólares respectivamente.

# In[28]:


df_01['SalePrice'].min()


# In[54]:


df_01.loc[df_01['SalePrice'] == df_01['SalePrice'].min(),['Shop','Country']].value_counts().head()


# **¿En qué fecha se hizo el descuento más alto en el Reino Unido?**

# El descuento más alto para el Reino Unido se llevó a cabo entre el 26/01/2014 y 17/12/2016

# In[31]:


# Descuento más alto
df_01.loc[df_01['Country'] == 'United Kingdom','Discount'].max()


# In[56]:


df_01.loc[(df_01['Country'] == 'United Kingdom') & (df_01['Discount'] == 50),'Date'].min()


# In[33]:


df_01.loc[(df_01['Country'] == 'United Kingdom') & (df_01['Discount'] == 50),'Date'].max()


# **¿De qué talla americana (Variable Size(US)) es el zapato femenino más grande?**
# 
# El zapato femenino más grade para mujer es 12

# In[74]:


df_01.sort_values(by = ['Size (US)'], ascending = False).loc[df_01['Gender'] == 'Female','Size (US)'].head()


# **¿En qué país se han hecho más descuentos?**
# 
# Se hicieron más decuentos en EEUU

# In[81]:


df_01.head()


# In[83]:


df_01.groupby(by = ['Country']).sum(numeric_only = True)['Dif_Price'].sort_values(ascending = True)


# **Basándose solo en el precio (Sale Price), ¿en qué país prefería comprar una mujer que calza talla 6.5 americana?**
# 
# SI se toma la media como un decisor es más recomendable comprar en EEUU. Si tomamos el valor mínimo, es mejor en Canadá.

# In[36]:


df_01.head()


# In[85]:


(df_01.loc[(df_01['Size (US)'] == 6.5) & (df_01['Gender'] == 'Female'),['Country','SalePrice','Size (US)','Gender']].
 groupby(by = ['Country']).mean(numeric_only = True).sort_values(by = ['SalePrice']))


# In[38]:


(df_01.loc[(df_01['Size (US)'] == 6.5) & (df_01['Gender'] == 'Female'),['SalePrice','Country']].
 groupby(by = ['Country'])).min().sort_values(by = ['SalePrice'])


# **Realice las gráficas de la evolución de ventas en cada país e interprete el mercado que tiene mejor perspectiva de ventas.**

# El análisis de evolución de las ventas muestra que el mercado de EEUU tuvo la mejores ventas a los largo de los años. Sin embargo, su tendencia está en el descenso. Por otra parte, el mercado en Canada tuvo ventas ligeramente crecientes, con vaivenes pero con señales de crecimiento.

# In[172]:


sns.set_theme(style = 'whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

df_02 = df_01.copy()
for i in df_02['Country'].unique().tolist():
    aux = df_02.copy().loc[df_02['Country'] == i]
    aux = aux.set_index('Date')
    aux = aux.resample(rule = 'M').sum(numeric_only = True)
    sns.lineplot(data = aux, x = 'Date', y = 'SalePrice', label = i,
                 marker = "o", linestyle = 'dashed', markersize = 7)

ax.legend()
plt.title(label = 'Evolución de ventas', loc = 'left', size = 12, fontstyle = 'italic', fontweight = 'bold')
plt.xlabel(xlabel = None)
plt.ylabel(ylabel = 'Precio de venta', size = 12);


# In[181]:


# Valores de la gráfica de tiempo

df_03 = df_01.copy()
df_03['year'] = pd.DatetimeIndex(df_01['Date']).year
df_03['month'] = pd.DatetimeIndex(df_01['Date']).month

df_03 = pd.pivot_table(df_03, index = ['year','month'],
                       values = ['SalePrice'],
                       columns = ['Country'],
                       aggfunc = np.sum)
df_03

