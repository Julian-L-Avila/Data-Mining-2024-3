---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# EDA
Julian Leonardo Avila Martinez
20212107030

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = "https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv"
df = pd.read_csv(url, sep=";")

df.head()
```

```python
df.shape
```

```python
df.columns
```

```python
df.info()
```

The data shows a list of 252 countries with information about certain aspects
such as Name, Capital, Continent, Area, Codes for telephone communication,
shipping, internet extension; Languages spoken on the territory and more.

One can see that the data types seem to be consistent, being quantities the
expected features, one may expect the phone to be a number type, however due
to "+" extension it is treated as an object or string.

There is a big amount of missing data, specially for the postal services.
Some capitals are missing, some continents and neighbours are missing as well
this could be due to being island nation without a clear continent of pertinence
and the lack of land borders with other nations.

Most notably there is one one country that has an equivalent flip code.

```python
df.describe()
```

The numeric info one has is:
- Area - Land are by its territory
- GeoName ID - Number that identifies the location
- Numeric - Numeric code that identifies country for communications
- Population - Amount of inhabitants

Due to the nature of the data, the only ones that could be related are the
area and the population, as countries with larger area are more likely to have
a bigger population.

```python
df_numeric = df[["area", "geoname_id", "numeric", "population"]]
df_numeric
```

```python
corr_var = df_numeric.corr()
print(corr_var)
```

```python
plt.imshow(corr_var, cmap="coolwarm", interpolation="none")
plt.colorbar()

plt.xticks(np.arange(len(corr_var.columns)), corr_var.columns)
plt.yticks(np.arange(len(corr_var.columns)), corr_var.columns)

for i in range(len(corr_var.columns)):
    for j in range(len(corr_var.columns)):
        plt.text(j, i, f"{corr_var.iloc[i, j] : .2f}", ha="center", va="center", color="black")

plt.title("Correlation matrix of numerical variables")
plt.tight_layout()
```

As it was expected, there is no high correlation of variables, being areas and
population the closest related by a 0.41 metric. As mentioned before, the
higher area allows for a higher population, however due to some countries being
huge area but not being densely populate as the (US, China, Russia, Canada),
the correlation is less than 5.0.


# Population Growth

```python
url_pop = "https://raw.githubusercontent.com/DrueStaples/Population_Growth/master/countries.csv"
df_pop = pd.read_csv(url_pop, sep=",")
df_pop.head()
```

```python
df_pop.info()
```

```python
df_pop_es = df_pop[df_pop["country"] == "Spain"]
df_pop_es.head()
```

```python
df_pop_es["population"].plot.bar()
plt.xlabel("Year")
plt.ylabel("Population")
plt.show()
```

```python
df_pop_esar = df_pop[df_pop["country"].isin(["Argentina", "Spain"])]
df_pop_esar.head()
```

```python
pivoted = df_pop_esar.pivot(index='year', columns='country', values='population')
pivoted.plot.bar(figsize=(12, 6))
plt.title('Population Comparison: Country A vs. Country B')
plt.ylabel('Population')
plt.show()
```

The graph allows to see the population growth of Spain and Argentina,
while in 1952 the difference is less than half of Spain population, on 2007
that difference decreases by a lot.
It shows how the population growth Argentina experienced was bigger than Spain's

```python
df_es = df[df["languages"].notnull() & df["languages"].str.startswith("es")]
df_es.info()
```

```python
df_es.head(20)
```
