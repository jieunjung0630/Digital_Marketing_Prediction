import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

st.set_option('deprecation.showPyplotGlobalUse', False)
# Loading dataset 
df = pd.read_csv('data/income_df.csv')
df = df[[    'Age',
             'Education',
             'Income',
             'Mnt_Product',
#              'NumCatalogPurchases',
#              'NumDealsPurchases',
#              'NumStorePurchases',
#              'NumWebPurchases',
#              'NumWebVisitsMonth',
             'NumPurchases',
             'Sum_AcceptedCmp'
         
        ]]

st.title('Jieun Jung Project')
st.header('EDA: Digital Marketing Campaign')

st.subheader("DataFrame")
st.dataframe(df)
selected_column = st.sidebar.selectbox('Select a column to visualize', df.columns)

st.write("Histogram Plots")
sns.histplot(df[selected_column])
st.pyplot()

st.write("Scatter plot")
x_axis = st.sidebar.selectbox('Select the x-axis', df.columns)
y_axis = st.sidebar.selectbox('Select the y-axis', df.columns)

fig = px.scatter(df, x=x_axis, y=y_axis)
st.plotly_chart(fig)

st.write("Pair Plot")
sns.pairplot(df, hue='Education')
st.pyplot()
st.write("Description of the data")
st.table(df.describe())

st.header('Correlation Matrix')

corr = df.corr()
sns.heatmap(corr, annot=True)
st.pyplot()

st.header('Boxplot')

fig = px.box(df, y=selected_column)
st.plotly_chart(fig)

selected_class = st.sidebar.selectbox('Select a Education to visualize', df['Education'].unique())

if st.sidebar.button('Show Violin Plot'):
    
    fig = px.violin(df[df['Education'] == selected_class], y=selected_column)
    
    st.plotly_chart(fig)







