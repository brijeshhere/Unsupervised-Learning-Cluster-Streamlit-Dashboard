import streamlit as st
import pickle
import pandas as pd
import duckdb as duck
import os
import plotly.express as px
from plotly.subplots import make_subplots
from umap.umap_ import UMAP

# duckdb- making connection
conn=duck.connect('country_cluster.db')

# Streamlit

st.set_page_config(page_title='Clustering on Countries',layout='wide')
st.title('Clustering on Countries')

# Prblem Description

st.markdown("""# Problem Statement

Problem Statement :¶

HELP International is an international humanitarian NGO that is committed to fighting poverty and providing the people of backward countries with basic amenities and relief during the time of disasters and natural calamities. HELP International have been able to raise around $ 10 million. This money now needs to be allocated strategically and effectively. Hence, inorder to decide the selection of the countries that are in the direst need of aid, data driven decisions are to be made. Thus, it becomes necessary to categorise the countries using socio-economic and health factors that determine the overall development of the country. Thus, based on these clusters of the countries depending on their conditions, funds will be allocated for assistance during the time of disasters and natural calamities. It is a clear cut case of unsupervised learning where we have to create clusters of the countries based on the different feature present.""")

# Data Description

st.markdown("""# Dataset Desciption

    country : Name of the country
    child_mort : Death of children under 5 years of age per 1000 live births
    exports : Exports of goods and services per capita. Given as %age of the GDP per capita
    health : Total health spending per capita. Given as %age of GDP per capita
    imports : Imports of goods and services per capita. Given as %age of the GDP per capita
    Income : Net income per person
    Inflation : The measurement of the annual growth rate of the Total GDP
    life_expec : The average number of years a new born child would live if the current mortality patterns are to rem...
    total_fer : The number of children that would be born to each woman if the current age-fertility rates remain th...
    gdpp : The GDP per capita. Calculated as the Total GDP divided by the total population.
""")
# Header- Explore
st.header("Explore the Data",divider='rainbow')
# Stats- Take input
stats=st.multiselect('Select a Statistic to explore the data: ',conn.sql("""from country limit 0""").df().columns[1:])


# Graph stats
if stats:
    fig=make_subplots(rows=len(stats),cols=1,subplot_titles=([value.upper() for value in stats]))
    for i in range(len(stats)):
        fig.add_trace(px.bar(data_frame=conn.sql(f"select country,{stats[i]} from country order by {stats[i]} desc").df(),x='country',y=f'{stats[i]}',title=f"{stats[i]} VS Countries").data[0],row=i+1,col=1)
    fig.update_layout(height=400*len(stats),width=1000)
    st.plotly_chart(fig,use_container_width=True)
else:
    st.write("Select Above")

# Ingesting data
data=conn.sql("""select * exclude country from country""").df()

# Umap Data
_umap=UMAP(n_components=3,random_state=42)
_umap.fit_transform(data)
embed=pd.DataFrame(_umap.embedding_)

# HDBSCAN
from hdbscan import HDBSCAN
hdb=HDBSCAN()
hdb.fit(embed)

# Dataframe with country name
withcountry_df=pd.concat([conn.sql('select country from country').df(),embed],axis=1)

# labels add to dataframe
withcountry_df['labels']=hdb.labels_
withcountry_df.loc[(withcountry_df['labels']==-1),'Class']='Outlier'
withcountry_df.loc[(withcountry_df['labels']==0),'Class']='Help Needed'
withcountry_df.loc[(withcountry_df['labels']==1),'Class']='May Need help'
withcountry_df.loc[(withcountry_df['labels']==2),'Class']='May Need help'
withcountry_df.loc[(withcountry_df['labels']==3),'Class']='Do not need help'
withcountry_df.loc[(withcountry_df['labels']==4),'Class']='Do not need help'

# Graph- Cluster
st.header('Clustering of Countries',divider='rainbow')
st.subheader('Umap With HDBSCAN')
fig=px.scatter_3d(x=withcountry_df[0],
               y=withcountry_df[1],
               z=withcountry_df[2],
               color=withcountry_df['Class'],
               hover_name=withcountry_df['country'],
               color_continuous_scale=px.colors.named_colorscales()[2])
fig.update_layout(height=800,width=1000,template='ggplot2',legend=dict(title='Country'))
st.plotly_chart(fig,use_container_width=True)

# Map
st.header('Help required as per Country')
fig_map=px.choropleth(data_frame=withcountry_df[['country','Class']],
                      locationmode='country names',
                      locations='country',
                      color='Class',
                      color_discrete_map={
                        'Outlier':'Black',
                        'Help Needed':'Red',
                        'May Need help':'Yellow',
                        'Do not need help':'Green'})
st.plotly_chart(fig_map,use_container_width=True)

st.markdown("""#Conclusion
            
            The countries marked red required help.""")

