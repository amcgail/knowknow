from knowknow import *
import os
import streamlit as st
import pandas as pd
basedir = os.getcwd()
@st.cache(allow_output_mutation=True)
def initknowknow(basedir,selected_keys):
    kk = KnowKnow(NB_DIR=basedir, BASEDIR=basedir)
    #'counting'
    #cnt = kk.get_cnt("sociology-wos", 'doc', ['ty', 'ta', 'c', 'fa.c', 'c.fy', 'c1.c2'])
    #'knowknow initialized'
    return kk.get_cnt("sociology-wos", 'doc', selected_keys) #['ta', 'c','c.fy'])
    

#@st.cache
#def get_most_c(cnt):
    #return Counter(cnt['ta']).most_common(50)
st.title('KnowKnow ,most cited authors ')
st.sidebar.header('Configure parameter')
dataset_list =['Sociology-wos','Socialogy-jstor'] 
dataset=st.sidebar.selectbox('Select dataset',dataset_list,0)
keys = ['ty', 'ta', 'c', 'fa.c', 'c.fy', 'c1.c2']
selected_keys=st.sidebar.multiselect('Choose Keys',keys,['ta', 'c','c.fy'])
#kk = KnowKnow( NB_DIR=basedir, BASEDIR=basedir )
#'knowknow class'
cnt = initknowknow(basedir,selected_keys)
#cnt = kk.get_cnt("sociology-wos", 'doc', ['ty', 'ta', 'c', 'fa.c', 'c.fy', 'c1.c2'])
#'count loaded'
a= 10

most_c = Counter(cnt['ta']).most_common(50) #get_most_c(cnt)
most_author_list =[ x.ta for x,c in most_c]
#st.write(str(x.ta)+' , '+str(c))
#st.write(most_author_list)
selected_author=st.selectbox('Select author to examine ,among 50 most common ',most_author_list,0) 

if st.button('Show 50 most common cited authors,Citations'):
    df = pd.DataFrame.from_records(most_c, columns=['Author','Citaton'])
    #df = df.rename(columns={'index':'Author', 0:'Citations'})
    st.dataframe(df)
    #for x,c in most_c:
        #st.write(str(x.ta)+' , '+str(c))


year = st.sidebar.slider('Select year', 1950, 2020,2010)

from random import choice
to_examine = choice(most_c)

to_examine = to_examine[0].ta
to_examine = selected_author
st.info('Examining most c '+str(to_examine))
my_cits = [x for x in cnt['c'] if x.c.split("|")[0] == to_examine]
#my_cits

yr_plts = [
    [ cnt['c.fy'][(x.c,YY)] for YY in range(1950,year) ]
    for x in my_cits
]
fig1,ax = plt.subplots()
#year = 2020

ax.stackplot(range(1950,year),*yr_plts, labels=range(len(my_cits)));
#ax.title("Works cited with "+ str(to_examine))

st.pyplot(fig1)