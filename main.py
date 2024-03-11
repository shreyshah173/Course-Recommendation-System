from bs4 import BeautifulSoup
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_lottie import st_lottie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import logging.config

logging.config.fileConfig('temp.txt')
logger = logging.getLogger('Course')


navbar = """
    <style>
        nav {
            display: flex;
            justify-content: space-between;
            padding: 1em;
            background-color: #333;
            color: white;
        }

        nav a {
            text-decoration: none;
            color: white;
            margin: 0 1em;
        }
    </style>

    <nav>
        <div>
            <a href="#" style="font-weight: bold;">Home</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </div>
        <div>
            <a href="#">Login</a>
            <a href="#">Sign Up</a>
        </div>
    </nav>
"""
st.markdown(navbar, unsafe_allow_html=True)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

#Edureka Courses
def find_courses_masters(text):
    if not text:
        print('Error during course finding: Empty search query')
        return []

    courses_info = []

    text = text.replace(' ', '%20')
    base_url = "https://www.edureka.co/"
    search_url = f"{base_url}search/{text}"

    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            courses = soup.find_all('div', class_='courseinfo')

            for course in courses:
                course_name = course.find('h3').text.strip()
                rating = course.find('span', class_='rating').text.strip()

#                 # working here

#                 learners_tag = course.find('span', class_='stats')
#                 learners = learners_tag.find('span').text.strip() if learners_tag else 'N/A'

#                 #working on learners

#                 total_reviews_tag = course.find('span', class_='totalreviews')
#                 total_reviews = total_reviews_tag.text.strip('()') if total_reviews_tag else 'N/A'
                if(rating == ''):
                    rating = 0
                courses_info.append([course_name, float(rating), "Edureka"])

        else:
            print(f'Error: Unable to access {search_url}. Status code: {response.status_code}')
    except ValueError:
        return NULL

    except Exception as e:
        print(f'Error: {str(e)}')
        return ""

    return courses_info

#Coursera courses
def find_courses_coursera(text):
    if not text:
        return []
    courses_info = []
    text.replace(' ', '%20')
    base_url = "https://www.coursera.org/"
    search_url = f"{base_url}search/?query={text}"
    try:
        response = requests.get(search_url) # add logging here 
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            courses = soup.find_all('div', class_='cds-ProductCard-content')
            
            for course in courses:
                course_name = course.find('h3', class_='cds-119 cds-CommonCard-title css-e7lgfl cds-121').text.strip()
                rating = course.find('p' , class_='cds-119 css-11uuo4b cds-121').text.strip()
                # reviews = course.find('p' , class_='cds-119 css-dmxkm1 cds-121').text.strip()                
                courses_info.append([course_name , rating , "coursera"])
        else:
            print("Error")
    except Exception as e:
        print(f'Error: {str(e)}') 
        return ""  

    return courses_info   
st.markdown("<h1 style='text-align: center;'>Course Recommendation</h1>", unsafe_allow_html=True)

#Input the text
user_input = "web"
user_input = st.text_input("Enter your text:")

# Display the entered text
st.write("You entered:", user_input)
logging.info("Web Site loaded")
text = user_input
logging.info("searched for " , text)

logging.info("analysis started")
#Get Edureka courses
if text :
    courses_masters_edureka = find_courses_masters(text)
    df = pd.DataFrame([i for i in courses_masters_edureka], columns=['Title','Rating','source'])

    #Get Coursera courses
    courses_coursera = find_courses_coursera(text)
    df2 = pd.DataFrame([i for i in courses_coursera],columns=['Title','Rating','source'])
    df2 = df2.drop_duplicates()

    #init the vetorizer
    vectorizer = TfidfVectorizer()

    #Applying automation to df
    lss = vectorizer.fit_transform(df['Title'])
    qq = vectorizer.transform([text])
    cosines1 = cosine_similarity(qq,lss).flatten()
    df['Score'] = [i for i in cosines1]
    df = df[df['Score'] > 0.35].drop_duplicates().reset_index(drop=True)
    df = df[df['Rating'] > 0]

    #Applying automation of df2
    ls = vectorizer.fit_transform(df2['Title'])
    q = vectorizer.transform([text])
    cosines = cosine_similarity(q,ls).flatten()
    df2['Score'] = [i for i in cosines]
    df2 = df2[df2['Score'] > 0.35]
    df2['Rating'] = df2['Rating'].astype(float)

    #Concat both DataFrames
    final_df = pd.concat([df,df2],axis=0).sort_values(by=['Rating'],ascending=False).reset_index(drop=True)
    rating = final_df.sort_values(by=['Rating'],ascending=False).reset_index(drop=True)
    print("4")
    print(final_df)
    print("4")
    #To count courses with sources respectively
    chart = final_df.groupby(['source']).count()

    #Plot the bar chart of the output
    x1 = ['Edureka','Coursera']
    y1 = np.array(chart['Title'])
    plt.bar(x1,y1)
    plt.xlabel("Sources")
    plt.ylabel("Number of Courses")
    plt.title("Recommended Courses")
    plt.show()
    # text1 = "course1 \n"
    # text2 = "course2"
    # text3 = "course3"
    # text4 = "course4"
    # text5 = "course5"
    

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("Top Courses")
            st.write("##")
            # st.dataframe(final_df)
            final = final_df
            final = final.drop(columns=['Score'])
            final = final.drop(columns=['Rating'])
            # final = final.drop(columns=['source'])
            st.dataframe(final)
        with right_column:
            st_lottie(lottie_coding, height=300, key="coding") 
        
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("Graphs analysis")


            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            data = pd.DataFrame({'Title':x1, 'Rating':y1})
            st.bar_chart(data.set_index('Title'),color='#ff0000',use_container_width=True) 
        with right_column:
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            x = rating['Title']
            y = rating['Rating']
            final_rating = pd.DataFrame({'Title':x,'Rating':y})
            st.bar_chart(final_rating.set_index('Title'),color='#ff0000',use_container_width=True)
            # st.bar_chart(final_df['Rating'])
            st.write("---")

else :
    logging.warning("Text is empty")
