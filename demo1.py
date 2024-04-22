import streamlit as st
import ollama
import requests
import smtplib
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def train_spam_classifier():
    # Sample data - Replace this with your dataset
    spam_data = [
        ("Claim your prize now!", 1),  # 1 indicates spam
        ("Hi, how are you?", 0),       # 0 indicates not spam
        ("Congratulations, you've won a lottery!",1),
        ("Meeting at 3 PM tomorrow", 0),
        ("Amount UPI ref",0),
        # Add more examples as needed
    ]
    messages, labels = zip(*spam_data)
    processed_messages = [preprocess_text(message) for message in messages]
    X_train, _, y_train, _ = train_test_split(processed_messages, labels, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)
    return classifier, vectorizer


def send_email():
    email = "mr.habith78"
    receiver = st.text_input("To:", "")
    subject = st.text_input("Subject:", "")
    message = st.text_area("Body:", "")

    if st.button("Send Email"):
        email_sender(email, receiver, subject, message)


def email_sender(email, receiver, subject, message):
    email_1 = email + "@gmail.com"
    receiver_1 = receiver
    text = f"Subject: {subject}\n\n{message}"

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(email, "nzgbtfbvqwhenfce")
    server.sendmail(email_1, receiver_1, text)
    time.sleep(2)

    st.success(f"Email has been sent successfully to {receiver_1}!")


def search_image_api(search_query, num_images=10):
    Api_key = "AIzaSyAhf8v05UVQ1n-CB0-QHhe8bkHpVbQASD4"  # Replace 'YOUR_API_KEY' with your actual API key
    search_engine_id = "a13d88ad64b7d44cb"  # Replace 'YOUR_SEARCH_ENGINE_ID' with your actual search engine ID
    s_type = 'image'

    url = 'https://www.googleapis.com/customsearch/v1'

    params = {
        'q': search_query,
        'key': Api_key,
        'cx': search_engine_id,
        'searchType': s_type,
        'num': num_images  # Specify the number of images to retrieve
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for any HTTP error
        results = response.json()

        if 'items' in results and results['items']:
            # Display multiple image results
            images = [item['link'] for item in results['items']]
            num_columns = 3  # Number of images per row
            num_rows = -(-len(images) // num_columns)  # Ceiling division to determine number of rows

            for i in range(num_rows):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    index = i * num_columns + j
                    if index < len(images):
                        cols[j].image(images[index], caption="Image Result", width=200)
        else:
            st.warning("No images found for the given query.")
    except requests.RequestException as e:
        st.error(f"Error occurred during image search: {e}")


def main():
    st.title("IGPT Assistant")
    st.sidebar.header("MENU")
    selected_page = st.sidebar.radio("Go to", ["Home", "Chat"])

    if selected_page == "Home":
        st.write("Prompts To Perform Task\n1. Generate email : Send an email\n2. Generate Image : Search for an image\n3. Ask Questions : IGPT Assistant\n4. Spam Analysis : spam message detection")

    elif selected_page == "Chat":
        user_message = st.text_input("User:")
        user_message_1 = user_message
        color="#FF5733"

        if any(keyword in user_message_1.lower() for keyword in [ "your founder"," your company owner","your ceo"]):
            st.spinner("Processing...")
            time.sleep(3)
            st.title("MR.HABITH")
            st.image('habith.jpg',caption='Founder of Integrated generative pre-trained Transformer')
            st.subheader("Mr. Habith is a student specializing in B.Tech Artificial Intelligence and Data Science  He has interests in AI & ML, as seen on Hugging Face and has been involved in mobile app development workshops, as shared on LinkedIn")
            st.subheader("Anna University B.tech artificial intelligence and data science  Artificial Intelligence Oct 2021 - Dec 2025 Activities and societies: district level chess“A year spent in artificial intelligence is enough to make one believe in God.” “There is no reason and no way that a human mind can keep up with an artificial intelligence machine by 2035.” “Is artificial intelligence less than our intelligence")
            st.subheader("Python (Programming Language)  Exploratory Data Analysis  Full-Stack Development  Machine Learning  Deep Learning  Deep Reinforcement Learning  Data Analysis  Analytical Skills  Business Analysis  Problem Solving  Communication  Marketing  English,Large language model developer then lanchaing technology models")

        ####emails protocols
        elif any(keyword in user_message_1.lower() for keyword in ["send a email","generate a email","email","send email"]):
            st.title("E-mail Assistant")
            send_email()
        ###spam classifier using ML
        elif "spam" in user_message_1:
            message = user_message_1
            processed_message = preprocess_text(message)

            classifier, vectorizer = train_spam_classifier()
            st.title("Spam Analysis")

            processed_input = vectorizer.transform([processed_message])
            prediction = classifier.predict(processed_input)[0]

            st.markdown(f"## Prediction:")
            if prediction == 1:
                st.error("This message will be spam!")
            else:
                st.success("This message is not spam.")


        elif  len(user_message_1)>=1:
            response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': user_message_1}])
            assistant_response = response["message"]["content"]
            if any(keyword in user_message_1.lower() for keyword in ["code","write","program"]):
                text=assistant_response.strip("\\n").strip()
                st.code(text)
                st.title("Related Images")
                st.spinner("Processing...")
                search_query = text
                search_image_api(search_query)

            else:
                response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': user_message_1}])
                t1=assistant_response.strip("\\n").strip()
                st.text_area("IGPT:",value=t1,height=200,max_chars=None,key=None)
                st.title("Related Images")
                search_query = user_message_1
                search_image_api(search_query)


if __name__ == "__main__":
    main()
