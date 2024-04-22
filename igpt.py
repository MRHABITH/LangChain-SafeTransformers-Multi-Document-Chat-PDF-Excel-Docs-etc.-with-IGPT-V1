#import streamlit as st
import openai
import requests
import smtplib
import time
#import streamlit as st
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




#import pickle
#from PyPDF2 import PdfReader
#from streamlit_extras.add_vertical_space import add_vertical_space
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
#from langchain.llms import OpenAI
#from langchain.chains.question_answering import load_qa_chain
#from langchain.callbacks import get_openai_callback
import os

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


# Set up OpenAI API key
API_KEY = "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"
openai.api_key = API_KEY
chat_log = []






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




import requests
import streamlit as st

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

#search_query = st.text_input("Enter an image search query:")
#search_image_api(search_query)

def generate_images(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=5,
        size='256x256',
        response_format='url'
    )
    return [image_data["url"] for image_data in response["data"]]




def main():
    st.title("IGPT Assistant")
    st.sidebar.header("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Home", "Enigma"])

    if selected_page == "Home":
        st.write("Prompts To Perform Task\n1. Generate email : Send an email\n2. Generate Image : Search for an image\n3. Ask Questions : IGPT Assistant\n4. Spam Analysis : spam message detection")

    elif selected_page == "Enigma":
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

        ###openai image generator
        elif any(keyword in user_message_1.lower() for keyword in ["photo", "image", "generate image","generate a image","generate a images","generate images","photos","images","logo"]):
            if user_message_1.strip():  # Check if user input is not empty
                prompt = user_message_1
                image_urls = generate_images(prompt)
                num_columns = 3
                # Calculate the number of rows required
                num_rows = (len(image_urls) + num_columns - 1) // num_columns

                # Display images row by row
                for i in range(num_rows):
                    row = st.container()
                    with row:
                        for j in range(num_columns):
                            index = i * num_columns + j
                            if index < len(image_urls):
                                st.image(image_urls[index], width=256)


            else:
                st.error("Please provide a valid prompt to generate images.")



         ###pdf lang chain technology
        elif any(keyword in user_message_1.lower() for keyword in ["dobut in my pdf","i have dobut in my pdf","chat with my pdf"]):

            load_dotenv()
            def initialize_session_state():
                if 'history' not in st.session_state:
                    st.session_state['history'] = []

                if 'generated' not in st.session_state:
                    st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

                if 'past' not in st.session_state:
                    st.session_state['past'] = ["Hey! 👋"]

            def conversation_chat(query, chain, history):
                result = chain({"question": query, "chat_history": history})
                history.append((query, result["answer"]))
                return result["answer"]

            def display_chat_history(chain):
                reply_container = st.container()
                container = st.container()

                with container:
                    with st.form(key='my_form', clear_on_submit=True):
                        user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
                        submit_button = st.form_submit_button(label='Send')

                    if submit_button and user_input:
                        with st.spinner('Generating response...'):
                            output = conversation_chat(user_input, chain, st.session_state['history'])

            #st.text_area("IGPT:",value=output,height=100,max_chars=None,key=None)

                        st.session_state['past'].append(user_input)
                        st.session_state['generated'].append(output)

                if st.session_state['generated']:
                    with reply_container:
                        for i in range(len(st.session_state['generated'])):
                            st.write(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                            st.write(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

            def create_conversational_chain(vector_store):
                load_dotenv()
    # Create llm
    #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        #streaming=True, 
                        #callbacks=[StreamingStdOutCallbackHandler()],
                        #model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
                llm = Replicate(
                    streaming = True,
                    model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
                    callbacks=[StreamingStdOutCallbackHandler()],
                input = {"temperature": 0.01, "max_length" :500,"top_p":1})
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

                chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
                return chain

            def main():
                load_dotenv()
    # Initialize session state
                initialize_session_state()
                st.title("Multi-Docs ChatBot using llama2 :books:")
    # Initialize Streamlitxs
    #st.sidebar.title("Document Processing")
                uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
                if uploaded_files:
                    text = []
                    for file in uploaded_files:
                        file_extension = os.path.splitext(file.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_file.write(file.read())
                            temp_file_path = temp_file.name

                            loader = None
                            if file_extension == ".pdf":
                                loader = PyPDFLoader(temp_file_path)
                            elif file_extension == ".docx" or file_extension == ".doc":
                                loader = Docx2txtLoader(temp_file_path)
                            elif file_extension == ".txt":
                                loader = TextLoader(temp_file_path)
                            elif file_extension==".ppt":
                                loader = UnstructuredPowerPointLoader(temp_file_path)
                            if loader:
                                text.extend(loader.load())
                                os.remove(temp_file_path)
                    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
                    text_chunks = text_splitter.split_documents(text)
        # Create embeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})
        # Create vector store
                    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        # Create the chain object
                    chain = create_conversational_chain(vector_store)
                    display_chat_history(chain)
            if __name__ == "__main__":
                main()

        elif  len(user_message_1)>=1:
            chat_log.append({"role": "user", "content": user_message})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=chat_log
            )
            assistant_response = response["choices"][0]["message"]["content"]
            if any(keyword in user_message_1.lower() for keyword in ["code","write","program"]):
                text = assistant_response.strip("\\n").strip()
                st.code(text)
                st.title("Related Images")
                st.spinner("Processing...")
                time.sleep(5)
                search_query = text
                search_image_api(search_query)

            else:
                t1=assistant_response.strip("\\n").strip()
                st.text_area("IGPT:",value=t1,height=200,max_chars=None,key=None)
                st.title("Related Images")
                search_query = user_message_1
                search_image_api(search_query)


if __name__ == "__main__":
    main()
