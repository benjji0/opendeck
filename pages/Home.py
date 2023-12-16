import streamlit as st
import pandas as pd
import numpy as np
import time

# st.title('OpenDeck 1.0')
# txt = st.text_area(
#     "Text to analyze",
#     "It was the best of times, it was the worst of times, it was the age of "
#     ,
#     height=420
#     )

# st.write(f'You wrote {len(txt)} characters.')

# import spacy

# nlp = spacy.load("en_core_web_sm")
# doc = nlp(txt)

# for token in doc:
#     st.write(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)

# import spacy
# from spacytextblob.spacytextblob import SpacyTextBlob

# nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe('spacytextblob')
# text = 'I had a really horrible day. It was the worst day ever! But every now and then I have a really good day that makes me happy.'
# doc = nlp(txt)
# st.write(doc._.blob.polarity)                         # Polarity: -0.125
# st.write(doc._.blob.subjectivity)                        # Subjectivity: 0.9
# st.json(doc._.blob.sentiment_assessments.assessments)  # Assessments: [(['really', 'horrible'], -1.0, 1.0, None), (['worst', '!'], -1.0, 1.0, None), (['really', 'good'], 0.7, 0.6000000000000001, None), (['happy'], 0.8, 1.0, None)]
# doc._.blob.ngrams() 


import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from htmlTemplates import css, bot_template, user_template
from streamlit_chat import message
from streamlit.components.v1 import html

os.environ['GOOGLE_API_KEY'] = os.environ.get("PALM_API_KEY")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm=GooglePalm()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            # st.write("You: ", message.content)
            with st.chat_message("human"):
                st.write(message.content)
            # st.write(user_template.replace(
            #     "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # st.write("OpenDeck: ", message.content)
            # st.write(f'OpenDeck wrote {len(message.content)} characters.')
            with st.chat_message("assistant"):
                st.write(message.content)

def main():
    
    st.set_page_config("OpenDeck 1.11")
    st.write(css, unsafe_allow_html=True)
    # st.write(user_template.replace("{{MSG}}", "message.content"), unsafe_allow_html=True)
    # with st.chat_message("user"):
    #     st.write(user_template.replace("{{MSG}}", "message.content"), unsafe_allow_html=True)
    st.header("OpenDeck 2.11 💬")
    user_question = st.text_input("Ask a Question from the PDF Files")
    st.write(f'You wrote {len(user_question.split())} words.')
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True, key=6)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")
    


if __name__ == "__main__":
    main()