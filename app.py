import streamlit as st
from PIL import Image
from transformers import pipeline
import logging
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def process_file(file, process_type):
    # Read TXT file
    text = file.read().decode("utf-8")
    
    if process_type == 'NLP Summarization':
        logging.info("Running NLP Summarization for file: %s", file.name)
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        # Truncate or summarize the text if it exceeds the model's maximum length
        max_length = summarizer.tokenizer.model_max_length
        if len(text) > max_length:
            text = text[:max_length-512]  # Truncate the text
        
        # Generate summary
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        # Return the generated summary as bytes
        return summary.encode("utf-8")
        
    elif process_type == 'Topic Modelling':
        logging.info("Running NLP Process for file: %s", file.name)
        # Perform any desired operations on the text
        # ...

        # Define the Streamlit app
        st.title("NLP Topic Modeling")
            
            # User input for documents
        documents = st.text_area("Enter your documents (one document per line):")
        if st.button("Run Topic Modeling"):
            processed_docs = preprocess_lda(documents)
            dictionary, corpus = create_corpus(processed_docs)
            lda_model = train_lda_model(corpus, dictionary)
            display_topics(lda_model)
        
        # Preprocess the documents
        def preprocess_lda(documents):
            processed_docs = []
            for doc in documents.splitlines():
                processed_docs.append(simple_preprocess(doc, deacc=True))
            return processed_docs
        
        # Create a dictionary and corpus
        def create_corpus(processed_docs):
            dictionary = corpora.Dictionary(processed_docs)
            corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
            return dictionary, corpus
        
        # Train the LDA model
        def train_lda_model(corpus, dictionary):
            lda_model = LdaModel(corpus=corpus,
                                 id2word=dictionary,
                                 num_topics=2,
                                 random_state=42,
                                 passes=10)
            return lda_model
        
        # Display the topics
        def display_topics(lda_model):
            for idx, topic in lda_model.print_topics():
                st.write(f"Topic {idx + 1}: {topic}")
        
        
        
        # Return the processed data as bytes
        return text.encode("utf-8")


def original(file):
    # Read TXT file
    text = file.read().decode("utf-8")
    text = text[:30]
    return text

# Streamlit app
def main():
    st.title("File Upload and Processing")
    
    # Sidebar - Logo
    logo_path = "logo.png"  # Path to your logo image file
    st.sidebar.image(Image.open(logo_path), use_column_width=True,)
    
    st.header("Select the process")
    process_type = st.selectbox("Select the process", ['Topic Modelling', 'NLP Summarization'])


    # File upload section
    st.header("Upload a file")
    file = st.file_uploader("Choose a file", type=["txt"], help="Please upload a TXT file.")
    
    if file is not None:
        try:
            with st.spinner("Processing the file..."):
                processed_data = process_file(file, process_type)
            
            # Display original text and processed data side by side with adjustable column width
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Text")
                org = original(file)
                st.write(org)
                st.text_area("", org, height=300)

            with col2:
                st.subheader("Processed Data")
                st.write(processed_data.decode("utf-8"))
                st.text_area("", org, height=300)            

            
            # Download button
            st.subheader("Download Processed Data")
            st.download_button("Download Text File", data=processed_data, file_name="processed_data.txt")
            
            # Log user selections and file information
            logging.info("User selected process: %s, File uploaded: %s, Download file name: %s", process_type, file.name, "processed_data.txt")
            
        except Exception as e:
            logging.error(f"Error occurred for file: %s, Error: {str(e)}", file.name)  # Log the error message
            st.error("An error occurred during processing. Please try again.")
        
if __name__ == "__main__":
    main()
