import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
'''
This script demonstrates how to build a question-answering system for a large PDF document using LangChain.
'''
# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load the PDF
pdf_path = "your_file.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)

# Step 3: Create embeddings & store in FAISS
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 4: Create retriever and QA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_key, temperature=0.7),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Step 5: Query loop
def run_chat_pdf():
    print("📚 Ask me about your PDF! (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        result = qa_chain(query)
        print(f"\nBot: {result['result']}\n")

if __name__ == "__main__":
    run_chat_pdf()