from langchain_google_community import DocAIParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.chains.retrieval_qa import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Blob
from langchain.chains.question_answering import load_qa_chain

# Replace these with your actual configurations
PROCESSOR_NAME = "document_parsing"
GCS_OUTPUT_PATH = "gs://your-bucket/output-path"

# Load and parse PDF documents using DocAIParser
parser = DocAIParser(
    location="us", processor_name=PROCESSOR_NAME, gcs_output_path=GCS_OUTPUT_PATH
)

# Assuming Blob is a class that correctly loads the document
# If Blob is not defined in your imports, you may need to use another loader
blob = Blob(path="/path/to/medical_anatomy.pdf")
docs = list(parser.lazy_parse(blob))

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)

# Create a vector store and store the document splits
vectorstore = FAISS.from_documents(all_splits, OpenAIEmbeddings())

# Create a Retrieval Augmented Generation (RAG) chain with the chat model
chat = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens=500)
# qa_chain = RetrievalQAWithSourcesChain.from_llm(chat, vectorstore)
chain = load_qa_chain(chat, chain_type='stuff')

# Interactive loop for handling user input and generating responses
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the assistant.")
            break
        
        # Use the RAG chain to get the answer and sources
        # result = qa_chain({"question": user_input})
        result = chain.run(input_documents=vectorstore, question=user_input)
        print("Assistant:", result)
        # print("Sources:")
        # for source in result["sources"]:
            # Displaying the first 100 characters of the source content for brevity
            # print(source.page_content[:100] + "...")
    except Exception as e:
        print(f"An error occurred: {e}")
