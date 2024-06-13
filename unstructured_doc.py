# Check if the file exists
import os
file_path = "/Users/manikandanjagannathan/Developer/chat-app/lang-chain/medical_anotamy.pdf"
print("File exists:", os.path.exists(file_path))

# Try loading the document
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    
    loader = UnstructuredPDFLoader(file_path)
    print("Loader initialized successfully")

    data = loader.load()
    print("Data loaded successfully:", data)
except Exception as e:
    print(f"An error occurred: {e}")

# from langchain_community.document_loaders import OnlinePDFLoader
# from dotenv import load_dotenv
# import unstructured.partition.utils.ocr_models as ocr_models
# print(dir(ocr_models))

# load_dotenv()

# loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
# data = loader.load()

# from langchain_community.document_loaders import PDFMinerLoader

# file_path = (
#     "/Users/manikandanjagannathan/Developer/chat-app/lang-chain/medical_anotamy.pdf"
# )
# loader = PDFMinerLoader(file_path)
# data = loader.load()
# print(data)
