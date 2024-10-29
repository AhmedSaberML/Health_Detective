
# Import necessary libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set up embeddings and vector store
persist_directory = r"embedd_vecdb_bge-m3"
model_path="BAAI/bge-m3"
embeddings= HuggingFaceEmbeddings(model_name=model_path)
Load the FAISS vector store
vector_db = FAISS.load_local(persist_directory, embeddings,allow_dangerous_deserialization=True)
print("Database created and persisted successfully.")

# Load the LLM model for Qwen
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define the function for LLM-based answer generation
def qwen_llm(top_docs, question):
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": f"""
        Please provide a medical prediction based on the information given.
        Context: '''{top_docs}'''
        Question: '''{question}'''
        """}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response

def get_answer(question):
    top_docs = vector_db.similarity_search(question, k=1)
    response = qwen_llm(top_docs, question)
    return response