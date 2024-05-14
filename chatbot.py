import torch
from transformers import AutoTokenizer, TextStreamer, pipeline
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.document_loaders import GCSDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import os 

def load_model(model_name):
    hf_token = "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"
    os.environ['HF_TOKEN'] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        # load_in_8bit=True,
        # load_in_4bit=True,
        device_map="auto",
        use_cache=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1000,
            do_sample=False,
            repetition_penalty=1.15,
            streamer=streamer
        )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.1})
    return llm

def load_pdf(file_name):
    loader = PyPDFLoader(file_name)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                )
    vectordb.persist()
    return vectordb

def create_chain(llm, db):
    template = """
    [INST] <>
    Act as an excercise yoga expert. Use the following information to answer the question at the end. If you do not know, just reply "I do not know" and nothing else.
    <>

    {context}

    {question} [/INST]
    Possible Answer :
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def respond(qa_chain, query):
  eval_prompt = """My Query:\n\n {} ###\n\n""".format(query)
  result = qa_chain(query)
  split_text = result['result'].split('Possible Answer :')
  possible_answer = split_text[1].strip()
  return possible_answer