# agent_module.py

from typing_extensions import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import streamlit as st

# from dotenv import load_dotenv
from langchain_aws import ChatBedrock   
import os

# load_dotenv()

# Load AWS credentials from secrets
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
region_name = st.secrets["aws"]["region_name"]
tavily_api_key = st.secrets["TAVILY"]["API_KEY"]

# Set environment variables for boto3 to use
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
os.environ["AWS_REGION"] = region_name  # VERY IMPORTANT
os.environ["TAVILY_API_KEY"] = tavily_api_key


llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    temperature=0,
    max_tokens=None,
    region="us-east-1"
    # other params...
)


class State(TypedDict):
    messages: Annotated[List, add_messages]

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain import hub

# Load documents and build vectorstore
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

documents = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(docs)

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector = FAISS.from_documents(documents, embedding_model)
retriever = vector.as_retriever()

# Tools
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
search = TavilySearchResults(max_results=2,tavily_api_key=st.secrets["TAVILY_API_KEY"])
tools = [search, retriever_tool]

# Bind tools to LLM (define llm if not done already)
from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4")  # or gpt-3.5-turbo

model_with_tools = llm.bind_tools(tools)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, verbose=True)

# Store for chat sessions
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def wrap_as_ai_message(output):
    if isinstance(output, str):
        return AIMessage(content=output)
    elif isinstance(output, dict) and "text" in output:
        return AIMessage(content=output["text"])
    elif isinstance(output, dict) and "output" in output:
        return AIMessage(content=output["output"])
    return AIMessage(content=str(output))

formatted_executor = agent_executor | RunnableLambda(wrap_as_ai_message)

agent_with_chat_history = RunnableWithMessageHistory(
    formatted_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
