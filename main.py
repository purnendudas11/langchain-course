from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrock
from tavily import TavilyClient
# from langchain_tavily import TavilySearch 

load_dotenv()

tavily=TavilyClient()

@tool
def search(query: str) -> str:
    """ 
    A simple search tool that returns a string indicating the search results for the given query 
    from internet.
    Args:
        query (str): The search query.
    Returns:
        str: A string indicating the search results for the given query.
    """
    print(f"Search results for '{query}'")
    return tavily.search(query=query)

llm = ChatBedrock(
    model_id="amazon.nova-pro-v1:0",
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    temperature=0
)
tools = [search]
# tools = [TavilySearch()]
agent = create_agent(llm, tools)

def main():
    print("Hello from langchain-course!")
    response = agent.invoke(
        {"messages": [HumanMessage(content="show me 3 job openings for senior software engineer in Dallas, Texas and provide the links to apply for them.")]}
    )
    print(f"Agent response: {response}")


if __name__ == "__main__":
    main()
