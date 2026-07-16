import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print ("Hello from langchain-course!")
print ('initializing...')

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536
)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0
)

vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based on the following context:
    {context}
    Question: {question}
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """ Format retrieved documents into a single string """
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    #step 1 : retrieve relevent documents
    docs=retriever.invoke(query)
    
    #step 2: format documents into context string
    context=format_docs(docs)

    #step 3: format the prompt with question and context
    messages = prompt_template.format_messages(context=context,question=query)

    #step 4: invoke LLM with formated message
    response= llm.invoke(messages)

    #step 5: return the content
    return response.content

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter('question') | retriever | format_docs
        )
        | prompt_template 
        | llm 
        | StrOutputParser()
    )
    return retrieval_chain

if __name__ == "__main__":
    print ('retrieving ...')

    #Query
    query = "What is Pinecone in machine learning?"
    #==============================================
    # Option 0: Raw Invocation with LLM (NO RAG)
    #==============================================
    print ("\n" + "=" * 70)
    print ("IMPLEMTENTATION 0: Raw Invocation with LLM (NO RAG)\n")
    print ("=" * 70)
    raw_response = llm.invoke([HumanMessage(content=query)])
    print ('\nAnswer:')
    print (raw_response.content)
    #==============================================
    # Option 1: Use implementation WITHOUT LCEL
    #==============================================
    print ("\n" + "=" * 70)
    print ("IMPLEMTENTATION 1: With out langchain expression language")
    print ("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print ('\nAnswer:')
    print (result_without_lcel)
    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)