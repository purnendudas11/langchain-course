from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock

load_dotenv()

def main():
    print("Hello from langchain-course!")
    llm = ChatBedrock(
        model_id="amazon.nova-pro-v1:0",
        region_name=os.getenv("AWS_DEFAULT_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        temperature=0.7
    )

    message = """
    Lionel Andrés "Leo" Messi born 24 June 1987) is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team. 
    Widely regarded as one of the greatest players in history, Messi has set numerous records for individual accolades won throughout his professional footballing career, including eight Ballons d'Or, six European Golden Shoes, and being named the world's best player by FIFA eight times. 
    In 2025, he was named the All Time Men's World Best Player by the IFFHS.
    """

    prompt_template = """
    Given the information {message} about a person, I want you to create:
    1. A short summary of the person in 2-3 sentences.
    2. A list of 5 key achievements of the person.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["message"],
        template=prompt_template
    )

    try:
        print("Sending request to Amazon Nova Pro...")
        chain=summary_prompt_template | llm
        response = chain.invoke(input={"message": message})
        print("\n--- Response ---")
        print(response.content)
    except Exception as e:
        print(f"\nInitialization failed: {e}")
        print("Ensure you have requested 'Amazon Nova Pro' model access in the AWS Bedrock Console.")

if __name__ == "__main__":
    main()
