import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import openai
from langchain_ollama import ChatOllama

# Set your API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

information = """
    List of country - capital
    France - Paris
    Germany - Berlin
    Bangladesh - Dhaka
    South Africa - Pretoria
    """

if __name__ == "__main__":

    # Modify the template to ask a specific question
    summary_template = """ 
        From the following information, what is the capital of Bangladesh? {information}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Define the language model
    #llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm = ChatOllama (model="llama3.1" )

    # Create the chain (pipe the prompt template into the language model)
    chain = summary_prompt_template | llm 

    # Invoke the chain with input
    response = chain.invoke(input={"information": information})

    # Print only the content
    print(response.content)
