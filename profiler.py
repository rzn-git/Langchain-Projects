import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import openai 

openai.api_key = os.getenv('OPENAI_API_KEY')

information = """
    Elon Reeve Musk  has expressed views that have made him a polarizing figure.[5] He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, promoting right-wing conspiracy theories, and "endorsing an antisemitic theory"; he later apologized for the last of these.[6][5][7] His ownership of Twitter has been controversial because of the layoffs of large numbers of employees, an increase in hate speech, misinformation and disinformation posts on the website, and changes to Twitter Blue verification. 
    """

if __name__ == "__main__":
    print ("Hello Langchain")


    summary_template = """ 
        Provide me with:
        1. a short summary
        2. two interesting facts about them from the following information about a person.
        information-  {information} 
    """

    summary_prompt_template = PromptTemplate(input_variables=   ["information"], template = summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = summary_prompt_template | llm

    response = chain.invoke(input = {"information": information})

    print (response)