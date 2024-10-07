from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


if __name__ == "__main__"
    print ("Hello Langchain")


    summary_template = """ 
        Provide me with:
        1. a short summary
        2. two interesting facts about them from the following information about a person.
        information-  {information} 
    """

    summary_prompt_template = PromptTemplate(input_variables="information", template = summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = summary_prompt_template