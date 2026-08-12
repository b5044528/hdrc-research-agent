import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():

    # ---------------------------------------
    # 1. This is for connecting the AutoGen to the Local Llama model
    # ---------------------------------------

    model_client = OpenAIChatCompletionClient(
        model="llama3.2:3b",
        api_key="not-needed",
        base_url="http://localhost:4000",
        model_info={
            "vision":False,
            "function_calling":False,
            "json_output":False,
            "family": "unknown",
            "structured_output":False
        },
    )

    # ---------------------------------------
    # 2. Creating the Research Scoper Agent
    # ---------------------------------------

    research_scoper = AssistantAgent(
        name="research_scoper",
        model_client=model_client,
        system_message="""
        
        You are a research scoping assistant supporting a local authority.
        
        Given a service or public health problem:
        
        1. Restate the problem Clearly.
        2. Formulate 1-3 research questions.
        3. Identify the population or groups affected.
        4. Identify potentially useful data or variables.
        5. Suggest Proportionate analytical methods.
        6. Explain how the findings may support a decision.
        7. State assumptions and missing information.
        
        Do not invent evidence.
        Do not claim that a dataset exists unless it has been provided.
        Clearly distinguish suggestions form established facts.
        
        Prefer the simplest analytical method capable of answering the research question.

        Do not recommend machine learning unless there is a clear reason 
        that simpler descriptive or statistical methods would be insufficient.
        
        Never state that a specific dataset is available unless the user has
        provided evidence that it exists.

        When suggesting data, use language such as "potentially useful data
        could include..."
        
        Always use exactly these headings:
        
        ## Problem
        ## Research Questions
        ## Population or Groups
        ## Potentially useful data or variables
        ## Suggestions
        ## Explain how the findings may support a decision.
        ## State assumptions and missing information.
        """
    )
    # ---------------------------------------
    # 3. Give the research agent a research problem
    # ---------------------------------------

    question = """
    How could a council investigate inequalities in fuel poverty?
    """

    # 4. Run the Agent

    result = await research_scoper.run(
        task=question
    )
    # ---------------------------------------
    # 5. extract the final Response
    # ---------------------------------------

    final_response = result.messages[-1].content


    # ---------------------------------------
    # 6. Display response
    # ---------------------------------------

    print("\n" + "=" * 70)
    print("HDRC RESEARCH SCOPER")
    print("=" * 70)

    print(f"\nQUESTION:\n{question.strip()}")

    print("\n" + "-" * 70)
    print("\nRESEARCH BRIEF:\n")

    print(final_response)

    print("\n" + "=" * 70)

    # ---------------------------------------
    # 7. Close model connection
    # ---------------------------------------

    await model_client.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
