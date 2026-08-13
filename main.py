import asyncio
import json

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src import reviewer_agent


async def main():

    # ---------------------------------------
    # Model client
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
    # Agent 1 Research Scoper Agent
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
    # Agent 2 Reviewer Agent
    # ---------------------------------------

    critical_reviewer = AssistantAgent(
        name="critical_reviewer",
        model_client=model_client,
        system_message="""
        You are a critical research-methods and responsible-AI reviewer
        supporting a local authority.
        
        You will receive a proposed research-scoping brief.
        
        Critically review it and identify:
        
        1. Unsupported assumptions.
        2. Weak or inappropriate analytical claims.
        3. Possible correlation-versus-causation problems.
        4. Potential bias or unfairness.
        5. Missing data-quality considerations.
        6. Privacy or data-governance considerations.
        7. Missing stakeholder or community perspectives.
        8. Methodological limitations.
        9. Missing evidence or information.
        10. Specific improvements.
        
        Do not simply agree with the proposal.
        
        Do not invent evidence or claim that a dataset, policy or programme
        exists unless it has been supplied.
        
        Always use exactly these headings:
        
        ## Overall Assessment
        ## Unsupported Assumptions
        ## Methodological Issues
        ## Data Quality Issues
        ## Bias / Fairness
        ## Privacy / Governance
        ## Stakeholder Considerations
        ## Limitations
        ## Recommended Improvements
    """
    )

    # ---------------------------------------
    # 3. Give the research agent a research problem
    # ---------------------------------------

    question = """
    How could a council investigate inequalities in fuel poverty?
    """

    # 4. Run the Agent 1

    scoper_result = await research_scoper.run(
        task=question
    )

    research_brief = scoper_result.messages[-1].content

    # 4.5 Run the Agent 2

    review_task = f"""
    Review the following research-scoping proposal.
    
    Original Research problem:
    
    {question}
    
    Research Scoper Proposal:
    
    {research_brief}
    
    Critically Review this proposal.
"""

    review_result = await critical_reviewer.run(
        task=review_task
    )

    # ---------------------------------------
    # 5. extract the final Response
    # ---------------------------------------


    critical_review = review_result.messages[-1].content

    # ---------------------------------------
    # 6. Display response
    # ---------------------------------------


    # Display results
    print("\n" + "=" * 70)
    print("ORIGINAL QUESTION")
    print("=" * 70)
    print(question.strip())

    print("\n" + "=" * 70)
    print("RESEARCH SCOPER")
    print("=" * 70)
    print(research_brief)

    print("\n" + "=" * 70)
    print("CRITICAL REVIEWER")
    print("=" * 70)
    print(critical_review)

    await model_client.close()



    # ---------------------------------------
    # 7. Close model connection
    # ---------------------------------------

    await model_client.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    with open("tests/evaluation_cases.json", "r") as file:
        evaluation_cases = json.load(file)

    for case in evaluation_cases:
        print(case["id"], "-", case["topic"])
        print(case["prompt"])
        print()

    #asyncio.run(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
