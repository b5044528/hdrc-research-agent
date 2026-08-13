import asyncio
import json
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

BASE_DIR = Path(__file__).resolve().parent
EVALUATION_FILE = BASE_DIR / "tests" / "evaluation_cases.json"
OUTPUT_DIR = BASE_DIR / "output"

def create_model_client():
    """ This creates the connection for the AutoGen to the local LiteLLM Proxy
    LiteLLM then forwards requests to Ollama, which would run Llama 3.2 3B locally.
    :return: the client
    """
    return OpenAIChatCompletionClient(
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

def create_research_scoper(model_client):
    """
    This is used to create the Research Agent for the Research Scoper.
    :param model_client:
    :return:
    """

    return AssistantAgent(
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

def create_critical_reviewer(model_client):
    """
    This is used to create the Research Agent for the Research Reviewer.

    :param model_client:
    :return:
    """

    return AssistantAgent(
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

def load_evaluation_cases():
    """
    Load the synthetic HDRC-style evaluation question.

    :return:
    """
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def save_result(case_id, result_data):
    """

    :param case_id:
    :param result_data:
    :return:
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{case_id}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            result_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_file



async def run_case(case, model_client):
    """
    Run one question through:

    :param case:
    :param model_client:
    :return:
    """

    case_id = case["id"]
    topic = case["topic"]
    prompt = case["prompt"]

    print("\n")
    print("=" * 70)
    print(f"RUNNING {case_id}: {topic}")
    print("=" * 70)

    print("\nQUESTION:")
    print(prompt)


    # create the Agents

    research_scoper = create_research_scoper(model_client)

    critical_reviewer = create_critical_reviewer(model_client)

    # Agent 1 the Research scoper

    print("\nRunning Research Scoper...")

    scoper_result = await research_scoper.run(
        task=prompt
    )

    research_brief = scoper_result.messages[-1].content

    print("Research Scoper complete.")


    review_task = f"""
        Review the following research-scoping proposal.
        
        ORIGINAL RESEARCH PROBLEM:
        
        {prompt}
        
        RESEARCH SCOPER PROPOSAL:
        
        {research_brief}
        
        Critically review the proposal using your required review structure.
        
        """

    # Agent 2 Critical Reviewer

    print("Running Critical Reviewer...")

    review_result = await critical_reviewer.run(
        task=review_task
    )

    critical_review = review_result.messages[-1].content

    print("Critical Reviewer complete.")

    # --------------------------------------------------------
    # CREATE RESULT RECORD
    # --------------------------------------------------------

    result_data = {
        "case_id": case_id,
        "topic": topic,
        "question": prompt,
        "scoper_output": research_brief,
        "reviewer_output": critical_review,
    }

    # Save Result

    output_file = save_result(case_id, result_data)


    print(f"Saved result to: {output_file}")

    return result_data


async def main():

    print("=" * 70)
    print("HDRC RESEARCH SCOPING AI AGENT")
    print("STAGE F — AUTOMATED TEST RUN")
    print("=" * 70)

    # CHECK EVALUATION FILE EXISTS

    if not EVALUATION_FILE.exists():
        print(
            f"\nERROR: Evaluation file not found:\n"
            f"{EVALUATION_FILE}"
        )
        return


    # LOAD TEST CASES

    evaluation_cases = load_evaluation_cases()

    print(
        f"\nLoaded {len(evaluation_cases)} "
        f"evaluation cases."
    )


    # CREATE OUTPUT DIRECTORY

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # CREATE MODEL CLIENT

    model_client = create_model_client()

    completed = 0
    failed = 0

    try:
        # RUN EVERY EVALUATION CASE

        for case in evaluation_cases:
            try:
                await run_case(
                    case=case,
                    model_client=model_client,
                )

                completed += 1

            except Exception as error:

                failed += 1

                print("\n" + "!" * 70)
                print(
                    f"ERROR RUNNING "
                    f"{case.get('id', 'unknown case')}"
                )
                print("!" * 70)

                print(
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                print(
                    "\nContinuing to the next case..."
                )

    finally:

        # CLOSE MODEL CLIENT

        await model_client.close()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STAGE F COMPLETE")
    print("=" * 70)

    print(
        f"\nSuccessful cases: {completed}"
    )

    print(
        f"Failed cases: {failed}"
    )

    print(
        f"Outputs directory: {OUTPUT_DIR}"
    )

    print("=" * 70)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
