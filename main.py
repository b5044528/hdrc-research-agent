import asyncio

from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from huggingface_hub.hf_api import api


async def main():
    client = OpenAIChatCompletionClient(
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

    response = await client.create(
        [
            UserMessage(
                content="Hello. In one sentence, tell me what you can help with.",
                source="user"
            )
        ]
    )
    print(response)

    await client.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
