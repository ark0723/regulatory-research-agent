
from dotenv import load_dotenv
from regulatory_agent.persistence import open_conversation_service

load_dotenv()


with open_conversation_service(
    storage_mode="postgres",
    provider="groq",
    model="openai/gpt-oss-120b",
) as service:
    conversation = service.create_conversation()

    run = service.start_run(
        conversation.conversation_id,
        "What must informed consent contain?",
    )