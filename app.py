import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

async def run_memory_chat():
    """Run a chat using MCPAgent's built-in conversation memory."""
    # Load environment variables
    load_dotenv()
    os.environ["gROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    #config file path change this to your config file path
    config_file = "leavemanagementserver.json"

    print("Initializing MCPClient...")

    # Create MCPClient from config file
    client = MCPClient.from_config_file(config_file)

    print("MCPClient initialized.")

    llm = ChatGroq(model="llama3-70b-8192", temperature=0.1)
    # Create agent with the client
    agent = MCPAgent(
        llm=llm, 
        client=client, 
        max_steps=30, 
        memory_enabled=True, # Enable memory
        verbose=True, # Enable debug mode
        )

    print("\n ===== Interactive MCP Chat ======\n")
    print("Type 'exit' to quit the chat.\n")
    print("Type 'clear' to clear the memory.\n")
    print("=========================\n")

    try:
        while True:
            # Get user input
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "clear":
                agent.memory.clear()
                print("Memory cleared.")
                continue

            # Run the query
            result = await agent.run(user_input, max_steps=30)
            print(f"\nResult: {result}")
    except KeyboardInterrupt:
        print("\nExiting chat...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await client.close_all_sessions()
        print("Client closed.")

    # # Run the query
    # result = await agent.run(
    #     "Find the best restaurant in San Francisco USING GOOGLE SEARCH",
    #     max_steps=30,
    # )
    # print(f"\nResult: {result}")


async def main():
    """Run the example using a configuration file."""
    # Load environment variables
    load_dotenv()

    config = {
        "mcpServers": {
            "http": {
                "url": "http://localhost:8931/sse"
            }
        }
    }

    # Create MCPClient from config file
    client = MCPClient.from_dict(config)

    # Create LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    result = await agent.run(
        "Find the best restaurant in San Francisco USING GOOGLE SEARCH",
        max_steps=30,
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    # Run the appropriate example
    asyncio.run(run_memory_chat())