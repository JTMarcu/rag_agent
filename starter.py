import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

"""
This script implements a chatbot using LangChain, OpenAI's ChatGPT model, and a conversation memory buffer.

Key components:
1. **Environment Setup**:
   - Loads the OpenAI API key from a `.env` file using the `dotenv` library.

2. **Language Model Initialization**:
   - Initializes the `ChatOpenAI` model with the API key and a temperature setting for response variability.

3. **Conversation Memory**:
   - Uses `ConversationBufferMemory` to store the entire conversation history, allowing the chatbot to maintain context across interactions.

4. **Conversation Chain**:
   - Combines the language model and memory into a `ConversationChain` to handle user inputs and generate responses.

5. **Chatbot Loop**:
   - Runs an interactive chatbot in a loop, where users can type messages and receive responses.
   - The loop exits when the user types "exit".

Usage:
- Run the script, and interact with the chatbot in the terminal.
- Ensure a `.env` file with the `OPENAI_API_KEY` is present in the working directory.
"""

# Load API key from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=openai_key, temperature=0.7)

# Create a memory to hold the conversation
memory = ConversationBufferMemory()

# Create the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Run the chatbot in a loop
def run_chat():
    print("🤖 LangChain Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = conversation.run(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    run_chat()
