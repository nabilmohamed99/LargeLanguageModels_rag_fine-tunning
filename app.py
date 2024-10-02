from langchain_community.llms import Ollama
import streamlit as st

# Initialize the LLM
llm = Ollama(model="phi:latest", base_url="http://ollama-container:11434", verbose=True)

def sendPrompt(prompt):
    global llm
    response = llm.invoke(prompt)
    return response

# Streamlit app layout
st.title("Chat with Ollama")

# Define available models
available_models = ["phi:latest", "model_2", "model_3"]  # Add your actual model names here

# Create an editable selectbox
model_choice = st.selectbox("Select or Enter a Language Model", available_models)

# Allow users to add a custom model
custom_model = st.text_input("Or enter a custom model (leave empty for default):")

# Use the custom model if provided
if custom_model:
    model_choice = custom_model

# Update the LLM based on the selection
llm = Ollama(model=model_choice, base_url="http://ollama-container:11434", verbose=True)

# Initialize chat messages
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question!"}
    ]

# Capture user input
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = sendPrompt(prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
