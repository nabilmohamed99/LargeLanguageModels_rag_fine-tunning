import streamlit as st
from utils.electrical_agent_workflow import app

st.title("Chat avec l'Agent Power Studio SCADA")

# Initialisation des messages de chat dans la session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Posez-moi une question concernant Power Studio SCADA ou la qualité du réseau électrique."}
    ]

def send_prompt(prompt):
    try:
        response = app.invoke({"question": prompt})
        assistant_reply = response.get("llm_output", "Je suis désolé, je n'ai pas pu comprendre la question.")
    except Exception as e:
        assistant_reply = f"Erreur : {e}"
    return assistant_reply

if prompt := st.chat_input("Votre question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Affichage des messages de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit..."):
            response = send_prompt(st.session_state.messages[-1]["content"])
            st.write(response)
            # Ajouter la réponse de l'assistant à l'historique
            st.session_state.messages.append({"role": "assistant", "content": response})
