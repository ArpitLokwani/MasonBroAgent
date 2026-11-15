
import streamlit as st

from rag_app import (
    load_documents,
    create_chunks,
    create_embeddings,
    store_in_vector_db,
    retrieve_relevant_documents,
    generate_answer
)


# 1. Configure UI
st.set_page_config(page_title="MasonBro AI Assistant")

st.title("🏠 MasonBro AI Assistant")
st.write("Ask me anything about MasonBro services!")


# 2. Prepare RAG system only once
@st.cache_resource
def setup_rag():

    document = load_documents()

    chunks = create_chunks(document)

    embeddings = create_embeddings(chunks)

    collection = store_in_vector_db(chunks, embeddings)

    return collection


# 3. Load knowledge base
try:
    with st.spinner("Loading MasonBro knowledge base..."):
        collection = setup_rag()

except Exception as e:
    st.error(f"Could not load MasonBro knowledge base: {e}")
    st.stop()


# 4. Maintain conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# 5. Display previous conversation
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# 6. Accept user input
user_input = st.chat_input("Ask MasonBro anything...")


if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)


    # 7. Retrieve information and generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching MasonBro knowledge..."):

            try:

                relevant_documents = retrieve_relevant_documents(
                    user_input,
                    collection
                )

                ai_response = generate_answer(
                    user_input,
                    relevant_documents
                )

                st.write(ai_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response
                })

            except Exception as e:
                st.error(f"Error generating response: {e}")
