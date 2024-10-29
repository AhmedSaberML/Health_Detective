import streamlit as st
import time
import uuid
from code import get_answer

def main():
    st.title("Predict Medical Outcome")

    # Session state initialization
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())

    if "count" not in st.session_state:
        st.session_state.count = 0



    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Search"):
        with st.spinner("Processing..."):
            answer_data = get_answer(user_input)
            st.success("Completed!")
            st.write(answer_data)


            # Generate a new conversation ID for next question
            st.session_state.conversation_id = str(uuid.uuid4())


if __name__ == "__main__":
    main()