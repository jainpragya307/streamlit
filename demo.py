# Chaatbot using Streamlit

import streamlit as st
# streamlit → create the web app UI

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory

# Streamlit page  -> Configures the browser tab and shows the main heading.
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
st.title("🤖 AI Chat Assistant with Evaluation")

# LLMs
SYSTEM_PROMPT = """
You are an expert AI assistant for a FastAPI and LangChain development course.
You remember everything discussed in this conversation.
Be concise and technical. Give code examples when helpful.
"""

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_retries=3
)

eval_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_retries=3
)

# Evaluation Chain 

EVAL_SYSTEM = """
You are an expert evaluator of AI responses.
Return ONLY JSON.
"""

EVAL_HUMAN = """
Question : {question}
Response : {answer}

Score each dimension from 0.0 to 1.0.

Return ONLY this JSON:
{{
  "relevance": 0.0,
  "coherence": 0.0,
  "conciseness": 0.0,
  "feedback": ""
}}
"""

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", EVAL_SYSTEM),
    ("human", EVAL_HUMAN),
])

eval_chain = eval_prompt | eval_llm | JsonOutputParser()

# Evaluate Function 

def evaluate(question: str, answer: str) -> dict:
    try:
        scores = eval_chain.invoke({
            "question": question,
            "answer": answer
        })

        overall = round((
            scores.get("relevance", 0) +
            scores.get("coherence", 0) +
            scores.get("conciseness", 0)
        ) / 3, 2)

        scores["overall"] = overall

        return scores

    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return {}

# Streamlit Session State -> Think of it as temporary storage for the current user session.
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=10,
        return_messages=True
    )

# If this is the first time you run the file and you have no messages so create an empty chat list.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Chat Chain 
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# Show Old Messages -> It checks what is stored in the history list and displays each message one by one.
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):    # who sent it
        st.write(msg["content"])          # what was sent

# User Input
user_input = st.chat_input("Ask something...")

if user_input:

    # Shows user input what it asked for
    st.chat_message("user").write(user_input)

    # Get history from memory
    history = st.session_state.memory.load_memory_variables({})["history"]

    # Generate reply
    reply = chain.invoke({
        "input": user_input,
        "history": history
    })

    # Save to memory -> It stores both the user question and the AI answer.
    st.session_state.memory.save_context(
        {"input": user_input},
        {"output": reply}
    )

    # Evaluate
    scores = evaluate(user_input, reply)

    # Show AI response
    with st.chat_message("assistant"):

        st.write(reply)

        if scores:

            st.divider()    #Creates a horizontal line.
            st.subheader("📊 Evaluation")

            c1, c2, c3, c4 = st.columns(4)   #This makes 4 boxes side by side.

            c1.metric("Relevance", f"{scores.get('relevance', 0):.2f}")
            c2.metric("Coherence", f"{scores.get('coherence', 0):.2f}")
            c3.metric("Conciseness", f"{scores.get('conciseness', 0):.2f}")
            c4.metric("Overall", f"{scores.get('overall', 0):.2f}")

            st.info(scores.get("feedback", ""))

    # Save messages for UI
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })