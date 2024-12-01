import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate


class ChatLLM:
    def __init__(self):
        self._model = ChatOllama(model="gemma2:2b", temperature=3)
        self._template = """주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요.

Question: {question}
"""
        self._prompt = ChatPromptTemplate.from_template(self._template)
        self._chain = (
            {"question": RunnablePassthrough()} |
            self._prompt |
            self._model |
            (lambda x: x) 
        )

    def invoke(self, user_input):
        return self._chain.invoke(user_input)


class ChatWeb:
    def __init__(self, llm, page_title="Gazzi Chatbot", page_icon=":books:"):
        self._llm = llm
        self._page_title = page_title
        self._page_icon = page_icon

    def run(self):
        st.set_page_config(page_title=self._page_title, page_icon=self._page_icon)
        st.title(self._page_title)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_input := st.chat_input("질문을 입력해 주세요."):
            st.chat_message("user").write(user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})

            response = self._llm.invoke(user_input)

            with st.chat_message("assistant"):
                msg_assistant = response.content 
                st.write(msg_assistant)
                st.session_state["messages"].append({"role": "assistant", "content": msg_assistant})



if __name__ == "__main__":
    llm = ChatLLM()
    web = ChatWeb(llm=llm)
    web.run()
