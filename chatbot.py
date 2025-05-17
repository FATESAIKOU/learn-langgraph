#!/usr/bin/env python

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from typing import TypedDict

class GraphState(TypedDict):
    messages: list

llm = ChatOllama(model="qwen3:8b")

def reply(state: GraphState) -> GraphState:
    history = state["messages"]
    response = llm.invoke(history)
    return {"messages": history + [response]}

def is_done(state: GraphState) -> str:
    content = state["messages"][-1].content.lower()
    return END if "bye" in content else "continue"

graph = StateGraph(GraphState)
graph.add_node("chat", reply)
graph.set_entry_point("chat")
graph.add_conditional_edges("chat", is_done)
app = graph.compile()

if __name__ == "__main__":
    state: GraphState = {"messages": []}

    while True:
        user_input = input("你: ")
        state["messages"].append(HumanMessage(content=user_input))

        for output in app.stream(state):
            # 只保留 chat 節點的輸出，維持結構一致
            state = output["chat"]
            ai_msg = state["messages"][-1]
            print("AI:", ai_msg.content)

        if is_done(state) == END:
            print("對話結束。")
            break
