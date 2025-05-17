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
    user_input = input("你: ")
    state = {"messages": [HumanMessage(content=user_input)]}
    for s in app.stream(state):
        print("AI:", s['chat']['messages'][-1].content)