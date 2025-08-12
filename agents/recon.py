import os

import git
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from tools.view_directory_tools import (
    DirectoryListingTool,
    DirectoryStructureTool,
    FileListingTool,
)

# Import our custom tools
from tools.view_file_tools import ViewFileLinesTool, ViewFileTool


# Load environment variables
load_dotenv()

repo_url = "https://github.com/juice-shop/juice-shop.git"
repo_path = "./repo"

if os.path.isdir(repo_path) and os.path.isdir(os.path.join(repo_path, ".git")):
    print("Directory already contains a git repository.")
else:
    try:
        repo = git.Repo.clone_from(repo_url, repo_path)
        print(f"Repository cloned into: {repo_path}")
    except Exception as e:
        print(f"An error occurred while cloning the repository: {e}")

# Define tools and LLM
tools = [
    ViewFileTool(),
    ViewFileLinesTool(),
    DirectoryListingTool(),
    FileListingTool(),
    DirectoryStructureTool(),
]
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model_kwargs={"temperature": 0.6},
)

instructions = """
You are an expert security analyst with years of experience analyzing code and identifying vulnerabilities. You are given a web application written in Node.js, Express and Angular. 

"""
prompt = PromptTemplate.from_template(instructions)

# Create agent and executor
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)


def analyze_code(input_code: str) -> dict:
    """
    Analyze the given code using the agent_executor and return the result.
    """
    # Use return_intermediate_steps=True to capture the thinking process
    response = agent_executor.invoke(
        {"input": input_code}, return_intermediate_steps=True
    )

    # Print the thinking steps for visibility
    if "intermediate_steps" in response and response["intermediate_steps"]:
        print("\n🧠 Agent Thinking Process:")
        print("-" * 40)
        for i, (action, observation) in enumerate(response["intermediate_steps"], 1):
            print(f"Step {i}:")
            print(f"  Action: {action.tool} - {action.tool_input}")
            print(f"  Observation: {observation}")
            print()

    return response


# Simple LangGraph integration
try:
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    class AgentState(TypedDict):
        input: str
        output: str

    def agent_node(state: AgentState) -> AgentState:
        """Run the ReAct agent"""
        result = agent_executor.invoke({"input": state["input"]})
        return {"input": state["input"], "output": result["output"]}

    # Create simple LangGraph workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)

    # Compile the graph
    langgraph_app = workflow.compile()

    def analyze_code_with_langgraph(input_code: str) -> dict:
        """Analyze code using LangGraph wrapper around ReAct agent"""
        result = langgraph_app.invoke({"input": input_code})
        return result

except ImportError:
    print("LangGraph not available. Using standard ReAct agent only.")

    def analyze_code_with_langgraph(input_code: str) -> dict:
        return analyze_code(input_code)


if __name__ == "__main__":
    print("🚀 ReAct Agent SAST Demo")
    print("=" * 50)

    # Task for autonomous analysis
    analysis_task = "Analyze the Node.js, Express and Angular code in ./repo/ for security vulnerabilities. Start by exploring the directory structure to understand the codebase."

    # Standard ReAct agent
    print("\n📋 Standard ReAct Agent:")
    result = analyze_code(analysis_task)
    print(result)

    # LangGraph wrapped ReAct agent
    print("\n🔄 LangGraph ReAct Agent:")
    langgraph_result = analyze_code_with_langgraph(analysis_task)
    print(langgraph_result)
