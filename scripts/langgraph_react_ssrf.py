from langchain.agents import create_react_agent, AgentExecutor
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import git

# Import our custom tools
from view_file_tools import ViewFileTool, ViewFileLinesTool
from view_directory_tools import (
    DirectoryListingTool,
    FileListingTool,
    DirectoryStructureTool,
)


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
    model_kwargs={"temperature": 0.4},
)



# Define instructions and prompt
instructions = """
You are an agent designed to analyze Python/Django code for SQL injection vulnerabilities.
The source code is located at ./repo/

### Analysis Process
1. Initial Review:
    - Search the code for interactions with SQL. 
    - Note down all files that interact with SQL. You will need this later.
    - Analyze those files for SQL injection vulnerabilities
    - After finding SQL injection vulnerabilities, continue looking in other files for vulnerabilities.
2. Reflection Questions:
    Consider these questions carefully:
    - Are there any files that interact with a SQL database that I am missing?
    - Have I looked at every file for potential SQL injection?
3. Challenge Initial Assessment:
    - Are there more files for me to look at?

### **TOOLS**
You have access to a vector database to search for code-related information. Use it to understand how custom functions handle authorization.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, 
or if you do not need to use a tool, 
you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

### **Output Format**
Your final output format should include ALL files that interact with SQL and that are vulnerable to SQL injection.
If you miss a file in the final response, you will be fired!

Your final response must be in JSON format, containing the following fields:
- `SQL-files`: (str) List of all files that use SQL.
- `SQLi-files`: (str) List of all files with SQL injection. 

New input: {input}
{agent_scratchpad}
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
        {"input": input_code}, 
        return_intermediate_steps=True
    )
    
    # Print the thinking steps for visibility
    #if "intermediate_steps" in response and response["intermediate_steps"]:
        #print("\n🧠 Agent Thinking Process:")
        #print("-" * 40)
       # for i, (action, observation) in enumerate(response["intermediate_steps"], 1):
           # print(f"Step {i}:")
            #print(f"  Action: {action.tool} - {action.tool_input}")
            #print(f"  Observation: {observation}")
            #print()
    
    return response


# Simple LangGraph integration
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict

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
    analysis_task = "Analyze the Python/Django code in ./repo/ for SQL injection vulnerabilities. Start by exploring the directory structure to understand the codebase. "

    # Standard ReAct agent
    #print("\n📋 Standard ReAct Agent:")
    result = analyze_code(analysis_task)
    #print(result)

    # LangGraph wrapped ReAct agent
    print("\n🔄 LangGraph ReAct Agent:")
    langgraph_result = analyze_code_with_langgraph(analysis_task)
    print(langgraph_result)
