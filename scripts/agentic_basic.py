from langchain.agents import create_react_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")


class CustomSearchTool(BaseTool):
    name: str = "custom_search"
    description: str = "Useful for when you need to answer questions about code"
    args_schema: Type[SearchInput] = SearchInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        faiss_db_path = "../vector_databases/juice_shop_faiss"
        db = FAISS.load_local(
            faiss_db_path,
            BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"),
            allow_dangerous_deserialization=True,
        )
        return db.similarity_search(query)

    async def _arun(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        raise NotImplementedError("custom_search does not support async")


# Define tools and LLM
tools = [CustomSearchTool()]
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model_kwargs={"temperature": 0.4},
)

# Define instructions and prompt
instructions = """
You are an agent designed to analyze Python code for potential Server Side Request Forgery (SSRF) vulnerabilities.

### Analysis Process
1. Initial Review:
   - Identify where the code makes HTTP requests
   - Locate user-supplied input that influences the request
   - Find authorization checks in the code

2. Reflection Questions:
   Consider these questions carefully:
   - How does the code determine which requests a user can make?
   - What prevents a user from accessing records belonging to others?
   - Is there a mismatch between authorization scope and data access?
   - Could changing the input parameters bypass the authorization?

3. Challenge Initial Assessment:
   - What assumptions did you make about the authorization?
   - Are you certain the authorization check applies to the specific record?
   - What would an attacker try first to bypass these controls?

### **TOOLS**
You have access to a vector database to search for code-related information. Use it to understand how custom functions handle authorization.

### **Output Format**
Your final response must be in JSON format, containing the following fields:
- `is_insecure`: (bool) Whether the code is considered insecure.
- `reason`: (str) The reason the code is considered insecure or secure.

TOOLS:
------

You have access to the following tools:

{tools}

You must use the tool.

```
Thought: Would the tool be helpful for my task? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, 
or if you do not need to use a tool, 
you MUST use the format:

```
Thought: Would the tool be helpful for my task? No
Final Answer: [your response here]
```

Your Final Answer should be in JSON format 
with the following fields:

- is_insecure: (bool) whether the code is considered insecure
- reason: (str) the reason the code is considered insecure
- used_tool: (bool) did I use a tool or not

Begin!

New input: {input}
{agent_scratchpad}
"""

examples = [

    {
        "context": """
            def get_users(username):
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM users WHERE username = '%s'" % username)
                return cursor.fetchall()
        """,
        "question": "Identify everywhere in the Django application code base that contains SQL Injection vulnerablities.",
        "answer": "This code is vulnerable to SQL Injection because it takes the username (user input) and directly concatenates it into the SQL query."
    },
    {
        "context": """
            def get_users(username):
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM users WHERE username = %s", [username])
                return cursor.fetchall()
        """,
        "question": "Identify everywhere in the Django application code base that contains SQL Injection vulnerablities.",
        "answer": """
            Although the code uses the `cursor.execute()` method, it is using
            parameterized queries, therefore it is not
            vulnerable to SQL Injection.
        """
    },
    {
        "context":"""
            if re.match('.*?(rm|sudo|wget|curl|su|shred) .*',ip,re.I):
                data = "Nice try on the dangerous commands, but no"
            else:
                cmd = "ping -c 5 %s" % ip
                data = subprocess.getoutput(cmd)
        """,
        "question":"Identify everywhere in the Django application code base that contains SQL Injection vulnerablities.",
        "answer": """
            This code is not vulnerable to SQL Injection because it is command injection so I will not mention it.
        """
    },
    {
        "context": """
            ssn = forms.CharField(max_length=11, required=False)
        """,
        "question": "Identify everywhere in the Django application code base that contains SQL Injection vulnerablities.",
        "answer": """
            This code is not vulnerable to SQL Injection because it is a form field definition.
        """
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "<question>{question}</question>\n<context>{context}</context>"),
        ("ai", "{answer}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        few_shot_prompt,
        ("human", """<question>{question}</question>""")
    ]
)

# Create agent and executor
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)


def analyze_code(input_code: str) -> dict:
    """
    Analyze the given code using the agent_executor and return the result.
    """
    response = agent_executor.invoke({"input": input_code})
    return response


if __name__ == "__main__":
    # Example input
    input_code = """
    @login_required
    @user_passes_test(can_create_project)
    def update_user_active(request):
        user_id = request.GET.get('user_id')
        User.objects.filter(id=user_id).update(is_active=False)
    """
    result = analyze_code(input_code)
    print(result)
