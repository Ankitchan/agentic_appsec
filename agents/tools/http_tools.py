import os
from typing import Optional, Type

import httpx
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class HttpInput(BaseModel):
    url: str = Field(description="a url to make a request to")
    method: str = Field(
        description="the http method to use (GET or POST)", default="GET"
    )
    data: Optional[dict] = Field(
        description="the data to send with a POST request", default=None
    )


class HttpTool(BaseTool):
    name: str = "http_tool"
    description: str = (
        "Useful for when you need to make a request to a url. Can be used for GET and POST requests."
    )
    args_schema: Type[HttpInput] = HttpInput

    def _run(
        self,
        url: str,
        method: str = "GET",
        data: Optional[dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if method.upper() == "POST":
            response = httpx.post(url, data=data)
        else:
            response = httpx.get(url)
        headers = str(response.headers)
        body = response.text
        return f"Headers:\n{headers}\n\nBody:\n{body}"

    async def _arun(
        self,
        url: str,
        method: str = "GET",
        data: Optional[dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("http_tool does not support async")
