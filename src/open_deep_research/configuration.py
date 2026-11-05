"""Configuration management for the Open Deep Research system."""
"""Open Deep Research 系统的配置管理。"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    """可用搜索 API 提供方的枚举。"""
    
    ANTHROPIC = "anthropic"  # Anthropic 原生网页搜索
    OPENAI = "openai"  # OpenAI 原生网页搜索
    TAVILY = "tavily"  # Tavily 搜索
    NONE = "none"  # 不使用搜索

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    """Model Context Protocol（MCP）服务器的配置。"""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    """MCP 服务器的 URL"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    """提供给 LLM 使用的工具列表"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""
    """MCP 服务器是否需要身份验证"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    """Deep Research 智能体的主配置类。"""
    
    # General Configuration（通用配置）

    # 结构化输出，最大重试次数配置，默认 3
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
                # 说明：模型结构化输出调用的最大重试次数
            }
        }
    )
    
    # 是否允许研究人员在开始研究前向用户提出澄清问题，默认 True（允许）
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
                # 说明：是否允许研究人员在开始研究前向用户提出澄清问题
            }
        }
    )

    # 同时运行的研究单元（子智能体）最大数量，默认 5，最少为 1，最大为 20。
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
                # 说明：可同时运行的研究单元最大数量。这允许研究人员使用多个子智能体开展研究。注意：并发越高，可能触发速率限制。
            }
        }
    )
    
    # Research Configuration（研究配置）
    
    # Search API Configuration（搜索 API 提供方配置）
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                # 说明：用于研究的搜索 API。注意：请确保你的 Researcher 模型支持所选的搜索 API。
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    
    # Research Supervisor 的最大研究迭代次数，范围 1 - 10, 默认 6
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
                # 说明：Research Supervisor 的最大研究迭代次数。即 Research Supervisor 反思研究并追问的轮数。
            }
        }
    )

    # 单个 DeepResearch 步骤中允许的最大工具调用迭代次数，范围 1 - 30, 默认 10
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
                # 说明：单个研究人员步骤中允许的最大工具调用迭代次数。
            }
        }
    )
    # Model Configuration（模型配置）
    
    # 汇总模型配置（Summarization Model Configuration）
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
                # 说明：用于汇总 Tavily 搜索结果所得研究结果的模型
            }
        }
    )

    # 汇总模型最大输出 token 数（Summarization Model Maximum Output Tokens）
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
                # 说明：汇总模型的最大输出 token 数
            }
        }
    )

    # 网页内容在汇总前的最大字符长度（Maximum Content Length Before Summarization）
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
                # 说明：网页内容在汇总前的最大字符长度
            }
        }
    )

    # 研究模型配置（Research Model Configuration）
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
                # 说明：用于开展研究的模型。注意：请确保你的 Researcher 模型支持所选的搜索 API。
            }
        }
    )

    # 研究模型最大输出 token 数（Research Model Maximum Output Tokens）
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
                # 说明：研究模型的最大输出 token 数
            }
        }
    )

    # 压缩模型配置（Compression Model Configuration）
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
                # 说明：用于压缩子智能体研究成果的模型。注意：请确保你的 Compression 模型支持所选的搜索 API。
            }
        }
    )

    # 压缩模型最大输出 token 数（Compression Model Maximum Output Tokens）
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
                # 说明：压缩模型的最大输出 token 数
            }
        }
    )

    # 最终报告模型配置（Final Report Model Configuration）
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
                # 说明：用于基于全部研究成果撰写最终报告的模型
            }
        }
    )

    # 最终报告模型最大输出 token 数（Final Report Model Maximum Output Tokens）
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
                # 说明：最终报告模型的最大输出 token 数
            }
        }
    )

    # MCP server configuration
    # MCP 服务器配置
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
                # 说明：MCP 服务器配置
            }
        }
    )

    # MCP 提示（MCP Prompt）
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
                # 说明：传递给智能体的、关于其可用 MCP 工具的任何附加说明。
            }
        }
    )

    # 配置加载机制（Configuration Loading Mechanism）
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        """从 RunnableConfig 创建 Configuration 实例。"""
        # 获取可配置项字典（若 config 为空则为空字典）
        configurable = config.get("configurable", {}) if config else {}
        # 获取本配置类的所有字段名
        field_names = list(cls.model_fields.keys())
        # 为每个字段取值：优先读取环境变量（字段名大写），否则从 configurable 中读取
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        # 仅使用非 None 的值构造实例
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        """Pydantic 配置。"""
        
        arbitrary_types_allowed = True  # 允许使用任意类型