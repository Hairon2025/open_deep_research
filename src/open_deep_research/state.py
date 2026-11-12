"""Graph state definitions and data structures for the Deep Research agent."""
"""Deep Research 智能体的图状态定义与数据结构。"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
# 结构化输出
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    """调用此工具以针对特定主题开展研究。"""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )
    # 说明：待研究的主题。应为单一主题，并且需要以高细节描述（至少一段话）。

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""
    """调用此工具以表示研究已完成。"""

class Summary(BaseModel):
    """Research summary with key findings."""
    """包含关键发现的研究摘要。"""
    
    summary: str  # 摘要
    key_excerpts: str  # 关键摘录

class ClarifyWithUser(BaseModel):
    """
    Model for user clarification requests.
    用于向用户请求澄清的模型。
    """
    # 说明：是否需要向用户提出澄清问题。
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    # 说明：用于询问用户以澄清报告范围的问题。
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    # 说明：确认消息，表示在用户提供必要信息后我们将开始研究。
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    """用于指导研究的研究问题与研究简报。"""
    
    # 说明：将用于指导研究的研究问题。
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################
# 状态定义
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    """允许覆盖状态中值的 reducer 函数。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        # 如果新值是字典且类型为 "override"，则直接返回其 value（实现覆盖）
        return new_value.get("value", new_value)
    else:
        # 否则使用加法合并当前值与新值
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""
    """InputState 仅包含 'messages'（输入状态仅包含消息）。"""

class AgentState(MessagesState):
    """Main agent state containing messages and research data.
    主智能体状态，包含消息与研究数据。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]  # 主管消息列表（支持覆盖合并）
    research_brief: Optional[str]  # 研究简报（可选）
    raw_notes: Annotated[list[str], override_reducer] = []  # 原始笔记列表（支持覆盖合并），默认空
    notes: Annotated[list[str], override_reducer] = []  # 笔记列表（支持覆盖合并），默认空
    final_report: str  # 最终报告

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks.
    主管状态，用于管理研究任务。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]  # 主管消息列表（支持覆盖合并）
    research_brief: str  # 研究简报
    notes: Annotated[list[str], override_reducer] = []  # 笔记列表（支持覆盖合并），默认空
    research_iterations: int = 0  # 研究迭代次数，默认 0
    raw_notes: Annotated[list[str], override_reducer] = []  # 原始笔记列表（支持覆盖合并），默认空

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    """单个研究人员开展研究时的状态。"""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]  # 研究人员消息列表（使用加法合并）
    tool_call_iterations: int = 0  # 工具调用迭代次数，默认 0
    research_topic: str  # 研究主题
    compressed_research: str  # 压缩后的研究内容
    raw_notes: Annotated[list[str], override_reducer] = []  # 原始笔记列表（支持覆盖合并），默认空

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    """单个研究人员的输出状态。"""
    
    compressed_research: str  # 压缩后的研究内容
    raw_notes: Annotated[list[str], override_reducer] = []  # 原始笔记列表（支持覆盖合并），默认空