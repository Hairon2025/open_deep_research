


import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    lead_researcher_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentState,
    ClarifyWithUser,
    ResearchQuestion
)
from open_deep_research.utils import (
    get_api_key_for_model,
    get_today_str,
)

# Initialize a configurable model that we will use throughout the agent
# 初始化一个可配置模型，在整个智能体中复用
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    分析用户消息，如果研究范围不清晰则提出澄清问题。该函数决定用户请求在进入研究前是否需要澄清。若澄清功能被禁用或无需澄清，则直接进入研究流程。

    Args:
        state: Current agent state containing user messages(当前智能体状态，包含用户消息)
        config: Runtime configuration with model settings and preferences(运行时配置，包含模型设置和偏好)
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
        返回的命令指示智能体是结束于澄清问题还是继续研究简报。
    """

    # Step 1: Check if clarification is enabled in configuration
    # 第 1 步：检查配置中是否启用了澄清功能
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    # 第 2 步: 准备用于结构化澄清分析的模型
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    # 为模型配置结构化输出与重试逻辑
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    # 第 3 步：分析是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    # 第 4 步：根据澄清分析结果进行路由
    if response.need_clarification:
        # End with clarifying question for user
        # 以澄清问题结束，向用户提问
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        # 继续研究并附上验证消息
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.

    将用户消息转化为结构化的研究简报，并初始化主管。

    该函数分析用户消息并生成一份聚焦的研究简报，用于指导研究主管。
    同时使用适当的提示词和指令初始化主管上下文。
    
    Args:
        state: Current agent state containing user messages（当前智能体状态，包含用户消息）
        config: Runtime configuration with model settings（运行时配置，包含模型设置）
        
    Returns:
        Command to proceed to research supervisor with initialized context
        Command：携带已初始化的上下文，继续进入研究主管
    """
    # Step 1: Set up the research model for structured output
    # 第 1 步：为结构化输出准备研究模型
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    # 为结构化研究问题生成配置模型
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    # 第 2 步：根据用户消息生成结构化研究简报
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with research brief and instructions
    # 第 3 步：使用研究简报与指令初始化主管
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            # 使用 override 覆盖主管消息，而不是追加
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )