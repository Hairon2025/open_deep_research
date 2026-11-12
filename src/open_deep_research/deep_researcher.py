"""Main LangGraph implementation for the Deep Research agent."""
"""Deep Research 智能体的主 LangGraph 实现。"""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
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
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
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


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    主导研究的主管：规划研究策略并委派给研究人员。

    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    主管分析研究简报，决定如何把研究拆解为可管理的任务。它可以使用 think_tool 做策略性思考，
    使用 ConductResearch 把任务委派给子研究人员，或在满意时使用 ResearchComplete 表示完成。

    Args:
        state: Current supervisor state with messages and research context（当前主管状态，包含消息与研究上下文）
        config: Runtime configuration with model settings（运行时配置，包含模型设置）
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
        Command：继续进入 supervisor_tools 执行工具调用
    """
    # Step 1: Configure the supervisor model with available tools
    # 第 1 步：为带有可用工具的主管模型进行配置
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    # 可用工具：研究委派、完成信号、策略性思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    # 为模型绑定工具、重试逻辑和模型设置
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    # 第 2 步：基于当前上下文生成主管响应
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    # 第 3 步：更新状态并进入工具执行
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    执行主管调用的工具，包括研究委派与策略性思考。

    This function handles three types of supervisor tool calls:
    该函数处理三类主管工具调用：

    1. think_tool - Strategic reflection that continues the conversation（策略性反思，继续对话）
    2. ConductResearch - Delegates research tasks to sub-researchers（将研究任务委派给子研究人员）
    3. ResearchComplete - Signals completion of research phase（表示研究阶段完成）

    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    # 第 1 步：提取当前状态并检查退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    # 定义研究阶段的退出条件
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    # 满足任一终止条件即退出
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    # 第 2 步：统一处理所有工具调用（think_tool 与 ConductResearch）
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    # 处理 think_tool 调用（策略性反思）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    # 处理 ConductResearch 调用（研究委派）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            # 限制并发研究单元数量，防止资源耗尽
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            # 并行执行研究任务
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            # 使用研究结果创建工具消息
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            # 对超额的委派调用返回错误消息
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            # 汇总所有研究结果的原始笔记
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            # 处理研究执行过程中的错误
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                # token 超限或其他错误 —— 结束研究阶段
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: Return command with all tool results
    # 第 3 步：携带全部工具结果返回命令
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
# 主管子图构建
# 创建管理研究委派与协调的主管工作流
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
# 添加用于研究管理的主管节点
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic / 主管主逻辑
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler / 工具执行处理器

# Define supervisor workflow edges
# 定义主管工作流的边
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor / 主管入口

# Compile supervisor subgraph for use in main workflow
# 编译主管子图，供主工作流使用
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    负责在特定主题上开展聚焦研究的单个研究人员。

    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    主管会给该研究人员指定一个具体研究主题，它使用可用工具（搜索、think_tool、MCP 工具）
    收集全面信息。它可以在搜索之间使用 think_tool 做策略性规划。


    Args:
        state: Current researcher state with messages and topic context；当前研究人员状态，包含消息与主题上下文
        config: Runtime configuration with model settings and tool availability；运行时配置，包含模型设置与工具可用性
        
    Returns:
        Command to proceed to researcher_tools for tool execution；
        继续进入 researcher_tools 执行工具调用
    """
    # Step 1: Load configuration and validate tool availability
    # 第 1 步：加载配置并校验工具可用性
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    # 获取所有可用研究工具（搜索、MCP、think_tool）
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    # 第 2 步：为研究人员模型绑定工具
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    # 准备系统提示词，若存在则包含 MCP 上下文
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    # 为模型配置工具、重试逻辑和设置
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    # 第 3 步：结合系统上下文生成研究人员响应
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    # 第 4 步：更新状态并进入工具执行
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1  # 工具调用迭代次数自增
        }
    )

# Tool Execution Helper Function
# 工具执行辅助函数
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    """带错误处理地安全执行工具。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    执行研究人员调用的工具，包括搜索工具与策略性思考。

    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    该函数处理多类研究人员工具调用：
    1. think_tool —— 策略性反思，继续研究对话
    2. 搜索工具（tavily_search、web_search）—— 信息收集
    3. MCP 工具 —— 外部工具集成
    4. ResearchComplete —— 表示单个研究任务完成

    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    # 第 1 步：提取当前状态并检查提前退出条件
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    # 若无工具调用（包括原生网页搜索）则提前退出
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    # 第 2 步：处理其他工具调用（搜索、MCP 工具等）
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    # 并行执行所有工具调用
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    # 根据执行结果创建工具消息
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    # 第 3 步：处理工具后检查后置退出条件
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls  # 超出最大工具调用次数
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )  # 调用了 ResearchComplete
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        # 结束研究并进入压缩
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    # 携带工具结果继续研究循环
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    """将研究发现压缩并综合为简洁、结构化的摘要。

    该函数接收研究人员的全部研究发现、工具输出和 AI 消息，将其提炼为一份干净、
    全面的摘要，同时保留所有重要信息与发现。

    参数：
        state：当前研究人员状态，包含累积的研究消息
        config：运行时配置，包含压缩模型设置

    返回：
        包含压缩研究摘要与原始笔记的字典
    """
    # Step 1: Configure the compression model
    # 第 1 步：配置压缩模型
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    # 第 2 步：准备用于压缩的消息
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    # 添加指令，从研究模式切换到压缩模式
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    # 第 3 步：尝试压缩，并针对 token 超限问题做重试
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            # 创建聚焦于压缩任务的系统提示词
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            # 执行压缩
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            # 从所有工具消息与 AI 消息中提取原始笔记
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            # 返回成功的压缩结果
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            # 若 token 超限，则移除较早的消息后重试
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            # 其他错误则继续重试
            continue
    
    # Step 4: Return error result if all attempts failed
    # 第 4 步：若全部尝试失败，返回错误结果
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# 研究人员子图构建
# Creates individual researcher workflow for conducting focused research on specific topics
# 创建用于在特定主题上开展聚焦研究的单个研究人员工作流
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
# 添加用于研究执行与压缩的研究人员节点
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic / 研究人员主逻辑
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler / 工具执行处理器
researcher_builder.add_node("compress_research", compress_research)   # Research compression / 研究压缩

# Define researcher workflow edges
# 定义研究人员工作流的边
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher / 研究人员入口
researcher_builder.add_edge("compress_research", END)      # Exit point after compression / 压缩后出口

# Compile researcher subgraph for parallel execution by supervisor
# 编译研究人员子图，供主管并行执行
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    生成最终的综合性研究报告，并针对 token 限制做重试。

    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    该函数接收所有收集到的研究成果，使用配置的报告生成模型将其综合为一份
    结构良好的综合性最终报告。

    Args:
        state: Agent state containing research findings and context；包含研究成果与上下文的智能体状态
        config: Runtime configuration with model settings and API keys；运行时配置，包含模型设置与 API 密钥
    Returns:
        Dictionary containing the final report and cleared state
        包含最终报告与已清空状态的字典
    """
    # Step 1: Extract research findings and prepare state cleanup
    # 第 1 步：提取研究成果并准备状态清理
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}  # 清空笔记
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    # 第 2 步：配置最终报告生成模型
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    # 第 3 步：尝试生成报告，并针对 token 超限做重试
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            # 构建包含全部研究上下文的完整提示词
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            # 生成最终报告
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # Return successful report generation
            # 返回成功的报告生成结果
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            # 处理 token 超限错误，逐步截断内容
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    # 首次重试：确定初始截断上限
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    # 用 token 上限的 4 倍作为字符近似截断上限
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    # 后续重试：每次减少 10%
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                # 截断研究成果后重试
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                # 非 token 超限错误：立即返回错误
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    # 第 4 步：若全部重试耗尽，返回失败结果
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# 主 Deep Researcher 图构建
# Creates the complete deep research workflow from user input to final report
# 创建从用户输入到最终报告的完整深度研究工作流
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
# 添加完整研究流程的主工作流节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase / 用户澄清阶段
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase / 研究规划阶段
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase / 研究执行阶段
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase / 报告生成阶段

# Define main workflow edges for sequential execution
# 定义主工作流的顺序执行边
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point / 入口
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report / 研究到报告
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point / 最终出口

# Compile the complete deep researcher workflow
# 编译完整的 Deep Researcher 工作流
deep_researcher = deep_researcher_builder.compile()