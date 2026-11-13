import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    compress_research_simple_human_message,
    compress_research_system_prompt,
    research_system_prompt,
)
from open_deep_research.state import (
    ResearcherOutputState,
    ResearcherState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
)

# Initialize a configurable model that we will use throughout the agent
# 初始化一个可配置模型，在整个智能体中复用
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


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