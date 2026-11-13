"""Main LangGraph implementation for the Deep Research agent."""
"""Deep Research 智能体的主 LangGraph 实现。"""

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState
)

from open_deep_research.workflows.preparation import (
    clarify_with_user,
    write_research_brief,
)
from open_deep_research.workflows.reporting import final_report_generation
from open_deep_research.workflows.supervisor import supervisor_subgraph

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