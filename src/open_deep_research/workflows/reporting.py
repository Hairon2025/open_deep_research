from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import final_report_generation_prompt
from open_deep_research.state import AgentState

from open_deep_research.utils import (
    get_api_key_for_model,
    get_model_token_limit,
    get_today_str,
    is_token_limit_exceeded,
)

# Initialize a configurable model that we will use throughout the agent
# 初始化一个可配置模型，在整个智能体中复用
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

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