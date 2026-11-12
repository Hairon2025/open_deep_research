## 整体编排流程

这个是整个 DeepResearch 的执行流程

```mermaid
flowchart LR
    START([START 入口]) --> CLARIFY{"clarify_with_user<br/>需要澄清？"}

    %% ---------- 主干：不需要澄清，横向推进 ----------
    CLARIFY -->|"否 / 禁用澄清"| BRIEF["write_research_brief<br/>生成结构化研究简报"]
    BRIEF --> SUP["research_supervisor<br/>主管子图（研究执行）"]
    SUP --> REPORT["final_report_generation<br/>生成最终报告"]
    REPORT --> END_OK([END 输出报告])

    %% ---------- 分支：需要澄清，纵向分叉 ----------
    CLARIFY -->|"是"| ASK["AIMessage 返回澄清问题"]
    ASK --> END_ASK([END 挂起等待用户])

    %% ---------- 外部循环：用户补充后再跑一轮 ----------
    END_ASK -.->|"用户补充后<br/>新一轮 invoke（外部循环）"| START

    %% ---------- 子图引用 ----------
    SUP -.->|"节点即子图<br/>展开见「图二」"| SUPSUB(["主管子图<br/>supervisor ⇄ supervisor_tools"])

    style START fill:#e1f5e1
    style END_OK fill:#d4e6ff
    style END_ASK fill:#ffe1e1
    style CLARIFY fill:#fff4d6
    style SUPSUB fill:#f0e6ff
```

## clarify_with_user澄清节点 函数流程图

Agent 觉得需求有歧义、范围不清，通过反问的方式问用户理解的对不对

```mermaid
flowchart LR
    A([clarify_with_user]) --> B{"allow_clarification?"}
    B -->|否| BRIEF(["开始研究流程<br/>write_research_brief"])
    B -->|是| C["调用模型<br/>ClarifyWithUser"]
    C --> D{"用户核对模型是否理解正确<br/>need_clarification?"}
    D -->|是| E(["END<br/>返回澄清问题"])
    D -->|否| F(["write_research_brief<br/>返回确认消息"])

    style A fill:#e1f5e1
    style E fill:#ffe1e1
    style BRIEF fill:#d4e6ff
    style F fill:#d4e6ff
```

## write_research_brief 节点

将和用户的聊天归纳/聚焦成研究问题，简单来说就好似「用 LLM 把聊天记录翻译成研究简报」+「用代码重置主管上下文」

```mermaid
flowchart LR
    IN["state.messages<br/>+ config"] --> S1["组装<br/>结构化模型"]
    S1 --> S2["消息 → 文本 → prompt<br/>→ LLM 调用"]
    S2 --> RQ["ResearchQuestion<br/>（research_brief）"]
    RQ --> CMD["进入下一节点research_supervisor<br/>也就是<br/>supervisor_subgraph"]

    style IN fill:#e1f5e1
    style RQ fill:#f0e6ff
    style CMD fill:#fff4d6
```

## research_supervisor 节点

### 子图流程

```mermaid
flowchart TD
    A["START"] --> B["supervisor<br/>模型决定下一步操作"]
    B --> C["supervisor_tools<br/>检查结束条件，处理工具调用"]
    C -->|"带回结果，继续决策"| B
    C -->|"研究阶段结束"| D["END"]
```

### supervisor_tools

`supervisor_tools` 节点判是否结束研究，最终子图END之后，有LangGraph 按主图定义的边调度 `final_report_generation`

```mermaid
flowchart TD
    A["进入 supervisor_tools"] --> B["读取配置、主管消息和迭代次数"]
    B --> C{"是否满足任一结束条件？<br/>① 迭代次数大于上限<br/>② 本轮没有工具调用<br/>③ 本轮调用了 ResearchComplete"}

    C -->|"是"| Z["从已有主管工具消息提取 notes<br/>返回 Command：goto=END<br/>之后模型按主图定义的边调用下一个节点"]
    C -->|"否"| D["处理本轮所有 think_tool 调用<br/>把反思内容记录成 ToolMessage"]

    D --> E{"本轮有 ConductResearch 调用吗？"}
    E -->|"没有"| J["把本轮工具消息写入 update_payload"]
    E -->|"有"| F["按并发上限划分任务<br/>获准执行的任务 / 超额任务"]

    F --> G["为获准任务调用 researcher_subgraph<br/>用 asyncio.gather 并行执行并等待结果"]
    G --> H["把每个研究员的 compressed_research<br/>封装成 ToolMessage"]
    H --> I["为超额任务生成提示消息<br/>汇总研究员的 raw_notes"]
    I --> J

    J --> K["返回 Command：goto=supervisor<br/>主管读取结果，开始下一轮决策"]

    G -.->|"研究任务处理过程中抛出异常"| Z
```


