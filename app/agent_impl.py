from pydantic import BaseModel
from agents import WebSearchTool,Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace

# Tool definitions
web_search_preview = WebSearchTool(
  user_location={
    "type": "approximate",
    "country": None,
    "region": None,
    "city": None,
    "timezone": None
  },
  search_context_size="medium"
)

class NodeDOpenSourceBuilderSignalSchema(BaseModel):
    summary: str | None = None


# -------- Agents --------

node_a_research_trend_analyzer = Agent(
    name="Node A - Research Trend Analyzer",
    instructions="""
    Find AI research trends in the last 30 days.
    
    Focus on:
    - arXiv
    - conference announcements
    - research lab blogs

    Return:
    - top 5 research trends
    - 1-2 concrete examples for each trends and release data and brief of the paper 
    - author of the most cited papers in these top 5 research trends
    - list 5 companies that authors are afflicated to 
    - why each trend appears to be rising now
    """,
    model="gpt-4.1",
    tools=[
    web_search_preview
    ],
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)

nodeb_lab_industry_signal = Agent(
    name="NodeB - Lab / Industry Signal",
    instructions="""
    You are analyzing VC and startup activity in AI in the last 30 days.

    Task:
    - Analyze AI startup funding, venture activity, notable rounds, acquisitions, and investor behavior
    from the 30 days immediately preceding.
    - Do not use June 2024 or any other assumed system knowledge date.
    
    Output:
    - 5 to 10 key investment signals
    - notable companies / rounds
    - recurring themes
    - investor sentiment
    - short bottom-line summary
    """,

    model="gpt-4.1",
    tools=[
    web_search_preview
    ],
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)

node_c_vc_investment_signal = Agent(
    name="Node C — VC / Investment Signal",
    instructions=f"""
    Analyze recent AI startup and VC activity in the last 30 days.

    Identify:
    - sectors receiving repeated investment
    - themes across multiple funds
    - new categories forming

    Output:
    - top 5 investment trends
    - example startup types
    - maturity (early / scaling)
    """,
    model="gpt-4.1",
    tools=[
    web_search_preview
    ],
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)

node_d_open_source_builder_signal = Agent(
    name="Node D — Open Source / Builder Signal",
    instructions=f"""
    Analyze trending AI open-source projects in the last 30 days.

    Focus on:
    - GitHub trending
    - HuggingFace models
    - developer tools gaining traction

    Output:
    - top trending tools or frameworks
    - what problem they solve
    - growth signals (stars, adoption)
    """,
    tools=[
    web_search_preview
    ],
    model="gpt-4.1",
    output_type=NodeDOpenSourceBuilderSignalSchema,
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)

summary_agent = Agent(
    name="summary",
    instructions="""You are the Trend Synthesizer...""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)


# -------- Runtime Entry --------

async def run_agent(user_input: str, user_id: str | None = None):
    with trace("research agent"):

        conversation_history: list[TResponseInputItem] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_input}],
            }
        ]

        run_config = RunConfig(
            trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69ce9019e3148190a71b1fabc5f1636d00f2e66db6e34c05",
                "user_id": user_id or "anonymous",
            }
        )

        # ---- Node A ----
        node_a = await Runner.run(node_a_research_trend_analyzer, input=conversation_history, run_config=run_config)
        conversation_history.extend([i.to_input_item() for i in node_a.new_items])

        # ---- Node B ----
        node_b = await Runner.run(nodeb_lab_industry_signal, input=conversation_history, run_config=run_config)
        conversation_history.extend([i.to_input_item() for i in node_b.new_items])

        # ---- Node C ----
        node_c = await Runner.run(node_c_vc_investment_signal, input=conversation_history, run_config=run_config)
        conversation_history.extend([i.to_input_item() for i in node_c.new_items])

        # ---- Node D ----
        node_d = await Runner.run(node_d_open_source_builder_signal, input=conversation_history, run_config=run_config)
        conversation_history.extend([i.to_input_item() for i in node_d.new_items])

        # ---- Summary ----
        summary = await Runner.run(summary_agent, input=conversation_history, run_config=run_config)

        return {
            "status": "ok",
            "input": user_input,
            "user_id": user_id,
            "results": {
                "node_a": node_a.final_output_as(str),
                "node_b": node_b.final_output_as(str),
                "node_c": node_c.final_output_as(str),
                "node_d": str(node_d.final_output),
                "summary": summary.final_output_as(str),
            },
        }