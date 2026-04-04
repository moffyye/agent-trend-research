from pydantic import BaseModel
from agents import Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace



class NodeDOpenSourceBuilderSignalSchema(BaseModel):
    summary: str | None = None


# -------- Agents --------

node_a_research_trend_analyzer = Agent(
    name="Node A - Research Trend Analyzer",
    instructions="""Find AI research trends in the last 30 days...""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)

nodeb_lab_industry_signal = Agent(
    name="NodeB - Lab / Industry Signal",
    instructions="""Analyze latest work from top AI labs...""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)

node_c_vc_investment_signal = Agent(
    name="Node C — VC / Investment Signal",
    instructions="""Analyze recent AI startup and VC activity...""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True),
)

node_d_open_source_builder_signal = Agent(
    name="Node D — Open Source / Builder Signal",
    instructions="""Analyze trending AI open-source projects...""",
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