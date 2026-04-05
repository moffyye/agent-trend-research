# run_once.py

from dotenv import load_dotenv
load_dotenv()

import asyncio
from dotenv import load_dotenv
load_dotenv()

from app.agent_impl import run_agent
from app.word_writer import write_report_to_word


async def main():
    result = await run_agent(
        "What are the hottest AI trends in the last 30 days?",
        "local-run"
    )

    # 👇 在这里写
    content = f"""
=== Node A ===
{result["results"]["node_a"]}

=== Node B ===
{result["results"]["node_b"]}

=== Node C ===
{result["results"]["node_c"]}

=== Node D ===
{result["results"]["node_d"]}

=== Summary ===
{result["results"]["summary"]}
"""

    file_path = write_report_to_word(
        title="AI Trend Report",
        content=content   # 👈 用 content 而不是 summary
    )

    print(f"✅ Report saved to: {file_path}")


if __name__ == "__main__":
    asyncio.run(main())