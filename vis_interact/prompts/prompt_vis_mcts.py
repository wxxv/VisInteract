"""Prompt templates for the MCTS visualization agent.

System prompt and task prompt only — the ReAct history is carried as
native tool-call / tool-result messages, not as prompt text.
"""

from __future__ import annotations

from typing import List, Tuple


MCTS_SYSTEM_PROMPT = """You are an expert data visualization agent. Your goal is to produce the best Altair visualization for the user's request.

**Critical mindset**: The user's natural language query is a noisy, incomplete approximation of their true intent. It may contain ambiguities, missing information, or suboptimal preferences — e.g., requesting a line chart when the data only has a single time point, asking for a stacked bar when there is no sub-category to stack, or referencing a column name like "revenue" when the actual column is "total_sales". Do NOT blindly follow the query. Verify against actual data and make the best judgment call at each step.

**Proactive communication mindset**: The user's query is just a rough starting point. You should ACTIVELY talk to the user via `ask_user_text` — not only when you're confused, but whenever confirming something with the user could lead to a better chart. This includes confirming your plan, verifying assumptions, seeking preferences, or reacting to data findings. When in doubt, ask; don't guess.

## How You Operate

A search algorithm calls you multiple times from different states to explore diverse design approaches. Your job is to make the single best decision at each step — the algorithm handles exploration across alternatives.

## Tools

| Tool | Purpose |
|------|---------|
| `execute_sql` | Explore data in the SQLite database |
| `execute_altair` | Render an Altair chart based on your design decisions |
| `ask_user_text` | Talk to the user — confirm your plan, verify assumptions, seek preferences, or resolve ambiguity |
| `finish` | Submit the visualization from your last successful `execute_altair` — takes NO parameters, can ONLY be called right after `execute_altair` |

## Workflow

1. **Explore data** — Use `execute_sql` to understand the data.
2. **Confirm with the user** — Before calling `execute_altair`, use `ask_user_text` to check in with the user. You can also ask at any other point during the workflow.
3. **Implement** — Use `execute_altair` to render the visualization.
4. **Finish promptly** — As soon as `execute_altair` renders successfully and you are satisfied, call `finish` immediately. `finish` takes no parameters — it submits the code from your most recent `execute_altair`.
5. **Respect prior signals** — If *User Clarifications* or *Previous Visual Feedback* are provided, build on what scored well and avoid what scored poorly.

## Rules
- **Talk to the user freely**: If you have text-question budget remaining, use `ask_user_text`. You do NOT need a strong reason to justify asking — any question that could help is worth asking.".
- Use only tables and columns from the provided schema in SQL and encodings — do NOT fabricate or guess names. Use `execute_sql` to validate data assumptions (ranges, nulls, cardinality), not to rediscover column names.
- The code submitted via `finish` is exactly the code from your last successful `execute_altair` — make sure it is complete and self-contained before calling `finish`.
- If `execute_altair` fails, read the error, fix the code, and retry.
- Be efficient. Once you have rendered a correct chart with `execute_altair` and are satisfied, call `finish` right away. Extra SQL queries or redundant renders waste budget without improving the result.

{code_structure}
"""


WIDEN_DIVERSITY_NOTE = """[Search Diversity] The following approaches have already been explored at this decision point:
{existing_actions}

You MUST take a DIFFERENT approach. Consider:
- A different tool entirely (e.g. ask_user_text instead of execute_sql)
- The same tool but with substantially different arguments (different SQL query, different chart type, different data scope)
- A different reasoning strategy

Do NOT repeat or trivially vary the listed approaches."""


def build_task_prompt(
    query: str,
    db_id: str,
    db_schema: str,
) -> str:
    """Build the user message describing the visualization task.

    This message is intentionally kept **static** (no global QA or feedback)
    so that it forms a stable prefix for prompt-cache reuse across rollouts.
    Dynamic context (QA, feedback) is appended via STATUS_NOTE at the end.
    """
    parts = [
        f"## User Request\n{query}\n",
        f"## Database: `{db_id}`\n## Schema (tables and columns)\n{db_schema}\n",
    ]
    return "\n".join(parts)


def build_context_suffix(
    global_qa: List[Tuple[str, str]],
    global_feedbacks: List[Tuple[int, str]],
) -> str:
    """Build the dynamic QA / feedback section appended after the tree path."""
    parts: list[str] = []
    if global_qa:
        qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in global_qa)
        parts.append(
            "## User Clarifications (from earlier exploration, not the current path)\n"
            + qa_text
            + "\n\nIncorporate these clarifications into your design. Do NOT ask the same or similar questions again."
        )
    if global_feedbacks:
        fb_text = "\n".join(f"- [Score: {s}/10] {f}" for s, f in global_feedbacks)
        parts.append(
            "## Previous Visual Feedback (from other attempts — use to guide your design)\n"
            + fb_text
            + "\n\nCarefully study the feedback above. Repeat strengths from high-scoring attempts and actively fix the specific issues mentioned in low-scoring ones. Do NOT reproduce the same mistakes."
        )
    return "\n\n".join(parts)


STATUS_NOTE = """[Current Status] Steps used: {steps_used}/{max_steps} ({steps_remaining} remaining), text-question budget: {b_text}.{urgency}"""


REWARD_DECOMPOSE_PROMPT = """You are evaluating a visualization generated by an AI agent. Given the chart image, the agent's reasoning trajectory, and the user's feedback, assess the relative quality of three independent aspects.

## Aspects:
- **data_fidelity**: Is the underlying data correct? Look at the SQL queries and their results — are the right tables, columns, filters, aggregations, and value ranges used?
- **vis_design**: Is the visual design appropriate? Look at the chart image — are the chart type, axis mapping, sort order, colors, labels, and readability good?
- **intent_alignment**: Does the result address what the user actually wanted? Consider the original query and any clarification exchanges.

## Rules:
1. Each aspect score must be between 1 and 10.
2. Use ALL available evidence: the chart image, the trajectory details, AND the feedback text.
3. Focus on the RELATIVE quality: which aspects are good and which are bad.
4. If the feedback does not mention a specific aspect, judge it yourself from the image and trajectory.

## Agent Trajectory:
{trajectory}

## User Feedback:
Overall score: {score}/10
Feedback: "{feedback}"

## Output (JSON only):
{{"data_fidelity": N, "vis_design": N, "intent_alignment": N}}"""
