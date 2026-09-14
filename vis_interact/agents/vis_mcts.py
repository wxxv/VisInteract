"""
VisInteract MCTS Agent

MCTS-guided interactive Text-to-Vis agent with Progressive Widening.
Each rollout may add one new child at a decision point via diversity-prompted
LLM calls; simulation extends the path to a terminal state.
"""

from __future__ import annotations

import time
import json
import logging
import math
import random
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from vis_interact.agents.user_sim import (
    VisualFeedback,
    ask_user_vis as _ask_user_vis_fn,
)
from vis_interact.utils.image import encode_image_to_base64
from vis_interact.config import settings
from vis_interact.tools.registry import (
    ToolContext,
    ToolRegistry,
    TOOL_SCHEMAS,
)
from vis_interact.prompts.prompt_vis_mcts import (
    MCTS_SYSTEM_PROMPT,
    STATUS_NOTE,
    WIDEN_DIVERSITY_NOTE,
    REWARD_DECOMPOSE_PROMPT,
    build_task_prompt,
    build_context_suffix,
)
from vis_interact.utils.llm import generate_reply_api
from vis_interact.utils.dataset import get_database_detailed_schema, CODE_STRUCTURE

LOG_FMT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(LOG_FMT)
    logger.addHandler(_console)


def _lp(sample_id: str, rollout: int, phase: str) -> str:
    """Uniform log prefix: [sample_id] [R{rollout+1}] [{phase}]."""
    return f"[{sample_id}] [R{rollout + 1}] [{phase}]"


# ── MCTS Tool Schemas (subset from registry) ─────────────────────────────────

_MCTS_TOOL_NAMES = frozenset({
    "execute_sql", "execute_altair",
    "ask_user_text", "finish",
})

MCTS_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    s for s in TOOL_SCHEMAS if s["function"]["name"] in _MCTS_TOOL_NAMES
]


# ── Dimension-Aware Reward Decomposition ─────────────────────────────────────

DIMENSION_WEIGHTS = {"data": 0.4, "vis": 0.4, "intent": 0.2}

DIMENSION_MAP = {
    "execute_sql":    "data",
    "execute_altair": "vis",
    "ask_user_text":  "intent",
}

ACTION_SUCCESSORS: Dict[str, frozenset] = {
    "root":           frozenset({"execute_sql", "execute_altair", "ask_user_text"}),
    "execute_sql":    frozenset({"execute_sql", "execute_altair", "ask_user_text"}),
    "execute_altair": frozenset({"execute_sql", "execute_altair", "finish", "ask_user_text"}),
    "ask_user_text":  frozenset({"execute_sql", "execute_altair", "ask_user_text"}),
    "finish":         frozenset(),
}


def _image_url_block(img_bytes: bytes) -> dict:
    b64 = encode_image_to_base64(img_bytes)
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class AgentResult:
    """MCTS Agent execution result."""

    sample_id: str
    final_code: Optional[str] = None
    error_message: Optional[str] = None
    total_nodes: int = 0
    total_rollouts: int = 0
    text_questions_asked: int = 0
    vis_questions_asked: int = 0
    best_score: Optional[int] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class MCTSNode:
    """A node in the MCTS search tree. Each node corresponds to one tool call."""

    node_id: int
    parent: Optional[MCTSNode] = None
    children: List[MCTSNode] = field(default_factory=list)

    thought: str = ""
    content: str = ""
    action_name: str = ""
    action_args: dict = field(default_factory=dict)
    observation: Optional[str] = None
    tool_call_id: Optional[str] = None

    visits: int = 0
    total_score: float = 0.0

    user_score: Optional[int] = None
    user_feedback: Optional[str] = None
    dimension_scores: Optional[Dict[str, float]] = None

    internal_status: str = "ok"
    code: Optional[str] = None
    chart_path: Optional[str] = None
    rollout_id: int = -1

    is_terminal: bool = False
    is_exhausted: bool = False
    is_root: bool = False

    @property
    def depth(self) -> int:
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d

    @property
    def avg_score(self) -> float:
        return self.total_score / self.visits if self.visits > 0 else 0.0

    def path_to_root(self) -> List[MCTSNode]:
        path: List[MCTSNode] = []
        node: Optional[MCTSNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


# ── MCTS Agent ───────────────────────────────────────────────────────────────


class MCTSAgent:
    """MCTS-guided interactive Text-to-Vis agent with Progressive Widening."""

    def __init__(
        self,
        model: str = "",
        user_vis_model: str = "",
        user_text_model: str = "",
        b_vis: int = 10,
        b_text: int = 10,
        n_rollout: int = 5,
        c_puct: float = 1.414,
        c_pw: float = 2.0,
        alpha_pw: float = 0.5,
        max_children: int = 5,
        max_steps: int = 20,
        temperature: float = 0.7,
    ):
        self.model = model or settings.models.default_model
        self.user_vis_model = user_vis_model or settings.models.user_vis_model
        self.user_text_model = user_text_model or settings.models.user_text_model
        self.b_vis_total = b_vis
        self.b_text_total = b_text
        self.n_rollout = n_rollout
        self.c_puct = c_puct
        self.c_pw = c_pw
        self.alpha_pw = alpha_pw
        self.max_children_cap = max_children
        self.max_steps = max_steps
        self.temperature = temperature

    # ── Progressive Widening helpers ──────────────────────────────────────────

    def _max_children(self, node: MCTSNode) -> int:
        """Max children allowed for *node* under progressive widening.

        Formula: min(ceil(C_pw * N^alpha), cap).
        With defaults (C=2, alpha=0.5): visits 1→2, 4→4, 9+→5 (capped).
        """
        n = max(node.visits, 1)
        return min(
            math.ceil(self.c_pw * n ** self.alpha_pw),
            self.max_children_cap,
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, sample: dict, sample_dir: Optional[Path] = None) -> AgentResult:
        """Main entry point: run MCTS for one sample."""
        sample_id = sample["sample_id"]
        db_id = sample["db_id"]
        query = sample["initial_question"]
        key_features = sample.get("key_features", [])
        gt_code = sample.get("ground_truth_code", "")
        db_tables = get_database_detailed_schema(db_id)
        work_dir = Path(sample_dir) if sample_dir else Path(tempfile.mkdtemp())
        work_dir.mkdir(parents=True, exist_ok=True)

        ctx = _RunContext(
            sample_id=sample_id,
            db_id=db_id,
            query=query,
            db_tables=db_tables,
            key_features=key_features,
            gt_code=gt_code,
            gt_chart_path=sample.get("gt_chart_path"),
            work_dir=work_dir,
            b_vis=self.b_vis_total,
            b_text=self.b_text_total,
        )
        result = AgentResult(sample_id=sample_id)

        def _client_tag(m: str) -> str:
            if "/" in m:
                return "OpenRouter"
            if m.startswith("gpt"):
                return "OpenAI"
            if m.startswith("qwen"):
                return "Qwen"
            if m.startswith("gemini"):
                return "Gemini"
            return "Unknown"

        logger.info(
            f"[{sample_id}] Model config: "
            f"agent={self.model} ({_client_tag(self.model)}), "
            f"user_vis={self.user_vis_model} ({_client_tag(self.user_vis_model)}), "
            f"user_text={self.user_text_model} ({_client_tag(self.user_text_model)})"
        )
        logger.info(
            f"[{sample_id}] MCTS Agent start: "
            f"b_vis={self.b_vis_total}, b_text={self.b_text_total}, "
            f"n_rollout={self.n_rollout}, "
            f"c_pw={self.c_pw}, alpha_pw={self.alpha_pw}"
        )

        root = MCTSNode(node_id=0, is_root=True, thought="Root")

        for rollout_idx in range(self.n_rollout):
            ctx.start_rollout(rollout_idx)
            lp = lambda phase: _lp(sample_id, rollout_idx, phase)
            logger.info(
                f"{lp('Start')} === Rollout {rollout_idx + 1}/{self.n_rollout} === "
                f"(B_vis={ctx.b_vis}, B_text={ctx.b_text})"
            )

            try:
                # 1. Selection + Progressive Widening
                node = self._select(root, ctx, result)
                if node.is_exhausted:
                    logger.info(f"{lp('Select')} tree exhausted, skip")
                    continue

                parent_id = node.parent.node_id if node.parent else "-"
                logger.info(
                    f"{lp('Select')} → node_{node.node_id} "
                    f"(parent=node_{parent_id}, depth={node.depth}, "
                    f"action={node.action_name or 'root'})"
                )

                # 2. Simulation (rollout to terminal)
                if not node.is_terminal:
                    leaf = self._simulate(node, ctx, result)
                else:
                    leaf = node

                # 3. Evaluation
                score, feedback = self._evaluate(leaf, ctx, result)

                # 4. Reward Decomposition
                reward_vec = self._decompose_reward(score, feedback, leaf, ctx, result)

                # 5. Backpropagation (dimension-aware)
                self._backpropagate(leaf, reward_vec, self._max_children)

            except Exception:
                logger.error(
                    f"{lp('Error')} Rollout {rollout_idx + 1} failed, "
                    f"continuing with remaining rollouts",
                    exc_info=True,
                )

            result.total_rollouts = rollout_idx + 1

            if ctx.b_vis <= 0:
                logger.info(f"{lp('Budget')} B_vis exhausted, stop")
                break

        best = self._best_leaf(root)
        if best and best.code:
            result.final_code = best.code
            result.best_score = best.user_score
            logger.info(
                f"[{sample_id}] Best leaf: node_{best.node_id}, "
                f"score={best.user_score}"
            )
        else:
            result.error_message = "No valid visualization produced"
            logger.warning(f"[{sample_id}] No valid output found")

        result.text_questions_asked = self.b_text_total - ctx.b_text
        result.vis_questions_asked = self.b_vis_total - ctx.b_vis
        result.total_nodes = ctx._id_counter

        self._log_tree(sample_id, root)
        return result

    # ── Selection with Progressive Widening ───────────────────────────────────

    def _select(
        self, root: MCTSNode, ctx: _RunContext, result: AgentResult,
    ) -> MCTSNode:
        """Walk the tree from *root*, widening where allowed.

        At each node, if the number of children is below the progressive
        widening limit, attempt to add a new child via ``_widen``.  Otherwise
        descend via UCT.  Returns the node to start simulation from.
        """
        node = root
        while True:
            if node.is_terminal or node.is_exhausted:
                return node

            if node.depth >= self.max_steps:
                return node

            max_ch = self._max_children(node)

            if len(node.children) < max_ch:
                child = self._widen(node, ctx, result)
                if child is not None:
                    return child

            available = [c for c in node.children if not c.is_exhausted]
            if not available:
                node.is_exhausted = True
                return node

            unvisited = [c for c in available if c.visits == 0]
            if unvisited:
                return random.choice(unvisited)

            node = max(available, key=lambda c: self._uct(c))

    def _uct(self, node: MCTSNode) -> float:
        if node.visits == 0:
            return float("inf")
        exploit = node.total_score / node.visits
        parent_visits = node.parent.visits if node.parent else 1
        explore = self.c_puct * math.sqrt(
            math.log(max(parent_visits, 1)) / node.visits
        )
        return exploit + explore

    # ── Message Building ──────────────────────────────────────────────────────

    def _build_messages(self, node: MCTSNode, ctx: _RunContext) -> List[dict]:
        """Build OpenAI-format conversation from root to *node*.

        Only the most recent chart image is embedded; earlier charts use
        text-only observations to avoid prompt-token bloat.
        """
        system = MCTS_SYSTEM_PROMPT.format(
            code_structure=CODE_STRUCTURE.replace("{db_id}", ctx.db_id),
        )
        user = build_task_prompt(ctx.query, ctx.db_id, ctx.db_tables)
        messages: List[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        path = node.path_to_root()

        last_chart_idx = -1
        for i, n in enumerate(path[1:]):
            if n.chart_path and n.action_name == "execute_altair":
                img_file = ctx.work_dir / n.chart_path
                if img_file.exists():
                    last_chart_idx = i

        for i, n in enumerate(path[1:]):
            if n.action_name:
                tc_id = n.tool_call_id or f"call_{n.node_id}"
                messages.append({
                    "role": "assistant",
                    "content": n.content or None,
                    "tool_calls": [{
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": n.action_name,
                            "arguments": json.dumps(
                                n.action_args, ensure_ascii=False,
                            ),
                        },
                    }],
                })
                if n.observation is not None:
                    content: Any = n.observation
                    if i == last_chart_idx:
                        img_file = ctx.work_dir / n.chart_path
                        try:
                            content = [
                                {"type": "text", "text": n.observation},
                                _image_url_block(img_file.read_bytes()),
                            ]
                        except Exception:
                            pass
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": content,
                    })
            else:
                messages.append({
                    "role": "assistant",
                    "content": n.content,
                })

        return messages

    # ── Step urgency hints ─────────────────────────────────────────────────────

    @staticmethod
    def _urgency_hint(steps_remaining: int, parent_action: str) -> str:
        can_finish = (parent_action == "execute_altair")
        if steps_remaining <= 2:
            if can_finish:
                return (
                    "\n\n**⚠️ FINAL STEP — You MUST call `finish` NOW "
                    "to submit your visualization.**"
                )
            return (
                "\n\n**⚠️ FINAL STEPS — You MUST call `execute_altair` NOW "
                "with your complete Altair code to render the chart.**"
            )
        if steps_remaining <= 4:
            return (
                "\n\n**⚠️ URGENT — Only {} steps left. Call `execute_altair` "
                "to render your chart, then `finish` to submit. "
                "Do NOT call `execute_sql`.**"
                .format(steps_remaining)
            )
        return (
            "\n\n**⚠️ WARNING — Only {} steps left. Stop exploring data and "
            "start building the visualization NOW.**"
            .format(steps_remaining)
        )

    # ── Single LLM Step (shared by _widen and _simulate) ──────────────────────

    def _single_llm_step(
        self,
        node: MCTSNode,
        ctx: _RunContext,
        result: AgentResult,
        extra_messages: Optional[List[dict]] = None,
    ) -> Optional[MCTSNode]:
        """One LLM call -> parse tool call -> execute tool -> create child node.

        *extra_messages* (e.g. diversity prompt) are appended after the status
        note.  Returns the new child, or ``None`` on failure.
        """
        lp = _lp(ctx.sample_id, ctx.current_rollout, "Step")

        parent_action = node.action_name or "root"
        allowed_actions = set(ACTION_SUCCESSORS.get(parent_action, _MCTS_TOOL_NAMES))
        if parent_action == "execute_altair" and node.internal_status != "ok":
            allowed_actions.discard("finish")
        allowed_tools = [
            s for s in MCTS_TOOL_SCHEMAS
            if s["function"]["name"] in allowed_actions
        ]
        if not allowed_tools:
            return None

        messages = self._build_messages(node, ctx)
        steps_remaining = self.max_steps - node.depth
        urgency = self._urgency_hint(steps_remaining, parent_action) if steps_remaining <= 6 else ""

        has_rendered = any(
            n.action_name == "execute_altair" and n.internal_status == "ok" and n.code
            for n in node.path_to_root() if n.action_name
        )
        if has_rendered and not urgency and "finish" in allowed_actions:
            urgency = (
                "\n\nYou have already rendered a chart successfully. "
                "Call `finish` now unless there is a specific issue to fix."
            )
        context_suffix = build_context_suffix(
            ctx.global_qa, ctx.global_feedbacks,
        )
        status_text = STATUS_NOTE.format(
            steps_used=node.depth,
            max_steps=self.max_steps,
            steps_remaining=steps_remaining,
            b_text=ctx.b_text,
            urgency=urgency,
        )
        if context_suffix:
            status_text = f"{context_suffix}\n\n{status_text}"
        messages.append({"role": "user", "content": status_text})
        if extra_messages:
            messages.extend(extra_messages)
        
        time_start = time.time()
        resp = generate_reply_api(
            messages,
            model=self.model,
            temperature=self.temperature,
            tools=allowed_tools,
            tool_choice="auto",
            log_prefix=lp,
        )
        time_end = time.time()

        result.prompt_tokens += resp.prompt_tokens
        result.completion_tokens += resp.completion_tokens
        result.total_tokens += resp.total_tokens

        if not resp.success or not resp.tool_calls:
            return None

        tc = resp.tool_calls[0]
        thought = (resp.thinking or "").strip()
        content = (resp.content or "").strip()
        if thought:
            logger.info(f"{lp} LLM thinking: {thought}")
        if content:
            logger.info(f"{lp} LLM content: {content}")

        try:
            action_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"{lp} invalid tool arguments, skipping: {tc.function.arguments!r}")
            return None

        logger.info(f"{lp} LLM (time: {time_end - time_start:.2f} seconds) returned: action={tc.function.name} args={self._summarize_args(tc.function.name, action_args)}")

        obs, status, code, chart_path = self._dispatch_tool(
            tc, ctx, depth=node.depth + 1, parent_node=node,
        )

        logger.info(f"{lp} Tool done: action={tc.function.name} status={status}")
        logger.info(f"{lp} Tool obs: {obs}")

        child = MCTSNode(
            node_id=ctx.next_id(),
            parent=node,
            thought=thought,
            content=content,
            action_name=tc.function.name,
            action_args=action_args,
            observation=obs,
            tool_call_id=tc.id,
            internal_status=status,
            is_terminal=(tc.function.name == "finish" and status == "ok"),
            code=code,
            chart_path=chart_path,
            rollout_id=ctx.current_rollout,
        )
        node.children.append(child)

        logger.info(f"{lp} Append child node_{child.node_id} to parent node_{node.node_id}")
        return child

    # ── Widen (add one diverse child) ─────────────────────────────────────────

    def _widen(
        self, node: MCTSNode, ctx: _RunContext, result: AgentResult,
    ) -> Optional[MCTSNode]:
        """Try to add one new, non-duplicate child to *node*.

        When siblings already exist, a diversity prompt is injected so the LLM
        produces a meaningfully different action.
        """
        lp = _lp(ctx.sample_id, ctx.current_rollout, "Widen")

        extra: Optional[List[dict]] = None
        if node.children:
            lines = []
            for c in node.children:
                summary = self._summarize_args(c.action_name, c.action_args)
                lines.append(f"- {c.action_name}: {summary}")
            diversity_text = WIDEN_DIVERSITY_NOTE.format(
                existing_actions="\n".join(lines),
            )
            extra = [{"role": "user", "content": diversity_text}]
            logger.info(
                f"{lp} node_{node.node_id}: "
                f"widening ({len(node.children)} existing siblings)"
            )

        child = self._single_llm_step(node, ctx, result, extra_messages=extra)
        if child is None:
            logger.debug(f"{lp} node_{node.node_id}: widen LLM call failed")
            return None

        for sibling in node.children:
            if sibling is child:
                continue
            if self._is_duplicate(sibling, child):
                node.children.remove(child)
                logger.info(
                    f"{lp} node_{child.node_id} duplicates "
                    f"node_{sibling.node_id}, discarded"
                )
                return None

        logger.info(
            f"{lp} node_{node.node_id} → new child node_{child.node_id} "
            f"({child.action_name})"
        )
        return child

    @staticmethod
    def _is_duplicate(existing: MCTSNode, candidate: MCTSNode) -> bool:
        """Check whether *candidate* is too similar to *existing*."""
        if existing.action_name != candidate.action_name:
            return False
        name = existing.action_name
        if name == "execute_sql":
            a = existing.action_args.get("sql", "").strip().lower()
            b = candidate.action_args.get("sql", "").strip().lower()
            return a == b
        if name == "finish":
            return True
        if name == "execute_altair":
            a = existing.action_args.get("code", "")[:200]
            b = candidate.action_args.get("code", "")[:200]
            return a == b
        if name == "ask_user_text":
            a = existing.action_args.get("question", "").strip().lower()
            b = candidate.action_args.get("question", "").strip().lower()
            return a == b
        return (
            json.dumps(existing.action_args, sort_keys=True)
            == json.dumps(candidate.action_args, sort_keys=True)
        )

    # ── Simulation ───────────────────────────────────────────────────────────

    def _simulate(
        self, node: MCTSNode, ctx: _RunContext, result: AgentResult,
    ) -> MCTSNode:
        """Rollout from *node* to a terminal state.

        Reuses existing children when available (from previous rollouts);
        otherwise extends with a single LLM call.  Nodes created during
        simulation are kept in the tree for future reuse and widening.
        """
        current = node
        lp = _lp(ctx.sample_id, ctx.current_rollout, "Sim")
        logger.info(
            f"{lp} start from node_{current.node_id} "
            f"(depth={current.depth})"
        )

        while not current.is_terminal and current.depth < self.max_steps:
            if current.children:
                available = [c for c in current.children if not c.is_exhausted]
                if available:
                    current = random.choice(available)
                    args_snip = self._summarize_args(
                        current.action_name, current.action_args,
                    )
                    brief = f" [{args_snip}]" if args_snip else ""
                    logger.info(
                        f"{lp} → node_{current.node_id} "
                        f"{current.action_name}{brief} (reuse, "
                        f"depth={current.depth}/{self.max_steps})"
                    )
                    continue

            child = self._single_llm_step(current, ctx, result)
            if child is None:
                current.is_terminal = True
                current.internal_status = "code_error"
                break

            args_snip = self._summarize_args(child.action_name, child.action_args)
            brief = f" [{args_snip}]" if args_snip else ""
            logger.info(
                f"{lp} → node_{child.node_id} "
                f"{child.action_name}{brief} "
                f"(depth={child.depth}/{self.max_steps})"
            )
            current = child

        if not current.is_terminal:
            current.is_terminal = True
            self._try_recover_code(current)
        elif current.internal_status != "ok":
            self._try_recover_code(current)

        return current

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate(
        self, leaf: MCTSNode, ctx: _RunContext, result: AgentResult,
    ) -> Tuple[float, str]:
        """Evaluate a terminal leaf. Returns ``(score, feedback)``."""
        lp = _lp(ctx.sample_id, ctx.current_rollout, "Eval")

        if leaf.internal_status != "ok" or not leaf.code:
            logger.info(
                f"{lp} skip (status={leaf.internal_status}, "
                f"has_code={bool(leaf.code)})"
            )
            return 0.0, ""

        if ctx.b_vis <= 0:
            logger.info(f"{lp} skip (B_vis=0)")
            return 0.0, ""

        try:
            vis_result: VisualFeedback = _ask_user_vis_fn(
                key_features=ctx.key_features,
                gt_code=ctx.gt_code,
                ask_code=leaf.code,
                db_id=ctx.db_id,
                model=self.user_vis_model,
                gt_chart_path=ctx.gt_chart_path,
            )
        except Exception as e:
            logger.error(f"{lp} ask_user_vis failed: {e}")
            return 0.0, ""

        leaf.user_score = vis_result.score
        leaf.user_feedback = vis_result.feedback
        ctx.b_vis -= 1
        ctx.global_feedbacks.append((vis_result.score, vis_result.feedback))
        result.vis_questions_asked = self.b_vis_total - ctx.b_vis

        if leaf.chart_path:
            old_path = ctx.work_dir / leaf.chart_path
            if old_path.exists():
                tag = "finish" if leaf.action_name == "finish" else "recover"
                new_name = (
                    f"{tag}_r{ctx.current_rollout + 1}"
                    f"_score{vis_result.score}.png"
                )
                new_path = ctx.work_dir / new_name
                try:
                    old_path.rename(new_path)
                    leaf.chart_path = new_name
                except OSError:
                    pass

        logger.info(
            f"{lp} score={vis_result.score}/10, "
            f"feedback='{vis_result.feedback}'"
        )
        return float(vis_result.score), vis_result.feedback

    # ── Reward Decomposition ─────────────────────────────────────────────────

    def _decompose_reward(
        self,
        score: float,
        feedback: str,
        leaf: MCTSNode,
        ctx: _RunContext,
        result: AgentResult,
    ) -> Dict[str, float]:
        """Decompose scalar (score, feedback) into dimension-aware reward vector.

        Sends the chart image + trajectory + feedback to the LLM so it can
        assess data fidelity, visual design, and intent alignment independently.
        Then normalizes so weighted average equals original ``score / 10``.
        """
        lp = _lp(result.sample_id, leaf.rollout_id, "Decompose")
        zero_vec = {"data": 0.0, "vis": 0.0, "intent": 0.0}

        if score <= 0:
            leaf.dimension_scores = zero_vec
            return zero_vec

        trajectory = self._build_trajectory_summary(leaf, ctx)

        prompt_text = REWARD_DECOMPOSE_PROMPT.format(
            trajectory=trajectory,
            score=int(score),
            feedback=feedback,
        )

        user_content: List[Any] = [{"type": "text", "text": prompt_text}]

        chart_img = self._read_chart_image(leaf, ctx)
        if chart_img is not None:
            user_content.append(_image_url_block(chart_img))

        messages = [{"role": "user", "content": user_content}]
        resp = generate_reply_api(
            messages,
            model=self.model,
            response_format={"type": "json_object"},
            log_prefix=lp,
        )
        result.prompt_tokens += resp.prompt_tokens
        result.completion_tokens += resp.completion_tokens
        result.total_tokens += resp.total_tokens

        if resp.success and resp.content:
            try:
                parsed = json.loads(resp.content)
                raw = {
                    "data":   max(1, min(10, int(parsed.get("data_fidelity", score)))),
                    "vis":    max(1, min(10, int(parsed.get("vis_design", score)))),
                    "intent": max(1, min(10, int(parsed.get("intent_alignment", score)))),
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                raw = {"data": int(score), "vis": int(score), "intent": int(score)}
        else:
            raw = {"data": int(score), "vis": int(score), "intent": int(score)}

        raw_avg = sum(DIMENSION_WEIGHTS[d] * raw[d] for d in raw)
        scale = score / raw_avg if raw_avg > 0 else 1.0

        reward_vec = {
            d: max(0.0, min(1.0, scale * raw[d] / 10.0)) for d in raw
        }
        leaf.dimension_scores = reward_vec

        logger.info(
            f"{lp} raw=({raw['data']},{raw['vis']},{raw['intent']}), "
            f"reward=(data={reward_vec['data']:.3f}, "
            f"vis={reward_vec['vis']:.3f}, "
            f"intent={reward_vec['intent']:.3f})"
        )
        return reward_vec

    @staticmethod
    def _build_trajectory_summary(leaf: MCTSNode, ctx: _RunContext) -> str:
        """Build a concise summary of the root-to-leaf action path."""
        path = leaf.path_to_root()
        lines = [f"User query: {ctx.query}"]
        for n in path[1:]:
            if not n.action_name:
                continue
            tag = n.action_name
            if tag == "execute_sql":
                sql = n.action_args.get("sql", "")
                obs_snip = (n.observation or "")[:200]
                lines.append(f"[SQL] {sql}\n  → {obs_snip}")
            elif tag == "execute_altair":
                code = n.action_args.get("code", "")
                code_lines = code.strip().splitlines()
                snip = "\n".join(code_lines[:15])
                if len(code_lines) > 15:
                    snip += f"\n  ... ({len(code_lines)} lines total)"
                status = "OK" if n.internal_status == "ok" else n.internal_status
                lines.append(f"[Altair] ({status})\n{snip}")
            elif tag == "ask_user_text":
                q = n.action_args.get("question", "")
                a = (n.observation or "")[:200]
                lines.append(f"[Ask User] Q: {q}\n  A: {a}")
            elif tag == "finish":
                lines.append("[Finish] submitted final code")
        return "\n".join(lines)

    @staticmethod
    def _read_chart_image(
        leaf: MCTSNode, ctx: _RunContext,
    ) -> Optional[bytes]:
        """Try to read the rendered chart image for this leaf."""
        if not leaf.chart_path:
            return None
        img_path = ctx.work_dir / leaf.chart_path
        if not img_path.exists():
            return None
        try:
            return img_path.read_bytes()
        except OSError:
            return None

    # ── Backpropagation ──────────────────────────────────────────────────────

    @staticmethod
    def _backpropagate(
        leaf: MCTSNode,
        reward_vec: Dict[str, float],
        max_children_fn: Callable[[MCTSNode], int],
    ):
        """Dimension-aware selective backpropagation.

        Each node only accumulates the reward dimension corresponding to its
        action type (via ``DIMENSION_MAP``).  ``finish`` and ``root`` nodes
        only increment visits — their UCT degenerates to the pure exploration
        term, which is the correct behaviour.
        """
        node: Optional[MCTSNode] = leaf
        while node is not None:
            node.visits += 1
            dim = DIMENSION_MAP.get(node.action_name)
            if dim is not None:
                node.total_score += reward_vec[dim]
            node = node.parent

        if leaf.is_terminal:
            leaf.is_exhausted = True
        cur: Optional[MCTSNode] = leaf.parent
        while cur is not None:
            if (cur.children
                    and all(c.is_exhausted for c in cur.children)
                    and len(cur.children) >= max_children_fn(cur)):
                cur.is_exhausted = True
            else:
                break
            cur = cur.parent

    # ── Tool Dispatch ────────────────────────────────────────────────────────

    def _dispatch_tool(
        self,
        tc: Any,
        ctx: _RunContext,
        depth: int = 0,
        parent_node: Optional[MCTSNode] = None,
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """Execute a tool call via ToolRegistry.

        Returns ``(observation, internal_status, code_if_any, chart_path)``.
        For ``finish``, inherits code/chart from the parent ``execute_altair``
        node instead of re-executing through the registry.
        """
        name = tc.function.name

        if name == "finish":
            if (parent_node is not None
                    and parent_node.action_name == "execute_altair"
                    and parent_node.code
                    and parent_node.internal_status == "ok"):
                return (
                    "FINISH: Submitted final visualization code.",
                    "ok", parent_node.code, parent_node.chart_path,
                )
            return (
                "FINISH FAILED: No prior successful execute_altair found.",
                "code_error", None, None,
            )

        try:
            args: dict = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            return "ERROR: Invalid tool arguments", "code_error", None, None

        code: Optional[str] = None
        chart_path: Optional[str] = None
        status = "ok"

        tool_ctx = ctx.make_tool_context()
        tool_ctx.model = self.model
        tool_ctx.user_text_model = self.user_text_model

        rollout_display = ctx.current_rollout + 1
        if name == "execute_altair":
            tool_ctx._image_path_override = (
                f"chart_temp_r{rollout_display}_d{depth}.png"
            )

        registry = ToolRegistry(tool_ctx, exclude={"ask_user_vis"})

        try:
            if name == "ask_user_text":
                with ctx.lock:
                    if ctx.b_text <= 0:
                        return (
                            "BUDGET EXCEEDED: No text-question budget remaining.",
                            "ok", None, None,
                        )
                    tool_ctx.text_budget_remaining = 1
                    obs = registry.dispatch(tc)
                    if tool_ctx.path_qa_history:
                        ctx.b_text -= 1
                        ctx.global_qa.extend(tool_ctx.path_qa_history)
            else:
                obs = registry.dispatch(tc)

            if name == "execute_sql":
                if obs.startswith("SQL ERROR") or obs.startswith("Query returned 0 rows"):
                    status = "sql_empty"
            elif name == "execute_altair":
                if "FAILED" in obs or "ERROR" in obs:
                    status = "render_fail"
                else:
                    code = args.get("code")
                    chart_path = (
                        Path(tool_ctx.generated_images[-1]).name
                        if tool_ctx.generated_images else None
                    )
                    if code and chart_path and tool_ctx.work_dir:
                        code_path = tool_ctx.work_dir / Path(chart_path).with_suffix(".py").name
                        code_path.write_text(code, encoding="utf-8")

        except Exception as e:
            obs = f"ERROR: {e}"
            status = "code_error"

        return obs, status, code, chart_path

    # ── Output Selection ─────────────────────────────────────────────────────

    def _best_leaf(self, root: MCTSNode) -> Optional[MCTSNode]:
        terminals: List[MCTSNode] = []
        self._collect_terminals(root, terminals)

        evaluated = [n for n in terminals if n.user_score is not None]
        if evaluated:
            best_score = max(n.user_score for n in evaluated)
            top = [n for n in evaluated if n.user_score == best_score]
            return max(top, key=lambda n: n.rollout_id)

        with_code = [n for n in terminals if n.code and n.internal_status == "ok"]
        return with_code[-1] if with_code else None

    def _collect_terminals(self, node: MCTSNode, out: list):
        if node.is_terminal:
            out.append(node)
        for child in node.children:
            self._collect_terminals(child, out)

    @staticmethod
    def _try_recover_code(node: MCTSNode):
        """Walk up from a forced-terminal node to find the last valid code."""
        current: Optional[MCTSNode] = node
        while current is not None:
            if current.code and current.internal_status == "ok":
                node.code = current.code
                node.chart_path = current.chart_path
                node.internal_status = "ok"
                return
            current = current.parent

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _summarize_args(action_name: str, args: dict) -> str:
        if action_name == "execute_sql":
            sql = args.get("sql", "")
            return (sql)
        elif action_name == "execute_altair":
            code = args.get("code", "")
            lines = code.strip().splitlines()
            if len(lines) <= 3:
                return code.strip().replace("\n", " | ")
            return f"{lines[0].strip()} ... ({len(lines)} lines)"
        elif action_name == "finish":
            return "(submit previous execute_altair code)"
        elif action_name == "ask_user_text":
            return args.get("question", "")
        return json.dumps(args, ensure_ascii=False)

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_tree(self, sample_id: str, root: MCTSNode, indent: int = 0):
        prefix = "  " * indent
        if root.user_score is not None:
            score_s = f"score={root.user_score}"
            if root.dimension_scores:
                ds = root.dimension_scores
                score_s += (
                    f" d={ds['data']:.2f}"
                    f" v={ds['vis']:.2f}"
                    f" i={ds['intent']:.2f}"
                )
        elif root.visits > 0:
            dim = DIMENSION_MAP.get(root.action_name)
            dim_s = f" ({dim})" if dim else ""
            score_s = f"Q{dim_s}={root.avg_score:.2f}"
        else:
            score_s = "N/A"

        label = root.action_name or "root"
        parent_s = f"p={root.parent.node_id}" if root.parent else "p=-"
        r_s = f"R{root.rollout_id + 1}" if root.rollout_id >= 0 else "-"
        ch_s = f"ch={len(root.children)}"
        info = (
            f"{prefix}[{root.node_id}] {label} "
            f"({parent_s}, {r_s}, N={root.visits}, {ch_s}, {score_s})"
        )
        if root.is_terminal:
            info += " *"
        if root.is_exhausted:
            info += " X"
        logger.info(f"[{sample_id}] TREE {info}")

        for child in root.children:
            self._log_tree(sample_id, child, indent + 1)


# ── Run Context (per-sample mutable state) ───────────────────────────────────


@dataclass
class _RunContext:
    """Mutable state shared across all rollouts within a single sample run."""

    sample_id: str
    db_id: str
    query: str
    db_tables: str
    key_features: list
    gt_code: str
    gt_chart_path: Optional[str] = None
    work_dir: Path = field(default_factory=lambda: Path("."))

    b_vis: int = 5
    b_text: int = 3

    global_qa: List[Tuple[str, str]] = field(default_factory=list)
    global_feedbacks: List[Tuple[int, str]] = field(default_factory=list)

    _id_counter: int = 0
    current_rollout: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def next_id(self) -> int:
        with self.lock:
            self._id_counter += 1
            return self._id_counter

    def start_rollout(self, rollout_id: int):
        with self.lock:
            self.current_rollout = rollout_id

    def make_tool_context(self) -> ToolContext:
        return ToolContext(
            sample_id=self.sample_id,
            db_id=self.db_id,
            key_features=self.key_features,
            gt_code=self.gt_code,
            gt_chart_path=self.gt_chart_path,
            work_dir=self.work_dir,
            text_budget_remaining=999,
            vis_budget_remaining=999,
        )
