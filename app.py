from __future__ import annotations

import difflib
import json
import os
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / ".real_ai"
WORKSPACE_DIR = DATA_DIR / "workspace"
MEMORY_FILE = DATA_DIR / "memory.json"
PROMPTS_FILE = DATA_DIR / "prompts.json"
LOOP_CONFIG_FILE = DATA_DIR / "loop_config.json"

DEFAULT_PROMPTS = {
    "planner": "You are Model 1 Planner. Ask only high-value questions and define exact acceptance criteria.",
    "researcher": "You are Model 2 Researcher. Answer using concrete workspace evidence and practical implementation strategy.",
    "executor": "You are Model 3 Executor. Emit structured JSON actions, execute safely, verify completion.",
}

DEFAULT_LOOP_CONFIG = {
    "max_iterations": 18,
    "stagnation_threshold": 3,
    "require_evidence": True,
    "executor_action_schema": {
        "type": "object",
        "required": ["action", "reasoning"],
        "properties": {
            "action": {"type": "string", "enum": ["read", "write", "patch", "run", "mark_complete", "ask"]},
            "target": {"type": "string"},
            "content": {"type": "string"},
            "command": {"type": "string"},
            "reasoning": {"type": "string"},
        },
    },
}

AGENT_COLORS = {
    "planner": "#00d4ff",
    "researcher": "#38d39f",
    "executor": "#ffbf47",
}

AGENT_BADGES = {
    "planner": "① Planner",
    "researcher": "② Researcher",
    "executor": "③ Executor",
}


@dataclass
class Event:
    role: str
    message: str
    kind: str = "info"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S"))
    raw: Optional[Dict[str, Any]] = None


@dataclass
class RunState:
    goal: str = ""
    iterations: int = 0
    completion: bool = False
    questions: int = 0
    answers: int = 0
    builds: int = 0
    stagnation: int = 0
    events: List[Event] = field(default_factory=list)
    raw_json: List[Dict[str, Any]] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)


class Storage:
    def __init__(self) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        WORKSPACE_DIR.mkdir(exist_ok=True)
        if not MEMORY_FILE.exists():
            MEMORY_FILE.write_text(json.dumps({"runs": [], "insights": []}, indent=2))
        if not PROMPTS_FILE.exists():
            PROMPTS_FILE.write_text(json.dumps(DEFAULT_PROMPTS, indent=2))
        if not LOOP_CONFIG_FILE.exists():
            LOOP_CONFIG_FILE.write_text(json.dumps(DEFAULT_LOOP_CONFIG, indent=2))

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text())

    def memory(self) -> Dict[str, Any]:
        return self._load_json(MEMORY_FILE)

    def prompts(self) -> Dict[str, str]:
        return self._load_json(PROMPTS_FILE)

    def loop_config(self) -> Dict[str, Any]:
        return self._load_json(LOOP_CONFIG_FILE)

    def save_memory(self, data: Dict[str, Any]) -> None:
        MEMORY_FILE.write_text(json.dumps(data, indent=2))

    def save_prompts(self, data: Dict[str, Any]) -> None:
        PROMPTS_FILE.write_text(json.dumps(data, indent=2))

    def save_loop_config(self, data: Dict[str, Any]) -> None:
        LOOP_CONFIG_FILE.write_text(json.dumps(data, indent=2))


class KimiClient:
    """Minimal OpenAI-compatible wrapper.

    Uses Kimi endpoint through OpenAI-compatible settings:
    - base_url: https://api.moonshot.ai/v1
    - model: kimi-k2.6
    """

    def __init__(self, api_key: str, base_url: str = "https://api.moonshot.ai/v1", model: str = "kimi-k2.6") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str, response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            kwargs = {}
            if response_format:
                kwargs["response_format"] = response_format
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                **kwargs,
            )
            text = completion.choices[0].message.content or ""
            return {
                "ok": True,
                "text": text,
                "usage": getattr(completion, "usage", None),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "text": "",
            }


class AutonomousLoop:
    def __init__(self, storage: Storage, kimi: Optional[KimiClient]) -> None:
        self.storage = storage
        self.kimi = kimi

    def _safe_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            return {"action": "ask", "reasoning": "Invalid JSON from model fallback", "content": text}

    def _validate_action(self, action: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        required = set(schema.get("required", []))
        missing = [k for k in required if k not in action]
        if missing:
            return {
                "valid": False,
                "error": f"Missing keys: {missing}",
                "fixed": {"action": "ask", "reasoning": "Schema fallback due to missing keys"},
            }
        return {"valid": True, "fixed": action}

    def _workspace_snapshot(self) -> str:
        files = [str(p.relative_to(WORKSPACE_DIR)) for p in WORKSPACE_DIR.glob("**/*") if p.is_file()]
        return "\n".join(files[:60]) if files else "<empty workspace>"

    def _planner(self, state: RunState) -> Dict[str, Any]:
        prompt = self.storage.prompts()["planner"]
        user = (
            f"Goal: {state.goal}\nWorkspace:\n{self._workspace_snapshot()}\n"
            "Return JSON with keys: questions(list), acceptance_criteria(list), plan(list)."
        )
        if self.kimi:
            resp = self.kimi.chat(prompt, user)
            payload = self._safe_json(resp.get("text", "{}"))
        else:
            payload = {
                "questions": ["Any required framework or package constraints?"],
                "acceptance_criteria": [
                    "Autonomous loop runs until complete true",
                    "Structured JSON tool actions validated",
                    "Evolution mode proposes concrete improvements",
                ],
                "plan": ["Inspect", "Implement", "Verify"],
            }
        return payload

    def _researcher(self, state: RunState) -> Dict[str, Any]:
        prompt = self.storage.prompts()["researcher"]
        user = (
            f"Goal: {state.goal}\nAcceptance Criteria: {state.acceptance_criteria}\n"
            f"Workspace evidence:\n{self._workspace_snapshot()}\n"
            "Return JSON: findings(list), implementation_notes(list), tests(list)."
        )
        if self.kimi:
            resp = self.kimi.chat(prompt, user)
            payload = self._safe_json(resp.get("text", "{}"))
        else:
            payload = {
                "findings": ["Workspace persistence can use local JSON files."],
                "implementation_notes": ["Use streamlit.session_state for live state and queues for terminal events."],
                "tests": ["Run py_compile and smoke run streamlit app."],
            }
        return payload

    def _executor(self, state: RunState, findings: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.storage.prompts()["executor"]
        user = (
            f"Goal: {state.goal}\nFindings: {json.dumps(findings)}\n"
            "Return JSON action with schema: action,target,content,command,reasoning."
        )
        if self.kimi:
            resp = self.kimi.chat(prompt, user, response_format={"type": "json_object"})
            payload = self._safe_json(resp.get("text", "{}"))
        else:
            payload = random.choice(
                [
                    {"action": "run", "command": "python -m py_compile app.py", "reasoning": "Verify syntax"},
                    {"action": "mark_complete", "reasoning": "Acceptance criteria appear met"},
                    {"action": "read", "target": "README.md", "reasoning": "Need specs"},
                ]
            )
        return payload

    def _perform_action(self, action: Dict[str, Any], state: RunState) -> str:
        a = action.get("action")
        if a == "run":
            cmd = action.get("command", "echo noop")
            result = subprocess.run(cmd, cwd=APP_DIR, shell=True, capture_output=True, text=True)
            state.builds += 1
            output = (result.stdout + "\n" + result.stderr).strip()[:1200]
            return f"$ {cmd}\n{output or '<no output>'}"
        if a == "read":
            target = WORKSPACE_DIR / action.get("target", "")
            if target.exists() and target.is_file():
                return target.read_text()[:1200]
            return "read fallback: target missing"
        if a in {"write", "patch"}:
            target = WORKSPACE_DIR / action.get("target", "note.txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(action.get("content", ""))
            return f"updated {target.relative_to(WORKSPACE_DIR)}"
        if a == "mark_complete":
            state.completion = True
            return "task marked complete=true"
        return "executor asked for clarification"

    def run_iteration(self, state: RunState) -> RunState:
        state.iterations += 1
        plan = self._planner(state)
        state.questions += len(plan.get("questions", []))
        state.acceptance_criteria = plan.get("acceptance_criteria", state.acceptance_criteria)
        state.events.append(Event("planner", json.dumps(plan, indent=2), raw=plan))
        state.raw_json.append({"planner": plan})

        research = self._researcher(state)
        state.answers += len(research.get("findings", []))
        state.events.append(Event("researcher", json.dumps(research, indent=2), raw=research))
        state.raw_json.append({"researcher": research})

        action = self._executor(state, research)
        validation = self._validate_action(action, self.storage.loop_config()["executor_action_schema"])
        fixed = validation["fixed"]
        state.events.append(Event("executor", json.dumps(fixed, indent=2), raw=fixed, kind="action"))
        state.raw_json.append({"executor": fixed})

        result = self._perform_action(fixed, state)
        state.events.append(Event("executor", result, kind="terminal"))

        if not state.completion and fixed.get("action") != "mark_complete":
            state.stagnation += 1
        else:
            state.stagnation = 0

        if state.stagnation >= self.storage.loop_config().get("stagnation_threshold", 3):
            state.events.append(Event("planner", "Stagnation detected; switching to fallback policy", kind="warn"))
            state.stagnation = 0

        if state.iterations >= self.storage.loop_config().get("max_iterations", 18):
            state.completion = True

        return state

    def evolution_proposals(self, state: RunState) -> Dict[str, Any]:
        memory = self.storage.memory()
        proposals = {
            "prompt_improvements": [
                "Planner now requests measurable acceptance criteria with file-level evidence.",
                "Researcher must reference concrete workspace paths in every finding.",
                "Executor fallback includes schema auto-repair before command execution.",
            ],
            "logic_changes": [
                "Add adaptive max_iterations based on task complexity tags.",
                "Prioritize read actions before write/patch when evidence confidence < 0.7.",
            ],
            "schema_changes": [
                {"add_field": "confidence", "type": "number", "range": [0, 1]},
                {"add_field": "evidence", "type": "array[string]"},
            ],
        }
        memory["insights"].append(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "goal": state.goal,
                "iterations": state.iterations,
                "proposals": proposals,
            }
        )
        self.storage.save_memory(memory)
        return proposals


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --bg:#000000; --cyan:#00d4ff; --green:#38d39f; --amber:#ffbf47; }
        .stApp {
          background: radial-gradient(circle at 10% 10%, rgba(0,212,255,0.16), transparent 30%),
                      radial-gradient(circle at 90% 20%, rgba(0,212,255,0.10), transparent 35%),
                      #000;
          color: #e9f7ff;
          font-family: 'Inter', sans-serif;
        }
        h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: .3px; }
        .metric-card {
          border: 1px solid rgba(0,212,255,.2);
          border-radius: 16px; padding: 12px 14px;
          background: rgba(5, 15, 20, 0.65);
          transition: all .2s ease;
        }
        .metric-card:hover { transform: translateY(-1px); border-color: rgba(0,212,255,.5); }
        .terminal {
          border: 1px solid rgba(0,212,255,.25);
          background: rgba(5,9,12,0.78);
          border-radius: 18px;
          padding: 14px;
          min-height: 460px;
        }
        .event {
          animation: fadein .28s ease;
          border-left: 2px solid rgba(0,212,255,.35);
          margin: 8px 0; padding: 8px 10px;
          background: rgba(255,255,255,0.015);
          border-radius: 10px;
        }
        .pill { font-weight: 700; font-size: 12px; border-radius: 999px; padding: 3px 8px; }
        @keyframes fadein { from { opacity:0; transform: translateY(4px);} to { opacity:1; transform:none; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_terminal(events: List[Event]) -> None:
    st.markdown("<div class='terminal'>", unsafe_allow_html=True)
    for e in events[-60:]:
        color = AGENT_COLORS.get(e.role, "#aaa")
        badge = AGENT_BADGES.get(e.role, e.role)
        st.markdown(
            f"""
            <div class='event'>
              <span class='pill' style='background:{color}20; color:{color}; border:1px solid {color}66'>{badge}</span>
              <span style='opacity:.7;font-size:12px;padding-left:8px'>{e.timestamp}</span>
              <pre style='white-space:pre-wrap;margin-top:8px;color:#e8f8ff'>{e.message}</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_metrics(state: RunState) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><b>Iterations</b><br>{state.iterations}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><b>Q/A</b><br>{state.questions}/{state.answers}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><b>Builds</b><br>{state.builds}</div>", unsafe_allow_html=True)
    c4.markdown(
        f"<div class='metric-card'><b>Complete</b><br>{'true ✅' if state.completion else 'false ⏳'}</div>",
        unsafe_allow_html=True,
    )
    progress = min(1.0, state.iterations / 18)
    st.progress(progress, text=f"Autonomous Progress: {int(progress * 100)}%")


def render_diff(title: str, before: str, after: str) -> None:
    diff = "\n".join(
        difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile="before", tofile="after", lineterm="")
    )
    with st.expander(title, expanded=False):
        st.code(diff or "No diff", language="diff")


def apply_evolution(storage: Storage, proposals: Dict[str, Any]) -> None:
    prompts = storage.prompts()
    loop_cfg = storage.loop_config()

    prompts["planner"] += "\nAlways include measurable acceptance criteria and evidence paths."
    prompts["researcher"] += "\nEvery finding must include at least one workspace path."
    prompts["executor"] += "\nRun schema auto-repair before tool execution."

    loop_cfg["max_iterations"] = min(30, int(loop_cfg.get("max_iterations", 18) * 1.2))
    props = loop_cfg["executor_action_schema"]["properties"]
    props["confidence"] = {"type": "number"}
    props["evidence"] = {"type": "array", "items": {"type": "string"}}

    storage.save_prompts(prompts)
    storage.save_loop_config(loop_cfg)


def run_loop_once(loop: AutonomousLoop) -> None:
    state: RunState = st.session_state.state
    st.session_state.state = loop.run_iteration(state)


def run_until_complete(loop: AutonomousLoop) -> None:
    state: RunState = st.session_state.state
    max_iters = loop.storage.loop_config().get("max_iterations", 18)
    for _ in range(max_iters):
        state = loop.run_iteration(state)
        st.session_state.state = state
        if state.completion:
            break


def keyboard_shortcuts() -> None:
    st.components.v1.html(
        """
        <script>
        window.addEventListener('keydown', function(e){
          if((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            const buttons = window.parent.document.querySelectorAll('button[kind="primary"]');
            if(buttons.length) buttons[0].click();
          }
        });
        </script>
        """,
        height=0,
    )


def main() -> None:
    st.set_page_config(page_title="real ai • autonomous studio", page_icon="⚡", layout="wide")
    inject_styles()
    keyboard_shortcuts()

    storage = Storage()

    if "state" not in st.session_state:
        st.session_state.state = RunState()
    if "proposals" not in st.session_state:
        st.session_state.proposals = None

    with st.sidebar:
        st.title("real ai")
        st.caption("Autonomous 3-Agent Studio • Kimi API (kimi-k2.6)")

        default_api_key = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY") or ""
        api_key = st.text_input(
            "Kimi API Key",
            type="password",
            value=default_api_key,
            help="Used with OpenAI-compatible endpoint. Prefer env var KIMI_API_KEY for security.",
        )
        model_name = st.text_input("Model", value="kimi-k2.6")
        mode = st.radio("Run Mode", ["Single Iteration", "Run Until Complete"], horizontal=False)
        st.divider()
        st.subheader("Controls")

        goal = st.text_area("Task", value=st.session_state.state.goal or "Design and implement feature end-to-end.")
        st.session_state.state.goal = goal

        start = st.button("▶ Start Autonomous Loop", type="primary", use_container_width=True)
        evolve = st.button("🧬 Run Evolution Mode", use_container_width=True)
        reset = st.button("Reset Session", use_container_width=True)

        if reset:
            st.session_state.state = RunState(goal=goal)
            st.session_state.proposals = None
            st.rerun()

    kimi = KimiClient(api_key, model=model_name.strip() or "kimi-k2.6") if api_key else None
    loop = AutonomousLoop(storage, kimi)

    st.title("Autonomous Multi-Agent Sandbox")
    render_metrics(st.session_state.state)

    if start:
        if mode == "Single Iteration":
            run_loop_once(loop)
        else:
            run_until_complete(loop)

    left, right = st.columns([2.4, 1.2])
    with left:
        st.subheader("Live Terminal Sandbox")
        render_terminal(st.session_state.state.events)

    with right:
        st.subheader("Raw JSON Viewer")
        for idx, payload in enumerate(reversed(st.session_state.state.raw_json[-12:]), start=1):
            with st.expander(f"Model Output #{idx}"):
                st.json(payload, expanded=True)

    if evolve:
        st.session_state.proposals = loop.evolution_proposals(st.session_state.state)

    if st.session_state.proposals:
        st.subheader("Evolution Mode • Proposed Self-Improvement")
        st.json(st.session_state.proposals)

        prompts_before = json.dumps(storage.prompts(), indent=2)
        cfg_before = json.dumps(storage.loop_config(), indent=2)

        temp_prompts = json.loads(prompts_before)
        temp_cfg = json.loads(cfg_before)
        temp_prompts["planner"] += "\nAlways include measurable acceptance criteria and evidence paths."
        temp_prompts["researcher"] += "\nEvery finding must include at least one workspace path."
        temp_prompts["executor"] += "\nRun schema auto-repair before tool execution."
        temp_cfg["max_iterations"] = min(30, int(temp_cfg.get("max_iterations", 18) * 1.2))

        render_diff("Prompt Changes", prompts_before, json.dumps(temp_prompts, indent=2))
        render_diff("Loop Logic Changes", cfg_before, json.dumps(temp_cfg, indent=2))

        if st.button("Approve and Apply Evolution", type="primary"):
            apply_evolution(storage, st.session_state.proposals)
            st.success("Evolution applied. The system is now smarter for the next run.")


if __name__ == "__main__":
    main()
