# ml-intern: deep exploration

`huggingface/ml-intern` (released 21 Apr 2026) — "an open-source ML engineer
that reads papers, trains models, and ships ML models." 73% Python / 27%
TypeScript. Three deployment shapes: interactive CLI, headless CLI, web app.

## 1. Repository layout

```
agent/                         core agent runtime (Python)
├── core/
│   ├── agent_loop.py     ~64 KB   the loop
│   ├── session.py                 turn state, pending approvals
│   ├── tools.py                   ToolSpec + registry
│   ├── doom_loop.py               repetition detector
│   ├── effort_probe.py            per-model reasoning-level probe
│   ├── model_switcher.py          /model command, fallback chains
│   ├── llm_params.py              provider-specific param shaping
│   ├── prompt_caching.py          Anthropic cache breakpoints
│   ├── context_manager already lives next door
│   ├── session_persistence.py     local checkpoint + resume
│   ├── session_uploader.py        sessions are uploaded to HF
│   ├── telemetry.py
│   ├── redact.py                  scrubs tokens before persistence
│   ├── hf_router_catalog.py       model-routing metadata
│   ├── hf_access.py / hf_tokens.py
├── context_manager/manager.py     ~17 KB compaction
├── prompts/                       system_prompt v1/v2/v3 (YAML, v3 ≈ 15 KB)
├── tools/                         22 tool implementations
├── messaging/                     notification gateways (Slack today)
├── sft/                           packaged SFT recipe
├── main.py                        CLI entry point (~52 KB)
└── config.py
backend/        FastAPI server (used by web app)
frontend/       Vite + React TS UI
configs/        cli_agent_config.json + variants
```

## 2. How a turn flows end-to-end

### 2.1 Entry — `agent/main.py`

```python
parser.add_argument("prompt", nargs="?", default=None,
                    help="Run headlessly with this prompt")
parser.add_argument("--model", "-m")
parser.add_argument("--max-iterations", type=int,
                    help="default 50, -1 for unlimited")
parser.add_argument("--no-stream", action="store_true")
```

If `prompt` is supplied → `headless_main` (auto-approve everything). Otherwise
`main()` runs the interactive REPL.

Both modes load `configs/cli_agent_config.json` and start a single
`asyncio.create_task(submission_loop(...))`, then push user turns onto a
`submission_queue` and drain typed `Event`s off an `event_queue`.

### 2.2 The submission loop

```python
async def submission_loop(submission_queue, event_queue, config, tool_router,
                         session_holder, hf_token, user_id, local_mode, stream,
                         notification_gateway, notification_destinations,
                         defer_turn_complete_notification): ...
```

Per iteration of `while session.is_running` it dequeues an op and dispatches:

| OpType            | Handler                                              |
|-------------------|------------------------------------------------------|
| `USER_INPUT`      | `Handlers.run_agent(session, text)` — the agentic loop |
| `COMPACT`         | force a context-compaction round                     |
| `UNDO`            | drop the last user turn                              |
| `EXEC_APPROVAL`   | run tool calls the user just approved                |
| `SHUTDOWN`        | save session, exit                                   |

This queue indirection is what lets the CLI, the headless runner, the test
harness, and the FastAPI backend all attach to the same agent core without
re-implementing it.

### 2.3 The agentic loop (`run_agent`)

Repeat up to `MAX_ITERATIONS = 300` (CLI default 50):

1. Compact context if `running_context_usage > 0.9 × model_max_tokens`.
2. Run doom-loop and malformed-call guards on recent history.
3. `litellm.acompletion(messages, tools=available_tools, stream=...)`.
4. Parse tool_calls.
5. Split into approval-required vs auto.
6. Execute auto tools in parallel (cancellable via `is_cancelled`).
7. If any tool needs approval: emit `approval_required`, stash `ToolCall`s on
   `session.pending_approval`, return early. The next `EXEC_APPROVAL` op
   resumes.
8. If no tool calls came back: emit `assistant_message` and finish the turn.

### 2.4 Events emitted

`processing`, `assistant_chunk`, `assistant_message`, `assistant_stream_end`,
`tool_log`, `tool_call`, `tool_state_change` (`approved` / `rejected` /
`running` / `cancelled` / `abandoned`), `tool_output`, `approval_required`,
`error`, `turn_complete`, `interrupted`, `undo_complete`, `compacted`,
`ready`, `shutdown`. The UI is a pure consumer of this stream.

### 2.5 Approval policy — `_needs_approval(name, args, config)`

| Tool                       | Rule                                                  |
|----------------------------|-------------------------------------------------------|
| `sandbox_create`           | always approval                                       |
| `hf_jobs`                  | approval unless CPU-only and `confirm_cpu_jobs=False` |
| `hf_private_repos` (upload)| approval unless `auto_file_upload=True`               |
| `hf_repo_files` upload/del | always approval                                       |
| `hf_repo_git` destructive  | always approval                                       |
| anything in yolo mode      | skipped (headless default)                            |

### 2.6 Retries and recovery

`_MAX_LLM_RETRIES = 3`. Rate-limit backoff `[30, 60]`s; transient (timeout /
5xx / connection) backoff `[5, 15, 30]`s. Special cases:

* `ContextWindowExceededError` → force compact, retry the iteration.
* `finish_reason == "length"` while tool-calling → drop the partial calls,
  inject "Your previous response was truncated…", let the LLM try again with
  smaller chunks (heredoc / multiple edits).
* Effort-config rejection → re-probe via `effort_probe`, or strip thinking,
  retry once.
* Invalid Anthropic thinking signature → strip metadata, retry once.

### 2.7 Context compaction — `agent/context_manager/manager.py`

* Storage: `self.items` is a list of litellm `Message`s.
* `_COMPACT_THRESHOLD_RATIO = 0.9`, `model_max_tokens` defaults to 180k → fires
  near 162k.
* Preserves verbatim: system message (idx 0), the original user task, last
  `untouched_messages` (default 5).
* Summarises everything in between via an LLM call using:

  > "Please provide a concise summary of the conversation above, focusing on
  > key decisions, the 'why' behind the decisions, problems solved, and
  > important context needed for developing further."

* Emits `compacted` with `{"old_tokens": …, "new_tokens": …}`.

### 2.8 Doom-loop detector — `agent/core/doom_loop.py`

* `lookback = 30` recent messages.
* Tool signatures are MD5(`name`, normalised-JSON args, result).
* Identical-consecutive: ≥3 same `(name, args, result)` in a row.
* Cyclic: subsequence lengths 2..5 repeating ≥2 times.
* Tracks the *result* hash too — constant args + changing results = legit
  polling, doesn't trigger.
* Intervention: injects

  > `[SYSTEM: REPETITION GUARD] You have called '{tool}' with the same
  > arguments multiple times in a row… STOP repeating this approach — it is
  > not working.`

  with suggestions to switch tools or ask the user.

## 3. The toolbelt

`ToolSpec` (`agent/core/tools.py`) — the only thing tools share:

```python
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]              # JSONSchema
    handler: Optional[Callable[[dict], Awaitable[tuple[str, bool]]]] = None
```

14 built-ins, registered conditionally:

| Group          | Tools                                                              |
|----------------|--------------------------------------------------------------------|
| Discovery      | `research`, `explore_hf_docs`, `hf_docs_fetch`, `hf_papers`, `web_search` |
| Datasets       | `hf_inspect_dataset`                                               |
| HF repos       | `hf_repo_files`, `hf_repo_git`                                     |
| Code reuse     | `github_find_examples`, `github_list_repos`, `github_read_file`    |
| Compute        | `hf_jobs` (immediate + scheduled)                                  |
| Planning / IO  | `plan`, `notify`                                                   |

Plus: sandbox tools when `local_mode=False`, local file tools when
`local_mode=True`, an OpenAPI search tool, and any MCP-server tools.

`NOT_ALLOWED_TOOL_NAMES = ["hf_jobs", "hf_doc_search", "hf_doc_fetch",
"hf_whoami"]` blocks the duplicate MCP versions of these.

### 3.1 `research` is a sub-agent, not a flat tool

Spawns its own loop with a cheaper model (e.g. `anthropic/claude-sonnet-4-6`)
and a read-only subset:

```python
RESEARCH_TOOL_NAMES = {"read", "bash", "explore_hf_docs", "fetch_hf_docs",
                       "web_search", ...}
```

Its directive: "mine the literature to find the best training recipes — then
back them up with working code and up to date documentation… Start from
papers. Papers contain the results, and results tell you what actually works."

Bounded output (500–1500 words); hard caps at 190k tokens, forces summary at
170k. The point: keep noisy paper / code text out of the parent's context
while still letting findings flow up.

### 3.2 `hf_jobs` — 11 ops, no resource validation in code

Ops: `run` / `ps` / `logs` / `inspect` / `cancel`, plus scheduled variants
(`run` / `ps` / `inspect` / `delete` / `suspend` / `resume`).

The interesting thing: there is **no** automatic OOM or sizing prediction.
All sizing rules live in the tool's `description` so the LLM has to reason
about them: "1-3B → t4-small/a10g-large, 7-13B → a10g-large, 30B+ →
a100-large." Default 30-min timeout is flagged as "kills training mid-run."

Embedded protocol (not code-enforced — prompt-enforced):

* "MUST have called `github_find_examples` + `github_read_file` to find a
  working reference implementation."
* "MUST have validated dataset format via `hf_inspect_dataset`."
* "MUST include `push_to_hub=True` and `hub_model_id`" — job storage is
  ephemeral.
* "Submit ONE job first… Only then submit the rest. Never submit all at once."
* OOM: drop batch size, raise `gradient_accumulation_steps` to keep effective
  batch constant; enable `gradient_checkpointing`; "Do NOT switch training
  methods… those change what the user gets."

### 3.3 `sandbox_*`

Sandboxes are private HF Spaces (persistent, not local Docker). Five ops:
`sandbox_create` (approval-gated, GPU-tier choice), `bash`, `read`, `write`,
`edit`. First use of any sandbox op auto-creates a `cpu-basic` sandbox
without approval. Reads cap at 100KB. Orphaned sandboxes >1h old are GC'd.

### 3.4 `plan`

TodoWrite-style. One global `_current_plan` of `{id, content, status}` items,
status ∈ `{pending, in_progress, completed}`, **only one in_progress at a
time**. Whole plan replaced on each call. Emits `plan_update` events.

## 4. Doctrine — what `system_prompt_v3.yaml` puts in the agent's head

The Python loop is small and generic. The "how to actually do ML" lives in
~15 KB of YAML. Highlights:

* **Knowledge skepticism.** "Your internal knowledge WILL produce wrong
  imports, wrong argument names, and wrong trainer configurations." → always
  research current TRL / transformers / PEFT APIs before coding.
* **5-step research loop.** Landmark paper → citation graph → methodology →
  recipe extraction (data + hyperparams) → validate.
* **Audit before use.** "Before working with any dataset, audit it first. Do
  not assume you know what the data looks like." Use `hf_inspect_dataset`.
* **Pre-flight checklist.** Reference impl found, dataset schema verified,
  `push_to_hub=True`, timeout ≥ 2 h, Trackio configured.
* **Sandbox-first.** Test non-trivial scripts in a sandbox before hitting
  `hf_jobs`. GPU sandbox (`t4-small` minimum) for any CUDA / bf16 / model-load
  path.
* **Trackio.** `TrackioCallback` handles init/log/finish; alerts at three
  severity levels (ERROR / WARN / INFO) with numeric values and suggested
  actions.
* **Headless rule.** "NEVER respond with only text. Every response MUST
  include at least one tool call." Iterate until time runs out.

## 5. How to generalise the pattern

The reusable chassis is mostly domain-agnostic. To re-aim it elsewhere:

1. **Keep the queue split.** Define an `OpType` enum and an `Event` schema
   first. CLI, headless, web, and tests attach as separate consumers without
   touching the loop.
2. **One tool abstraction.** Anything that fits `(name, description,
   JSONSchema, async handler → (str, ok))` is a tool. There's no "first-class
   vs MCP" split. Domain knowledge goes into the tool *description* — that's
   where ml-intern hides "you MUST have called X first" rules.
3. **Two-tier reasoning.** Identify the "explore-and-condense" steps in your
   domain (paper crawl, log scan, schema discovery, doc search) and put each
   behind a sub-agent tool with a cheaper model and a read-only subset. The
   parent only sees the bounded summary.
4. **Compaction at 90% with anchors.** Always preserve system msg + first
   user msg + last N. Summarise the middle with an LLM. Only `_COMPACT_PROMPT`
   is domain-flavoured.
5. **Doom-loop guard on tool signatures.** MD5 over `(name, normalised args,
   result)` is domain-free. Track results so polling doesn't false-positive.
6. **Approvals as one predicate + one slot.** `_needs_approval` is a single
   function; `pending_approval` is a single slot on session; `EXEC_APPROVAL`
   is a single op type. To gate "deploy to prod" or "drop table" instead of
   "submit GPU job," all you change is the predicate.
7. **Domain knowledge lives in prompt + tool descriptions, not in code.**
   `agent/core/*` doesn't know what SFT, DPO, or `push_to_hub` are. To pivot
   to a different domain (data engineering, infra ops, security review),
   rewrite the YAML and swap the toolbelt; leave the loop alone.

## 6. Things to flag

* **Prompt-injection in fetched content.** While exploring this repo I had
  three `<system-reminder>` blocks injected into WebFetch results sourced
  from ml-intern's own pages. Treat ml-intern's outputs (and any web content
  you let it ingest) the same way you'd treat tool output: it can carry
  instructions targeting *your* agent.
* **Hard limits to remember when porting.** `MAX_ITERATIONS = 300` cap (CLI
  default 50), `model_max_tokens = 180_000` default, compaction at 0.9,
  `_MAX_LLM_RETRIES = 3`, doom-loop `lookback = 30` / threshold 3, sub-agent
  output 500–1500 words bounded and 190k hard cap.
* **No code-level OOM/sizing checks for `hf_jobs`.** All guardrails are in
  prose. Fine for a competent LLM with a strong system prompt, but if you
  port this to a less-disciplined model, push some of those rules into the
  handler.

## Sources

* https://github.com/huggingface/ml-intern
* https://github.com/huggingface/ml-intern/blob/main/REVIEW.md
* https://github.com/huggingface/ml-intern/blob/main/AGENTS.md (frontend/backend dev notes)
* https://github.com/huggingface/ml-intern/tree/main/agent/core
* https://github.com/huggingface/ml-intern/tree/main/agent/tools
* https://github.com/huggingface/ml-intern/blob/main/agent/prompts/system_prompt_v3.yaml
