# TAU2-Bench user-simulator shim: interface spec

`text2_run.sh tau2 ...` (`~/evals/text2/benches/tau2.sh`) runs the official tau2-bench orchestrator. The agent under
test is our model. The other side of every conversation, the simulated customer, is an LLM that tau2 calls through
litellm as `openai/$USER_SIM_MODEL` at `$USER_SIM_BASE_URL`. This file specifies the endpoint that has to answer those
calls. The endpoint will be backed by Claude agents, and this spec does not implement that backing.

## 1. What the counterpart does

It plays the customer in a customer-service chat. It gets a scenario (persona, reason for call, known/unknown info,
task instructions) and must:
- write one customer message per turn, revealing information only when asked and never inventing facts;
- in **telecom only**, operate its own phone through user-side tools (`check_network_status`, `toggle_airplane_mode`,
  `toggle_data`, `set_network_mode_preference`, `reboot_device`, ..., 30 tool schemas) when the agent asks, one tool
  call per turn, and then report what it saw;
- end the conversation with a control token in its message:
  `###STOP###` (goal satisfied), `###TRANSFER###` (agent clearly transferred it to a human), or
  `###OUT-OF-SCOPE###` (the scenario lacks the information needed to continue). tau2 ends the episode when one of
  these substrings appears (`tau2/user/base.py`, `UserSimulator.is_stop`).

The simulator's behaviour sets the score: it decides when the customer is satisfied, which details it volunteers, and,
in telecom, which device actions get executed. The final reward is computed deterministically by tau2 (section 6).

Exact prompt sources (pinned repo `~/evals/tau2/tau2-bench`, commit 409231d = upstream c5b2d22 + PR #94):
- `data/tau2/user_simulator/simulation_guidelines.md` (airline, retail: the user has no tools)
- `data/tau2/user_simulator/simulation_guidelines_tools.md` (telecom: the user has tools)
- system prompt = `"{guidelines}\n\n<scenario>\n{str(task.user_scenario)}\n</scenario>"` (`src/tau2/user/user_simulator.py`)

The shim must pass the system prompt through **verbatim** as the Claude agent's role instructions. It must not
add hints about the task's expected outcome.

## 2. Requests the shim receives (OpenAI chat completions, non-streaming)

`POST {USER_SIM_BASE_URL}/chat/completions`, sent by litellm 1.65.1:
```json
{
  "model": "gpt-4.1-2025-04-14",          // USER_SIM_MODEL; names the official user model it stands in for
  "messages": [ ... ],
  "temperature": 0.0,                      // official user default
  "seed": 626729,                          // per-trial seed set by tau2 (same for all sims of one trial)
  "tools": [ ... ], "tool_choice": "auto"  // telecom only (user-side device tools, OpenAI function schema)
}
```
The `messages` are in the **simulated user's own perspective**. tau2 flips the roles (`UserState.flip_roles`):

| role in request | what it is |
|---|---|
| `system` | guidelines plus the `<scenario>` (sent on every call) |
| `user` | the **agent's** text messages. The first is always `"Hi! How can I help you today?"`. The agent's tool calls and their results are never shown. |
| `assistant` | the simulated customer's own previous turns (`content`, or `tool_calls` with `arguments` as a JSON string) |
| `tool` | results of the customer's own device tool calls (`tool_call_id`, `content`), telecom only |

Every request carries the whole conversation so far, so the endpoint can stay stateless. To keep one conversation
on one Claude agent, use a conversation key: sha256 of the system message plus `seed`, which is unique per
(task, trial) and stable across that conversation's turns. Within one key, the turns arrive strictly in sequence.

## 3. Response the shim must return

```json
{"id": "chatcmpl-...", "object": "chat.completion", "created": 0, "model": "gpt-4.1-2025-04-14",
 "choices": [{"index": 0, "finish_reason": "stop" | "tool_calls",
              "message": {"role": "assistant",
                          "content": "<customer utterance, possibly containing ###STOP###>" | null,
                          "tool_calls": [{"id": "call_x", "type": "function",
                                          "function": {"name": "toggle_data", "arguments": "{}"}}]}}],
 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
```
Rules:
- Return **either** a text message **or** tool calls, not both (the guidelines require this). Normally there is one
  tool call per turn.
- `content` must be a non-empty string whenever there are no `tool_calls`. Otherwise tau2 asserts, and the
  simulation is retried and eventually scored 0 as a harness exception.
- `function.arguments` must be a JSON **string** that parses to an object.
- Control tokens go inside `content` and may be the whole message (`"###STOP###"`).
- Return only the customer's words. Do not add reasoning or meta commentary. Hidden reasoning, if any, must not be
  in `content`.
- `GET {USER_SIM_BASE_URL}/models` should list `USER_SIM_MODEL`. tau2 itself does not call it, but it is handy for
  health checks.

## 4. Volume and latency (estimated from public trajectories of Qwen models on this task set)

Per-simulation means, from Sierra's public S3 trajectories (`s3://sierra-tau-bench-public/submissions/...`, local
copies in `~/evals/tau2/refs/`):

| domain | tasks | user-sim requests/sim | agent requests/sim | user prompt tokens mean / p95 / max | sim wall time (Qwen3.5-397B) mean |
|---|---|---|---|---|---|
| airline | 50 | 4.7-5.9 | 12-12.3 | 1.5k / 4.4k / 11.5k | 23 min |
| retail | 114 | 4.9-6.1 | 11.3-12.6 | 1.1k / 2.5k / 9.1k | 12 min |
| telecom | 114 | 19.9-20.7 (about 10 of them are tool calls) | 14-15.2 | 3.8k / 5.6k / 8.2k | 14 min |

(ranges: Qwen3.5-397B-A17B with a gpt-5.2 user / Qwen3-Max-Thinking with a gpt-4.1 user). Completions are short,
usually under 100 tokens.

Totals:
- **TRIALS=4 (default, official)**: 1,112 simulations; about **12,800 user-sim requests** (telecom about 9,300) and
  about 14,600 agent requests.
- **TRIALS=1**: 278 simulations; about **3,200 user-sim requests** and about 3,650 agent requests.
- Concurrency: at most `<concurrency>` simulations run at once (the text2 `<concurrency>` argument, passed as tau2
  `--max-concurrency`). Each simulation waits on one side at a time, so at most `<concurrency>` user-sim requests are
  in flight. In practice it is fewer, because the agent's thinking turns dominate.
- Latency: user turns lie on the critical path of each simulation. With a 30 s shim answer, a telecom simulation
  gains about 10 min. litellm timeout per user request = `USER_SIM_TIMEOUT` (default 3600 s), with 3 retries
  (`num_retries`).

## 5. Proposed shim (not implemented)

A small FastAPI process (e.g. `:28450`) that turns each HTTP request into a queued job and blocks until a Claude agent
answers it:
```
POST /v1/chat/completions
  -> key = sha256(messages[0].content + str(seed)); turn = number of non-system messages
  -> write  <queue>/pending/<key>.<turn>.json  {"id", "key", "turn", "created", "model", "messages", "tools"}
  -> wait for <queue>/done/<key>.<turn>.json   {"content": str|null, "tool_calls": [{"name","arguments"}]|null}
     (poll about 0.5 s; give up after USER_SIM_TIMEOUT and return HTTP 504, which litellm retries)
  -> validate it (section 3 rules; on violation ask again or return 500) and wrap it as a chat.completion
GET /v1/models, GET /stats (pending / answered / mean latency)
```
Worker side: one Claude agent per conversation key, for consistency. It gets the system prompt as its role
instructions and the flipped transcript, and writes only the next customer turn in the done file. Requests with the
same (system, messages) content are idempotent: if litellm retries, answer from the done file.

Fidelity notes. These are deviations to state when reporting.
- Official and Qwen setups use a fixed user model (`gpt-4.1-2025-04-14`, temperature 0). Qwen's own published
  tau2 submissions for Qwen3-Max used this setup: `web/leaderboard/public/submissions/qwen3-max_qwen_2026-01-23`.
  Claude role-play is a different user simulator. Scores are comparable only up to user-simulator effects, which
  published results show to be several points. For example, Sierra re-ran Qwen3.5-397B with a gpt-5.2 user and
  got airline 81.5 / retail 84.4 / telecom 97.8.
- The user model cannot be made deterministic. Seed and temperature are informational only.

## 6. LLM-judged parts of tau2 evaluation: none in the reward used here

At the pinned commit, `tau2 run` evaluates with `EvaluationType.ALL`:
- airline and retail reward = DB state match times COMMUNICATE (substring check of `communicate_info` in agent
  messages). This is the default `reward_basis` of every airline and retail base task.
- telecom reward = ENV_ASSERTION (device and line state), times ACTION for 20 of the 114 tasks.

`nl_assertions` exist in the data (all 50 airline tasks, 8 retail tasks). They are scored by an LLM judge only under
`EvaluationType.ALL_WITH_NL_ASSERTIONS` (marked WIP), which the official run does not use. So **no judge calls**
happen and none are needed. (tau2 2.0.0 later made communicate checks LLM-judged by gpt-5.5. That version is not
the one Qwen3.5 was evaluated with and is not used here.)

## 7. Implemented: Claude-worker relay (`~/evals/usersim/`)

`start_shim.sh` (shim.py on 127.0.0.1:28400), `source env.sh` (USER_SIM_BASE_URL / USER_SIM_MODEL=claude-usersim),
workers follow `WORKER.md` using `worker.py`. Job key = sha256 of the full simulator input (messages without tool-call
ids, seed, tool names); timeout -> HTTP 500 (litellm retries and re-joins the queued job). Verified end to end with the
gold stub agent (airline task 7 and telecom `airplane_mode_on|bad_network_preference[PERSONA:Hard]`, pass^1 = 1.0 each).
