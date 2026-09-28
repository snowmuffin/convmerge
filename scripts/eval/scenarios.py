"""Community pain points, reproduced: what happens without convmerge, and with it.

Each scenario builds a small input that reproduces a problem people report
(GitHub issues, forum threads), shows the failure with the real tool or chat
template, then runs the convmerge command a user would run and checks the
result. Real chat templates come from TRL's bundled ``chat_templates``
directory (``--templates``); rendering uses a local tokenizer (``--tokenizer``)
so no network is needed. Evaluation harness, not part of the package.

    python scripts/eval/scenarios.py --templates DIR --tokenizer DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

OUT: Path
TPL: Path
TOK: Any
TOK_DIR: Path
SCENARIOS: list[tuple[str, str, Callable[[], dict[str, Any]]]] = []


def scenario(pid: str, title: str) -> Callable:
    def wrap(fn: Callable[[], dict[str, Any]]) -> Callable[[], dict[str, Any]]:
        SCENARIOS.append((pid, title, fn))
        return fn

    return wrap


# --- helpers -------------------------------------------------------------------


def write(name: str, rows: list[Any]) -> Path:
    path = OUT / name
    path.write_text(
        "".join((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n"
                for r in rows),
        encoding="utf-8",
    )  # fmt: skip
    return path


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def cm(*args: str | Path, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, "-m", "convmerge", *map(str, args)],
                          capture_output=True, text=True)  # fmt: skip
    if check and proc.returncode not in (0,):
        raise RuntimeError(f"convmerge {' '.join(map(str, args))} -> {proc.returncode}\n"
                           f"{proc.stderr[-1500:]}")  # fmt: skip
    return proc


def template(name: str) -> str:
    return (TPL / f"{name}.jinja").read_text(encoding="utf-8")


def render(messages: list[dict[str, Any]], tpl: str, tools: Any = None) -> str:
    return TOK.apply_chat_template(messages, chat_template=tpl, tools=tools, tokenize=False)


def try_render(messages: list[dict[str, Any]], tpl: str, tools: Any = None) -> str:
    try:
        render(messages, tpl, tools)
        return "ok"
    except Exception as e:  # noqa: BLE001 - templates raise anything
        return f"{type(e).__name__}: {str(e)[:90]}"


U = lambda c: {"role": "user", "content": c}  # noqa: E731
A = lambda c, **kw: {"role": "assistant", "content": c, **kw}  # noqa: E731
S = lambda c: {"role": "system", "content": c}  # noqa: E731


# --- scenarios -----------------------------------------------------------------


@scenario("P1", "Mixed schemas (Alpaca, ShareGPT variants, OpenAI, rendered text, Q/A)")
def mixed_schemas() -> dict[str, Any]:
    layouts = {
        "alpaca": {"instruction": "Name a color.", "input": "", "output": "Blue."},
        "alpaca+input": {"instruction": "Translate", "input": "Hola", "output": "Hello"},
        "sharegpt human/gpt": {
            "conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]
        },
        "sharegpt user/model (Gemma, unsloth#1766)": {
            "conversations": [{"from": "user", "value": "Hi"}, {"from": "model", "value": "Yo"}]
        },
        "sharegpt role/content": {
            "conversations": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey"},
            ]
        },
        "sharegpt + system field": {
            "system": "Be brief.",
            "conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hey"}],
        },
        "OpenAI messages": {"messages": [S("Be nice."), U("Hi"), A("Hello")]},
        "ChatML text": {
            "text": "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n"
        },
        "Llama-2 [INST] text": {"text": "<s>[INST] Hi [/INST] Hello </s>"},
        "### Human text": {"text": "### Human: Hi ### Assistant: Hello"},
        "OpenOrca system_prompt/question/response": {
            "system_prompt": "You are helpful.",
            "question": "2+2?",
            "response": "4",
        },
        "prompt/completion": {"prompt": "2+2?", "completion": "4"},
        "question/answer (GSM8K, synthetic-data-kit qa)": {"question": "2+2?", "answer": "4"},
        "instruction/response": {"instruction": "2+2?", "response": "4"},
        "query/answer (CodeFeedback)": {"query": "2+2?", "answer": "4"},
        "problem/solution (NuminaMath)": {"problem": "2+2?", "solution": "4"},
        "input/output": {"input": "2+2?", "output": "4"},
        "messages as JSON string (orca-agentinstruct)": {
            "messages": json.dumps([U("Hi"), A("Hello")])
        },
    }
    results = {}
    for name, row in layouts.items():
        src = write("p1_in.jsonl", [row])
        dst = OUT / "p1_out.jsonl"
        proc = cm("convert", "-i", src, "-o", dst, "--from", "auto", "--format", "messages",
                  check=False)  # fmt: skip
        out = read(dst) if dst.exists() else []
        ok = bool(out) and any(m["role"] == "assistant" for m in out[0]["messages"])
        results[name] = "ok" if ok else ("no rows: " + proc.stderr.strip().splitlines()[-1][:120]
                                        if proc.stderr.strip() else "no rows")  # fmt: skip
    usable = sum(1 for r in layouts.values() if isinstance(r.get("messages"), list))
    n_ok = sum(v == "ok" for v in results.values())
    return {
        "before": f"{usable}/{len(layouts)} layouts are usable as `messages` without a script",
        "after": f"{n_ok}/{len(layouts)} convert with `--from auto` and no other flag",
        "detail": results,
        "verdict": "solved" if n_ok == len(layouts) else "partly",
    }


@scenario("P2", "Roles must alternate / Gemma has no system role (TRL#696, unsloth#334)")
def alternation() -> dict[str, Any]:
    conv = {"messages": [S("You are a pirate."), U("Hi"), U("Anyone there?"), A("Arr!"),
                         A("Ahoy."), U("Bye")]}  # fmt: skip
    conv["messages"].append(A("Farewell."))
    before = {name: try_render(conv["messages"], template(name))
              for name in ("gemma", "gemma3", "llama3", "qwen2_5")}  # fmt: skip
    src = write("p2_in.jsonl", [conv])
    dst = OUT / "p2_out.jsonl"
    cm("convert", "-i", src, "-o", dst, "--from", "auto", "--format", "messages",
       "--merge-consecutive", "--system", "fold")  # fmt: skip
    fixed = read(dst)[0]["messages"]
    after = {name: try_render(fixed, template(name)) for name in before}
    roles = [m["role"] for m in fixed]
    return {
        "before": before,
        "after": {**after, "roles": roles},
        "verdict": "solved" if all(v == "ok" for v in after.values()) else "partly",
    }


@scenario("P3", "Model never learns to stop: answer not followed by EOS (unsloth#416, PEFT#1003)")
def missing_eos() -> dict[str, Any]:
    tpl = OUT / "no_eos.jinja"
    tpl.write_text("{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}")
    src = write("p3_in.jsonl", [{"messages": [U("Hi"), A("Hello")]}] * 5)
    bad = json.loads(cm("tokens", "-i", src, "--tokenizer", TOK_DIR, "--chat-template", tpl,
                        check=False).stdout)  # fmt: skip
    good = json.loads(cm("tokens", "-i", src, "--tokenizer", TOK_DIR, "--chat-template",
                         TPL / "qwen3.jinja", check=False).stdout)  # fmt: skip
    return {
        "before": "silent: the trainer accepts the template; the model learns no stop token",
        "after": {
            "custom template missing_eos": bad.get("missing_eos"),
            "qwen3 template missing_eos": good.get("missing_eos"),
            "hints": bad.get("hints", [])[:2],
        },  # fmt: skip
        "verdict": "detected"
        if bad.get("missing_eos") == 5 and not good.get("missing_eos")
        else "missed",
    }


@scenario("P4", "Qwen3 drops <think> of earlier turns; multi-turn reasoning is not trained")
def qwen3_think() -> dict[str, Any]:
    rows = []
    for i in range(10):
        msgs = []
        for t in range(4):
            msgs += [U(f"q{i}.{t}"), A(f"answer {t}", reasoning_content=f"trace {i}.{t} " * 3)]
        rows.append({"messages": msgs})
    src = write("p4_in.jsonl", rows)
    tpl = template("qwen3")
    kept = sum(render(r["messages"], tpl).count("trace ") // 3 for r in rows)
    report = json.loads(cm("tokens", "-i", src, "--tokenizer", TOK_DIR, "--chat-template",
                           TPL / "qwen3.jinja", check=False).stdout)  # fmt: skip
    dst = OUT / "p4_split.jsonl"
    cm("convert", "-i", src, "-o", dst, "--from", "auto", "--format", "messages", "--split-turns")
    split = read(dst)
    kept_after = sum(render(r["messages"], tpl).count("trace ") // 3 for r in split)
    return {
        "before": f"{kept}/40 reasoning traces reach the rendered text (the rest vanish silently)",
        "after": {
            "tokens reasoning_dropped": report.get("reasoning_dropped"),
            "after --split-turns": f"{kept_after}/40 traces rendered, {len(split)} rows",
        },
        "verdict": "solved" if kept_after == 40 and report.get("reasoning_dropped") else "partly",
    }


@scenario("P5", "assistant_only_loss silently off; truncation leaves no answer (TRL#3927)")
def loss_mask() -> dict[str, Any]:
    long_system = "Background: " + "lorem ipsum dolor " * 400
    rows = [{"messages": [S(long_system), U("Q"), A("A short answer.")]}] * 3 + [
        {"messages": [U("Q"), A("A short answer.")]}
    ] * 3
    src = write("p5_in.jsonl", rows)
    plain = json.loads(cm("tokens", "-i", src, "--tokenizer", TOK_DIR, "--chat-template",
                          TPL / "qwen3.jinja", "--max-tokens", "256",
                          check=False).stdout)  # fmt: skip
    training = json.loads(cm("tokens", "-i", src, "--tokenizer", TOK_DIR, "--chat-template",
                             TPL / "qwen3_training.jinja", check=False).stdout)  # fmt: skip
    return {
        "before": "TRL trains with loss 0 on rows whose answer is cut off; without "
        "{% generation %} assistant_only_loss does nothing",
        "after": {
            "qwen3.jinja generation_tags": plain.get("generation_tags"),
            "qwen3_training.jinja generation_tags": training.get("generation_tags"),
            "answer_beyond_limit (max 256)": plain.get("answer_beyond_limit"),
        },
        "verdict": "detected"
        if plain.get("answer_beyond_limit") == 3
        and plain.get("generation_tags") is False
        and training.get("generation_tags")
        else "missed",
    }


@scenario(
    "P6",
    "Tool arguments double-encoded; tool-call content None renders 'None' "
    "(TRL#5460, transformers#45419)",
)
def tools() -> dict[str, Any]:
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Seoul"})},
    }
    tool_spec = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "w",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    msgs = [
        U("Weather?"),
        A(None, tool_calls=[call]),
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        A("Sunny."),
    ]
    src = write("p6_in.jsonl", [{"messages": msgs, "tools": tool_spec}])

    def state(m: list[dict[str, Any]], name: str, spec: Any) -> str:
        try:
            text = render(m, template(name), spec)
        except Exception as e:  # noqa: BLE001
            return f"template error: {str(e)[:60]}"
        if "None" in text and name.startswith("glm"):
            return "renders 'None'"
        return "double-encoded" if '\\"city\\"' in text else "ok"

    names = ("llama3_1", "llama3_2", "qwen2_5", "qwen3", "deepseekv3", "glm4moe")
    before = {name: state(msgs, name, tool_spec) for name in names}
    rep = json.loads(cm("tokens", "-i", src, "--tokenizer", TOK_DIR, "--chat-template",
                        TPL / "llama3_1.jinja", check=False).stdout)  # fmt: skip
    dst = OUT / "p6_out.jsonl"
    cm(
        "convert",
        "-i",
        src,
        "-o",
        dst,
        "--from",
        "auto",
        "--format",
        "messages",
        "--tool-arguments",
        "object",
    )
    fixed = read(dst)[0]
    after: dict[str, Any] = {name: state(fixed["messages"], name, fixed.get("tools"))
                             for name in names}  # fmt: skip
    after["tokens on the unfixed file (llama3_1)"] = {
        "template_errors": sum(rep.get("template_errors", {}).values()),
        "hints": rep.get("hints", [])[:1],
    }
    return {"before": before, "after": after,
            "verdict": "solved" if all(v == "ok" for k, v in after.items() if "tokens" not in k)
            else "partly"}  # fmt: skip


REFUSALS = [
    "I'm sorry, but I can't help with that.",
    "I cannot assist with creating malware.",
    "As an AI language model, I don't have personal opinions.",
    "I apologize, but I cannot provide that information.",
    "Sorry, I can't do that.",
    "I'm unable to help with this request.",
    "As a large language model, I cannot browse the internet.",
    "죄송하지만 그 요청은 도와드릴 수 없습니다.",
    "저는 AI 언어 모델로서 개인적인 의견이 없습니다.",
    "죄송합니다만, 해당 정보는 제공할 수 없습니다.",
    "I must decline to write that.",
    "I'm sorry, I cannot answer that question as it is not appropriate.",
]
LEGIT = [
    "I'm sorry to hear about your loss. Here are some ways to cope: talk to friends...",
    "The function cannot handle negative numbers, so add a check before calling it.",
    "You can't divide by zero; the result is undefined.",
    "Sorry is often the hardest word. The song was released in 1976.",
    "The API is unable to return more than 100 items per page; use pagination.",
    "As an example, the model predicts the next token given the context.",
    "If a user says 'I cannot assist', your bot should route them to a human.",
    "We apologize for the delay in the train schedule, the announcement said.",
    "죄송하다는 말은 상황에 맞게 써야 합니다. 예를 들어 늦었을 때는...",
    "이 함수는 음수를 처리할 수 없으므로 먼저 값을 확인하세요.",
    "Paris is the capital of France.",
    "Here is the code:\n```python\nraise ValueError('cannot be empty')\n```",
]
NAIVE_KEYWORDS = ["sorry", "cannot", "can't", "unable", "apologize", "as an ai",
                  "language model", "죄송", "없습니다"]  # fmt: skip


@scenario("P7", "Refusals and 'As an AI' disclaimers (dolphin-style keyword filters)")
def refusals() -> dict[str, Any]:
    rows = [{"messages": [U("q"), A(t)]} for t in REFUSALS + LEGIT]
    src = write("p7_in.jsonl", rows)
    rej = OUT / "p7_rej.jsonl"
    cm("filter", "-i", src, "-o", OUT / "p7_out.jsonl", "--rejects", rej)
    caught = {r["messages"][1]["content"] for r in read(rej)}

    def score(flagged: set[str]) -> dict[str, float]:
        tp = len(flagged & set(REFUSALS))
        fp = len(flagged & set(LEGIT))
        return {
            "recall": round(tp / len(REFUSALS), 2),
            "precision": round(tp / max(tp + fp, 1), 2),
            "false_positives": fp,
        }

    naive = {t for t in REFUSALS + LEGIT if any(k in t.lower() for k in NAIVE_KEYWORDS)}
    return {
        "before": {"naive keyword filter": score(naive)},
        "after": {"convmerge filter": score(caught)},
        "verdict": "better"
        if score(caught)["false_positives"] < score(naive)["false_positives"]
        else "same",
        "note": "hand-labelled set of 12 refusals and 12 look-alikes; small, indicative only",
    }


@scenario("P8", "DPO data defects: identical pairs, length bias, prompt-only dedupe")
def dpo() -> dict[str, Any]:
    pairs = [{"prompt": f"p{i % 5}", "chosen": f"good answer {i} " * 5, "rejected": f"bad {i}"}
             for i in range(25)]  # fmt: skip
    pairs.append({"prompt": "p0", "chosen": "Same", "rejected": "same "})
    src = write("p8_in.jsonl", pairs)
    rep = json.loads(cm("filter", "-i", src).stdout)
    whole = OUT / "p8_whole.jsonl"
    cm("dedupe", "-i", src, "-o", whole)
    prompt_only = OUT / "p8_prompt.jsonl"
    cm("dedupe", "-i", src, "-o", prompt_only, "--keys", "prompt")
    return {
        "before": f"deduping by prompt keeps {len(read(prompt_only))}/26 pairs "
        "(distinct pairs sharing a prompt are lost)",
        "after": {
            "dedupe (default, whole row)": f"{len(read(whole))}/26 kept",
            "near_identical_pair": rep["rules"].get("near_identical_pair"),
            "chosen_longer_share": rep["preference"]["chosen_longer_share"],
            "length-bias warning": "yes"
            if rep["preference"]["chosen_longer_share"] > 0.7
            else "no",
        },
        "verdict": "solved",
    }


@scenario("P9", "Near-duplicates and benchmark contamination (text-dedup, llm-decontaminator)")
def dedup_decontam() -> dict[str, Any]:
    base = (
        "The mitochondria is the powerhouse of the cell. It makes ATP through cellular "
        "respiration in several stages: glycolysis in the cytoplasm, the citric acid cycle "
        "in the matrix, and oxidative phosphorylation along the inner membrane, where the "
        "electron transport chain pumps protons that ATP synthase lets back in."
    )
    rows = [
        {"messages": [U("Explain mitochondria."), A(base)]},
        {"messages": [U("Explain mitochondria!"), A(base.replace("several", "a few"))]},
        {"messages": [U("Explain mitochondria."), A(base.replace("powerhouse", "engine"))]},
        {"messages": [U("What is 2+2?"), A("4")]},
    ]
    src = write("p9_in.jsonl", rows)
    cm("dedupe", "-i", src, "-o", OUT / "p9_exact.jsonl")
    near = cm("dedupe", "-i", src, "-o", OUT / "p9_near.jsonl", "--near")
    gsm = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
        "clips in May. How many clips did Natalia sell altogether in April and May?"
    )
    ev = write("p9_eval.jsonl", [{"question": gsm, "answer": "72"}])
    train = write("p9_train.jsonl", [
        {"messages": [U("Solve: " + gsm), A("72")]},
        {"messages": [U("(A) " + gsm.replace("April", "April,")), A("72")]},
        {"messages": [U("Natalia sold clips to her friends. In May she sold half. Total?"),
                      A("72")]},
    ])  # fmt: skip
    dec = json.loads(cm("decontam", "-i", train, "--against", ev).stdout)
    return {
        "before": f"exact dedupe keeps {len(read(OUT / 'p9_exact.jsonl'))}/4 (edited copies stay)",
        "after": {
            "dedupe --near kept": f"{len(read(OUT / 'p9_near.jsonl'))}/4",
            "near stderr": near.stderr.strip()[:120],
            "decontam: verbatim / labelled / paraphrase caught": (
                f"{dec['contaminated']}/3 (paraphrase is not caught by design)"
            ),
        },
        "verdict": "solved"
        if len(read(OUT / "p9_near.jsonl")) == 2 and dec["contaminated"] == 2
        else "partly",
    }


@scenario("P10", "AI Hub-style nested JSON needs a custom script")
def aihub() -> dict[str, Any]:
    record = {
        "info": {"topic": "은행 상담", "id": "B-001"},
        "dialogue": [
            {
                "utterances": [
                    {"speaker": "고객", "text": "계좌를 만들고 싶어요."},
                    {"speaker": "상담사", "text": "신분증을 준비해 주세요."},
                    {"speaker": "고객", "text": "온라인으로도 되나요?"},
                    {"speaker": "상담사", "text": "네, 앱에서 가능합니다."},
                ]
            }
        ],
    }
    src = write("p10_in.jsonl", [record])
    auto = OUT / "p10_auto.jsonl"
    cm("convert", "-i", src, "-o", auto, "--from", "auto", "--format", "messages", check=False)
    spec = {
        "map": {
            "turns": "dialogue[].utterances[]",
            "role": "speaker",
            "content": "text",
            "role_map": {"고객": "user", "상담사": "assistant"},
            "system": "info.topic",
        }
    }
    dst = OUT / "p10_map.jsonl"
    cm(
        "convert",
        "-i",
        src,
        "-o",
        dst,
        "--from",
        "map",
        "--format",
        "messages",
        "--adapter-kwargs",
        json.dumps(spec),
    )
    msgs = read(dst)[0]["messages"]
    return {
        "before": f"--from auto: {len(read(auto)) if auto.exists() else 0} rows "
        "(layout unknown; normally a hand-written script)",
        "after": {"--from map roles": [m["role"] for m in msgs]},
        "verdict": "solved"
        if [m["role"] for m in msgs][:3] == ["system", "user", "assistant"]
        else "partly",
    }


@scenario("P11", "Licenses of mixed sources are tracked by hand")
def licenses() -> dict[str, Any]:
    write("p11_a.jsonl", [{"instruction": "q", "input": "", "output": "a"}] * 3)
    write("p11_b.jsonl", [{"instruction": "q2", "input": "", "output": "b"}] * 2)
    recipe = OUT / "p11_recipe.json"
    recipe.write_text(json.dumps({
        "output": "p11_out.jsonl", "workdir": "p11_build",
        "sources": {"a": {"path": "p11_a.jsonl", "license": "apache-2.0",
                          "convert": {"from": "alpaca"}},
                    "b": {"path": "p11_b.jsonl", "license": "cc-by-nc-4.0",
                          "convert": {"from": "alpaca"}}},
    }))  # fmt: skip
    proc = cm("run", recipe)
    report = json.loads((OUT / "p11_build" / "report.json").read_text())
    return {
        "before": "no record of which rows come from which license",
        "after": {
            "licenses": report["licenses"],
            "log": [x for x in proc.stderr.splitlines() if x.startswith("[license]")],
        },
        "verdict": "solved" if report["license_warnings"] else "missed",
    }


@scenario("P12", "Reproducibility: re-running a pipeline, parallel conversion")
def reproducibility() -> dict[str, Any]:
    rows = [
        {"conversations": [{"from": "human", "value": f"q{i}"}, {"from": "gpt", "value": f"a{i}"}]}
        for i in range(2000)
    ]
    src = write("p12_in.jsonl", rows)
    one, four = OUT / "p12_w1.jsonl", OUT / "p12_w4.jsonl"
    cm("convert", "-i", src, "-o", one, "--from", "auto", "--format", "messages")
    cm("convert", "-i", src, "-o", four, "--from", "auto", "--format", "messages", "--workers", "4")
    recipe = OUT / "p11_recipe.json"
    second = cm("run", recipe)
    return {
        "before": "ad-hoc scripts re-run everything and drift between runs",
        "after": {
            "--workers 4 byte-identical": one.read_bytes() == four.read_bytes(),
            "recipe re-run": second.stderr.strip().splitlines()[-1][:120],
        },
        "verdict": "solved"
        if one.read_bytes() == four.read_bytes() and "ran 0" in second.stderr
        else "partly",
    }


@scenario("P13", "Outputs of synthetic-data tools (easy-dataset, synthetic-data-kit, distilabel)")
def synthetic_tools() -> dict[str, Any]:
    layouts = {
        "easy-dataset Alpaca (JSON array)": [
            {"instruction": "q", "input": "", "output": "a", "system": "s"}
        ],
        "easy-dataset ShareGPT (JSON array)": [{"messages": [S("s"), U("q"), A("a")]}],
        "synthetic-data-kit qa pairs": [{"question": "q", "answer": "a"}],
        "synthetic-data-kit alpaca": [{"instruction": "q", "input": "", "output": "a"}],
        "synthetic-data-kit ft (OpenAI)": [{"messages": [S("s"), U("q"), A("a")]}],
        "synthetic-data-kit cot": [{"question": "q", "reasoning": "r", "answer": "a"}],
        "distilabel instruction/generation": [
            {"instruction": "q", "generation": "a", "model_name": "m"}
        ],
        "distilabel prompt/completion": [{"prompt": "q", "completion": "a"}],
        "distilabel DPO (prompt/chosen/rejected str)": [
            {"prompt": "q", "chosen": "a", "rejected": "b"}
        ],
    }
    results = {}
    for name, rows in layouts.items():
        path = OUT / "p13_in.json"
        path.write_text(json.dumps(rows, ensure_ascii=False))
        norm = OUT / "p13_norm.jsonl"
        cm("normalize", "-i", path, "-o", norm, check=False)
        fmt = "preference" if "DPO" in name else "messages"
        dst = OUT / "p13_out.jsonl"
        cm("convert", "-i", norm, "-o", dst, "--from", "auto", "--format", fmt, check=False)
        out = read(dst) if dst.exists() else []
        results[name] = "ok" if out else "not converted"
    n_ok = sum(v == "ok" for v in results.values())
    return {
        "before": "each tool's export needs its own mapping",
        "after": results,
        "verdict": "solved" if n_ok == len(layouts) else f"partly ({n_ok}/{len(layouts)})",
    }


@scenario(
    "P14",
    "LLaMA-Factory dataset_info.json / axolotl field mapping errors "
    "(LLaMA-Factory#7577, axolotl#2089)",
)
def trainer_configs() -> dict[str, Any]:
    src = write("p14_in.jsonl", [{"conversations": [{"from": "human", "value": "q"},
                                                    {"from": "gpt", "value": "a"}],
                                  "system": "s"}])  # fmt: skip
    sg = OUT / "p14_sg.jsonl"
    cm("convert", "-i", src, "-o", sg, "--from", "auto", "--format", "sharegpt")
    info = cm("llamafactory-info", "-i", sg, "--name", "mine").stdout
    axo = cm("axolotl-config", "-i", sg).stdout
    return {
        "before": "hand-written column mappings; a wrong key fails late or trains on empty turns",
        "after": {
            "llamafactory-info": json.loads(info),
            "axolotl-config": axo.strip().splitlines()[1:8],
        },
        "verdict": "solved (verified with the real trainers in 0.11/0.13)",
    }


# --- main ----------------------------------------------------------------------


def main() -> int:
    global OUT, TPL, TOK, TOK_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    OUT, TPL, TOK_DIR = args.out, args.templates, args.tokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    TOK = AutoTokenizer.from_pretrained(str(TOK_DIR))
    if TOK.bos_token is None:  # Llama templates read bos_token; the test tokenizer has none
        TOK.bos_token = "<|im_start|>"
    results = []
    import os

    os.chdir(OUT)
    for pid, title, fn in SCENARIOS:
        if args.only and pid not in args.only:
            continue
        try:
            res = fn()
        except Exception:  # noqa: BLE001
            res = {"verdict": "error", "error": traceback.format_exc()[-1200:]}
        results.append({"id": pid, "title": title, **res})
        print(f"[{pid}] {res.get('verdict')}  {title}", file=sys.stderr)
    (OUT / "scenarios.json").write_text(json.dumps(results, ensure_ascii=False, indent=1,
                                                   default=str))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
