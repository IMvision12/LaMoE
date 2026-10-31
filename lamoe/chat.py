"""Portable IMvision SFT renderer: accepts canonical JSONL records and a Tokenizer.

No model imports, automatic tool execution, network access, or vocabulary changes.
The caller must use the tokenizer.json shipped with the dataset.
"""

import json


def compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def render_record(record, tokenizer):
    ids = [tokenizer.token_to_id("<|bos|>")]
    labels = [-100]
    end = tokenizer.token_to_id("<|end_turn|>")
    for message in record["messages"]:
        role = message["role"]
        body = message["content"]
        if role == "system" and record.get("tools"):
            body += "\n<tools>" + compact(record["tools"]) + "</tools>"
        if role == "assistant":
            for call in message.get("tool_calls", []):
                f = call["function"]
                body += (
                    ("\n" if body else "")
                    + "<tool_call>"
                    + compact(
                        dict(
                            id=call["id"],
                            name=f["name"],
                            arguments=json.loads(f["arguments"]),
                        )
                    )
                    + "</tool_call>"
                )
        if role == "tool":
            body = (
                "<tool_response>"
                + compact(dict(tool_call_id=message["tool_call_id"], content=body))
                + "</tool_response>"
            )
        token_role = "user" if role == "tool" else role
        encoded = tokenizer.encode(body, add_special_tokens=False).ids
        ids.extend([tokenizer.token_to_id(f"<|{token_role}|>")] + encoded + [end])
        labels.extend(
            [-100]
            + (encoded + [end] if role == "assistant" else [-100] * (len(encoded) + 1))
        )
    if not record["messages"][-1].get("tool_calls"):
        eos = tokenizer.token_to_id("<|eos|>")
        ids.append(eos)
        labels.append(eos)
    assert len(ids) == len(labels) and None not in ids
    return ids, labels


def assistant_prefix(record, tokenizer):
    """Render an INFERENCE history ending in user/tool, then add assistant header.

    Do not render a training row with this function. Histories may end in tool;
    tool execution, permission checks, and pending-call tracking belong to the host.
    """
    if record["messages"][-1]["role"] not in ("user", "tool"):
        raise ValueError("Expected history ending in user or tool")
    temp = dict(
        record, messages=record["messages"] + [dict(role="assistant", content="")]
    )
    ids, _ = render_record(temp, tokenizer)
    # Remove the empty assistant END_TURN and terminal EOS; leave its role header.
    assert ids[-2] == tokenizer.token_to_id("<|end_turn|>")
    return ids[:-2]
