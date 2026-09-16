from __future__ import annotations

_VALID_JSON_ESCAPES = set('"\\/bfnrtu')


def _is_control_character(char: str) -> bool:
    code = ord(char)
    return 0x00 <= code <= 0x1F


def _escape_control_character(char: str) -> str:
    mapping = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    if char in mapping:
        return mapping[char]
    return f"\\u{ord(char):04x}"


def repair_json(json_text: str) -> str:
    repaired: list[str] = []
    in_string = False
    index = 0
    while index < len(json_text):
        char = json_text[index]
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue
        if char == '"':
            repaired.append(char)
            in_string = False
            index += 1
            continue
        if char == "\\":
            next_char = json_text[index + 1] if index + 1 < len(json_text) else None
            if next_char is None:
                repaired.append("\\\\")
                index += 1
                continue
            if next_char == "u":
                unicode_digits = json_text[index + 2 : index + 6]
                if len(unicode_digits) == 4 and all(digit in "0123456789abcdefABCDEF" for digit in unicode_digits):
                    repaired.append(f"\\u{unicode_digits}")
                    index += 6
                    continue
            if next_char in _VALID_JSON_ESCAPES:
                repaired.append(f"\\{next_char}")
                index += 2
                continue
            repaired.append("\\\\")
            index += 1
            continue
        repaired.append(_escape_control_character(char) if _is_control_character(char) else char)
        index += 1
    return "".join(repaired)


def _close_json(json_text: str) -> str:
    in_string = False
    escape = False
    stack: list[str] = []
    for char in json_text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack and char == stack[-1]:
            stack.pop()
    closed = json_text
    if in_string:
        closed += '"'
    closed += "".join(reversed(stack))
    return closed


def parse_json_with_repair(json_text: str) -> object:
    import json

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as error:
        repaired = repair_json(json_text)
        if repaired != json_text:
            return json.loads(repaired)
        raise error


def parse_streaming_json(partial_json: str | None) -> dict[str, object]:
    import json

    if not partial_json or not partial_json.strip():
        return {}
    try:
        parsed: object = parse_json_with_repair(partial_json)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_close_json(repair_json(partial_json)))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}
