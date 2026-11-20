"""
Agent with NVIDIA NeMo Guardrails Integration (VinBank AI Guardrails).

Module này đóng gói logic kiểm duyệt nâng cao bằng NeMo Guardrails:
  - Input Rail:
      1) vinbank protected value check (kiểm tra rò rỉ / probing các giá trị secret)
      2) self check input (LLM phân tích ý đồ - Semantic Intent Classification)
  - Output Rail:
      1) vinbank leak check (quét tìm secret - kể cả giải mã base64, rot13, đảo ngược, squashed)
      2) self check output (LLM kiểm tra rò rỉ ngữ nghĩa / gợi ý thông số hệ thống)
"""
from __future__ import annotations

import time
from typing import Optional

from core.config import get_red_model, get_openai_api_key
from agents.agent import BLUE_INSTRUCTION
from agents.red_advance_nemo import (
    mentions_protected_value,
    SELF_CHECK_INPUT_PROMPT,
    SELF_CHECK_OUTPUT_PROMPT,
)


def check_nemo_input_rail(
    user_input: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict:
    """Kiểm tra tầng NeMo Input Rails:
       1) vinbank protected value check (deterministic)
       2) self check input (LLM Semantic Intent Classification)
    """
    if not api_key:
        api_key = get_openai_api_key()
    if not model_name:
        model_name = get_red_model()

    # 1. Deterministic check: protected values
    if mentions_protected_value(user_input):
        return {
            "allowed": False,
            "rail": "vinbank protected value check",
            "message": (
                "Request blocked by NeMo Guardrails (input rail: vinbank protected value check). "
                "I can only help with VinBank banking questions."
            ),
            "detail": "Phát hiện giá trị secret cần bảo vệ (side-channel/xác nhận)",
        }

    # 2. LLM self check input
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = SELF_CHECK_INPUT_PROMPT.replace("{{ user_input }}", user_input)
            call_kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
            }
            try:
                comp = client.chat.completions.create(**call_kwargs)
            except Exception as e:
                if "temperature" in str(e).lower():
                    call_kwargs["temperature"] = 1.0
                    comp = client.chat.completions.create(**call_kwargs)
                else:
                    raise
            ans = (comp.choices[0].message.content or "").strip().lower()
            if "yes" in ans:
                return {
                    "allowed": False,
                    "rail": "self check input",
                    "message": (
                        "Request blocked by NeMo Guardrails (input rail: self check input). "
                        "I can only help with VinBank banking questions."
                    ),
                    "detail": "LLM phân tích ý đồ (Intent): Phát hiện cố tình trích xuất cấu hình / đổi vai!",
                }
        except Exception as e:
            print(f"[NeMo Input Rail Error]: {e}")
            if any(
                w in user_input.lower()
                for w in ["password", "secret", "api key", "system prompt", "___"]
            ):
                return {
                    "allowed": False,
                    "rail": "self check input",
                    "message": (
                        "Request blocked by NeMo Guardrails (input rail: self check input). "
                        "I can only help with VinBank banking questions."
                    ),
                    "detail": "NeMo Intent Safety Rail: Phát hiện ý định moi dữ liệu nhạy cảm!",
                }
    else:
        if any(
            w in user_input.lower()
            for w in ["password", "secret", "api key", "system prompt", "___"]
        ):
            return {
                "allowed": False,
                "rail": "self check input",
                "message": (
                    "Request blocked by NeMo Guardrails (input rail: self check input). "
                    "I can only help with VinBank banking questions."
                ),
                "detail": "NeMo Intent Safety Rail: Phát hiện ý định moi dữ liệu nhạy cảm!",
            }

    return {
        "allowed": True,
        "rail": None,
        "message": None,
        "detail": "NeMo Intent Safety Rail: Ý định người dùng hợp lệ & an toàn",
    }


def check_nemo_output_rail(
    user_input: str,
    bot_response: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict:
    """Kiểm tra tầng NeMo Output Rails:
       1) vinbank leak check (deterministic: decoded views, rot13, base64, squashed)
       2) self check output (LLM Semantic Leak Check)
    """
    if not api_key:
        api_key = get_openai_api_key()
    if not model_name:
        model_name = get_red_model()

    # 1. Deterministic check: protected values
    if mentions_protected_value(bot_response):
        return {
            "allowed": False,
            "rail": "vinbank leak check",
            "message": (
                "Response withheld by NeMo Guardrails (output rail: vinbank leak check). "
                "How else can I help with your VinBank banking needs?"
            ),
            "detail": "Phát hiện secret bị rò rỉ (kể cả mã hóa, đảo ngược, squashed)!",
        }

    # 2. LLM self check output
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = (
                SELF_CHECK_OUTPUT_PROMPT.replace("{{ user_input }}", user_input).replace(
                    "{{ bot_response }}", bot_response
                )
            )
            call_kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
            }
            try:
                comp = client.chat.completions.create(**call_kwargs)
            except Exception as e:
                if "temperature" in str(e).lower():
                    call_kwargs["temperature"] = 1.0
                    comp = client.chat.completions.create(**call_kwargs)
                else:
                    raise
            ans = (comp.choices[0].message.content or "").strip().lower()
            if "yes" in ans:
                return {
                    "allowed": False,
                    "rail": "self check output",
                    "message": (
                        "Response withheld by NeMo Guardrails (output rail: self check output). "
                        "How else can I help with your VinBank banking needs?"
                    ),
                    "detail": "LLM phân tích phản hồi: Phát hiện tiết lộ một phần / gợi ý cấu hình nội bộ!",
                }
        except Exception as e:
            print(f"[NeMo Output Rail Error]: {e}")

    return {
        "allowed": True,
        "rail": None,
        "message": None,
        "detail": "NeMo Output Rail: Phản hồi an toàn, không có secret / PII",
    }
