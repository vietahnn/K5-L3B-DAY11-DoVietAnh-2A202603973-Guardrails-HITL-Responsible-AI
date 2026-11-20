"""
VinBank Presentation Server — Phục vụ Slide thuyết trình & Chatbot Live Demo
Chạy: python presentation_server.py
Truy cập: http://localhost:8000
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Thêm src/ vào sys.path để tái sử dụng toàn bộ code đã viết
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import (
    get_red_model,
    get_openai_api_key,
    get_openrouter_api_key,
    DEMO_SECRETS,
)
from guardrails.input_guardrails import detect_injection, topic_filter
from guardrails.output_guardrails import content_filter
from assignment.rate_limiter import RateLimitPlugin
from agents.agent import BLUE_INSTRUCTION
from agents.agent_with_nemo import (
    check_nemo_input_rail,
    check_nemo_output_rail,
)

# Khởi tạo instance Rate Limiter toàn cục (10 requests / 60 giây)
live_rate_limiter = RateLimitPlugin(max_requests=10, window_seconds=60)


def process_chat_message(user_input: str, user_id: str = "guest_student", use_nemo: bool = False) -> dict:
    """Xử lý tin nhắn qua Guardrail Pipeline (chọn Regex thông thường hoặc NVIDIA NeMo Guardrails)."""
    stages = []
    start_total = time.time()
    user_input = user_input.strip()
    model_name = get_red_model()
    api_key = get_openai_api_key()

    # -------------------------------------------------------------
    # STAGE 1: Rate Limiter (Chống DoS / Brute-force)
    # -------------------------------------------------------------
    now = time.time()
    window = live_rate_limiter.user_windows[user_id]
    while window and window[0] <= now - live_rate_limiter.window_seconds:
        window.popleft()

    if len(window) >= live_rate_limiter.max_requests:
        wait = live_rate_limiter.window_seconds - (now - window[0])
        live_rate_limiter.blocked_count += 1
        stages.append({
            "id": "rate_limiter",
            "name": "Rate Limiter",
            "status": "BLOCK",
            "time_ms": round((time.time() - now) * 1000, 2),
            "detail": f"Vượt ngưỡng 10 req/60s! Chờ {wait:.0f}s.",
        })
        input_id = "nemo_input" if use_nemo else "input_guardrail"
        input_name = "NeMo Input Rail" if use_nemo else "Input Guardrail"
        output_id = "nemo_output" if use_nemo else "output_guardrail"
        output_name = "NeMo Output Rail" if use_nemo else "Output Guardrail"
        stages.append({"id": input_id, "name": input_name, "status": "SKIPPED", "time_ms": 0, "detail": "Bị ngắt sớm"})
        stages.append({"id": "llm", "name": "LLM Reasoning", "status": "SKIPPED", "time_ms": 0, "detail": "Không gọi mô hình"})
        stages.append({"id": output_id, "name": output_name, "status": "SKIPPED", "time_ms": 0, "detail": "Bị ngắt sớm"})
        return {
            "response": f"⚠️ Rate limit exceeded. Bạn đã gửi quá 10 câu trong 60 giây. Vui lòng thử lại sau {wait:.0f}s.",
            "decision": "BLOCKED",
            "blocked_at": "Rate Limiter",
            "total_latency_ms": round((time.time() - start_total) * 1000, 2),
            "stages": stages,
        }

    window.append(now)
    stages.append({
        "id": "rate_limiter",
        "name": "Rate Limiter",
        "status": "PASS",
        "time_ms": round((time.time() - now) * 1000, 2),
        "detail": f"Hợp lệ ({len(window)}/{live_rate_limiter.max_requests} requests)",
    })

    # =============================================================
    # CHẾ ĐỘ 1: SỬ DỤNG NVIDIA NEMO GUARDRAILS (AI SEMANTIC)
    # =============================================================
    if use_nemo:
        # --- STAGE 2: NeMo Input Rails ---
        t_nemo_in = time.time()
        nemo_in = check_nemo_input_rail(user_input, api_key, model_name)
        t_nemo_in_ms = round((time.time() - t_nemo_in) * 1000, 2)

        if not nemo_in["allowed"]:
            stages.append({
                "id": "nemo_input",
                "name": "NeMo Input Rail",
                "status": "BLOCK",
                "time_ms": t_nemo_in_ms,
                "detail": nemo_in["detail"],
            })
            stages.append({"id": "llm", "name": "LLM Reasoning", "status": "SKIPPED", "time_ms": 0, "detail": "Không gọi mô hình"})
            stages.append({"id": "nemo_output", "name": "NeMo Output Rail", "status": "SKIPPED", "time_ms": 0, "detail": "Bị ngắt sớm"})
            return {
                "response": f"🛡️ {nemo_in['message']}",
                "decision": "BLOCKED",
                "blocked_at": f"NeMo Input Rail ({nemo_in['rail']})",
                "total_latency_ms": round((time.time() - start_total) * 1000, 2),
                "stages": stages,
            }

        stages.append({
            "id": "nemo_input",
            "name": "NeMo Input Rail",
            "status": "PASS",
            "time_ms": t_nemo_in_ms,
            "detail": nemo_in["detail"],
        })

        # --- STAGE 3: LLM Reasoning ---
        t_llm = time.time()
        raw_response = ""
        try:
            if api_key:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                call_kwargs = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": BLUE_INSTRUCTION},
                        {"role": "user", "content": user_input},
                    ],
                }
                try:
                    comp = client.chat.completions.create(**call_kwargs)
                except Exception as e:
                    if "temperature" in str(e).lower():
                        call_kwargs["temperature"] = 1.0
                        comp = client.chat.completions.create(**call_kwargs)
                    else:
                        raise
                raw_response = comp.choices[0].message.content or ""
            else:
                raw_response = (
                    "Chào bạn, tôi là trợ lý ảo VinBank. Lãi suất tiết kiệm kỳ hạn 12 tháng hiện tại là 4.25%/năm. "
                    "Bạn có thể mở sổ tiết kiệm tại quầy hoặc qua ứng dụng VinBank Mobile."
                )
        except Exception as exc:
            raw_response = f"[Lỗi gọi API {model_name}]: {exc}"

        t_llm_ms = round((time.time() - t_llm) * 1000, 2)
        stages.append({
            "id": "llm",
            "name": "LLM Reasoning",
            "status": "PASS",
            "time_ms": t_llm_ms,
            "detail": f"Mô hình {model_name} đã sinh phản hồi",
        })

        # --- STAGE 4: NeMo Output Rails ---
        t_nemo_out = time.time()
        nemo_out = check_nemo_output_rail(user_input, raw_response, api_key, model_name)
        t_nemo_out_ms = round((time.time() - t_nemo_out) * 1000, 2)

        if not nemo_out["allowed"]:
            stages.append({
                "id": "nemo_output",
                "name": "NeMo Output Rail",
                "status": "BLOCK",
                "time_ms": t_nemo_out_ms,
                "detail": nemo_out["detail"],
            })
            return {
                "response": f"🛡️ {nemo_out['message']}",
                "decision": "BLOCKED",
                "blocked_at": f"NeMo Output Rail ({nemo_out['rail']})",
                "total_latency_ms": round((time.time() - start_total) * 1000, 2),
                "stages": stages,
            }

        stages.append({
            "id": "nemo_output",
            "name": "NeMo Output Rail",
            "status": "PASS",
            "time_ms": t_nemo_out_ms,
            "detail": nemo_out["detail"],
        })

        return {
            "response": raw_response,
            "decision": "SAFE",
            "blocked_at": None,
            "total_latency_ms": round((time.time() - start_total) * 1000, 2),
            "stages": stages,
        }

    # =============================================================
    # CHẾ ĐỘ 2: SỬ DỤNG GUARDRAILS CƠ BẢN (REGEX & TOPIC)
    # =============================================================
    t_input = time.time()
    inj_status = detect_injection(user_input)
    top_status = topic_filter(user_input)
    t_input_ms = round((time.time() - t_input) * 1000, 2)

    if inj_status == "BLOCK":
        stages.append({
            "id": "input_guardrail",
            "name": "Input Guardrail",
            "status": "BLOCK",
            "time_ms": t_input_ms,
            "detail": "Phát hiện Prompt Injection / Jailbreak!",
        })
        stages.append({"id": "llm", "name": "LLM Reasoning", "status": "SKIPPED", "time_ms": 0, "detail": "Không gọi mô hình"})
        stages.append({"id": "output_guardrail", "name": "Output Guardrail", "status": "SKIPPED", "time_ms": 0, "detail": "Bị ngắt sớm"})
        return {
            "response": "🛡️ I cannot process that request. Prompt injection detected. (VinBank Security Guardrails)",
            "decision": "BLOCKED",
            "blocked_at": "Input Guardrail (Injection)",
            "total_latency_ms": round((time.time() - start_total) * 1000, 2),
            "stages": stages,
        }

    if top_status == "BLOCK":
        stages.append({
            "id": "input_guardrail",
            "name": "Input Guardrail",
            "status": "BLOCK",
            "time_ms": t_input_ms,
            "detail": "Chủ đề cấm hoặc không thuộc nghiệp vụ ngân hàng!",
        })
        stages.append({"id": "llm", "name": "LLM Reasoning", "status": "SKIPPED", "time_ms": 0, "detail": "Không gọi mô hình"})
        stages.append({"id": "output_guardrail", "name": "Output Guardrail", "status": "SKIPPED", "time_ms": 0, "detail": "Bị ngắt sớm"})
        return {
            "response": "🛡️ I can only help with VinBank banking-related questions. Vui lòng chỉ hỏi về tài khoản, tiết kiệm, lãi suất, thẻ tín dụng hoặc chuyển tiền.",
            "decision": "BLOCKED",
            "blocked_at": "Input Guardrail (Topic)",
            "total_latency_ms": round((time.time() - start_total) * 1000, 2),
            "stages": stages,
        }

    stages.append({
        "id": "input_guardrail",
        "name": "Input Guardrail",
        "status": "PASS",
        "time_ms": t_input_ms,
        "detail": "Prompt an toàn & Đúng chủ đề VinBank",
    })

    # STAGE 3: LLM Reasoning
    t_llm = time.time()
    raw_response = ""
    try:
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            call_kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": BLUE_INSTRUCTION},
                    {"role": "user", "content": user_input},
                ],
            }
            try:
                comp = client.chat.completions.create(**call_kwargs)
            except Exception as e:
                if "temperature" in str(e).lower():
                    call_kwargs["temperature"] = 1.0
                    comp = client.chat.completions.create(**call_kwargs)
                else:
                    raise
            raw_response = comp.choices[0].message.content or ""
        else:
            raw_response = (
                "Chào bạn, tôi là trợ lý ảo VinBank. Lãi suất tiết kiệm kỳ hạn 12 tháng hiện tại là 4.25%/năm. "
                "Bạn có thể mang CCCD đến phòng giao dịch gần nhất hoặc mở tài khoản trực tuyến qua ứng dụng VinBank Mobile."
            )
    except Exception as exc:
        raw_response = f"[Lỗi gọi API {model_name}]: {exc}"

    t_llm_ms = round((time.time() - t_llm) * 1000, 2)
    stages.append({
        "id": "llm",
        "name": "LLM Reasoning",
        "status": "PASS",
        "time_ms": t_llm_ms,
        "detail": f"Mô hình {model_name} đã sinh phản hồi",
    })

    # STAGE 4: Output Guardrails (Regex Redact)
    t_out = time.time()
    out_result = content_filter(raw_response)
    t_out_ms = round((time.time() - t_out) * 1000, 2)

    if not out_result["safe"]:
        stages.append({
            "id": "output_guardrail",
            "name": "Output Guardrail",
            "status": "REDACTED",
            "time_ms": t_out_ms,
            "detail": f"Đã che giấu: {', '.join(out_result['issues'])}",
        })
        final_response = out_result["redacted"]
        decision = "REDACTED"
    else:
        stages.append({
            "id": "output_guardrail",
            "name": "Output Guardrail",
            "status": "PASS",
            "time_ms": t_out_ms,
            "detail": "Không phát hiện rò rỉ PII / Secret",
        })
        final_response = raw_response
        decision = "SAFE"

    return {
        "response": final_response,
        "decision": decision,
        "blocked_at": None,
        "total_latency_ms": round((time.time() - start_total) * 1000, 2),
        "stages": stages,
    }


class PresentationHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Trỏ URL gốc '/' vào presentation.html
        if self.path == "/" or self.path == "/index.html":
            self.path = "/presentation.html"
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                user_msg = data.get("message", "")
                user_id = data.get("user_id", "guest_student")
                use_nemo = bool(data.get("use_nemo", False))
                result = process_chat_message(user_msg, user_id=user_id, use_nemo=use_nemo)
                res_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.end_headers()
                self.wfile.write(res_bytes)
            except Exception as e:
                err_bytes = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, PresentationHandler)
    print("=" * 60)
    print(f"🚀 VINBANK PRESENTATION & LIVE CHATBOT ĐANG CHẠY TẠI:")
    print(f"👉 http://localhost:{port}")
    print(f"👉 http://127.0.0.1:{port}")
    print("=" * 60)
    print("Nhấn Ctrl + C để dừng server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã tắt server.")
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
