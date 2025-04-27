# lambda/index.py
import json
import os
import re
import urllib.request
import urllib.error

# ---------- 設定 ----------
INFERENCE_URL = os.environ.get(
    "INFERENCE_URL",   # CDK かマネジメントコンソールで設定しておく
    "https://df52-34-143-237-29.ngrok-free.app/generate"
)
# デフォルトの生成パラメータ。必要なら環境変数にしても OK
GEN_CFG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}

# ---------- 共通ヘッダー ----------
CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": (
        "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token"
    ),
    "Access-Control-Allow-Methods": "OPTIONS,POST",
}


# ---------- Lambda ハンドラ ----------
def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # --- リクエスト取り出し ---
        body = json.loads(event["body"])
        message: str = body["message"]
        conversation_history = body.get("conversationHistory", [])
        print("User message:", message)

        # --- prompt 構築（最小構成） ---
        #   まずは最新メッセージだけを送る
        prompt = message

        # --- FastAPI 推論エンドポイント呼び出し ---
        payload = dict(GEN_CFG, prompt=prompt)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            INFERENCE_URL,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_body = resp.read().decode("utf-8")
                print("Inference raw response:", resp_body)
                resp_json = json.loads(resp_body)
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Inference API returned {e.code}: {e.read().decode()}"
            ) from e

        # --- 生成テキストを抽出 ---
        assistant_response = (
            resp_json.get("generated_text")
            or resp_json.get("text")
            or resp_json.get("response")
            or resp_json.get("result")
        )
        if not assistant_response:
            raise ValueError("No text found in inference response")

        # --- 会話履歴を拡張して返却 ---
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps(
                {
                    "success": True,
                    "response": assistant_response,
                    "conversationHistory": conversation_history,
                }
            ),
        }

    except Exception as err:
        print("Error:", err)
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"success": False, "error": str(err)}),
        }

