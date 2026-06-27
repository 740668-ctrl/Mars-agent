import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_INSTRUCTIONS = "You are a helpful AI assistant."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def run_agent():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({"error": "Server is missing OPENAI_API_KEY. Set it in your .env file."}), 500

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
        )
        output = response.choices[0].message.content
        return jsonify({"output": output})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
