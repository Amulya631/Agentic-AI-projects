"""
Full-Screen Gradio UI — Pet & Animal Needs Assistant
"""

from google import genai
from google.genai import types
import gradio as gr

# ---- Vertex AI Client ----
client = genai.Client(
    vertexai=True,
    project="agentic-ai-chatbot-476413",
    location="global",
)

# ---- Model call ----
def get_pet_response(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                ],
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ---- Gradio Interface ----
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="teal"),
    css="""
    html, body, #root, .gradio-container {
        height: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        background-color: #0e1117 !important;
    }

    .gradio-container {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    h1 {
        text-align: center;
        color: #f0f0f0;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    #chatbot {
        flex-grow: 1;
        height: calc(100vh - 220px) !important;
        overflow-y: auto;
        border-radius: 12px;
        background-color: #1b1f27 !important;
        padding: 10px;
    }

    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.9em;
        margin: 8px 0;
    }

    button {
        border-radius: 10px !important;
        font-weight: 600;
        height: 3em !important;
    }
    """,
) as demo:

    gr.Markdown("<h1>🐾 Pet & Animal Needs Assistant</h1>")

    chatbot = gr.Chatbot(
        label="Chat with your friendly pet care expert 🐕🐈",
        elem_id="chatbot",
        height=600,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your question... e.g. 'What fruits are safe for dogs?'",
            show_label=False,
            scale=5,
        )
        send = gr.Button("Send ✨", scale=1)

    def respond(message, history):
        if not message.strip():
            return "", history
        reply = get_pet_response(message)
        history.append((message, reply))
        return "", history

    send.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    gr.Markdown("<div class='footer'>Built with ❤️ using Vertex AI & Gradio</div>")

demo.launch(server_name="0.0.0.0", server_port=8080)
