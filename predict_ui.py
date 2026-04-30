"""
Gradio UI for character-level language model prediction

Loads a trained makemore or nanoGPT checkpoint and provides an interactive UI
to show top-k next character predictions with probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import sys

# Import model definitions
# We'll use the same model classes as in training

# Makemore model
class MakemoreModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, block_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * block_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        emb = self.embedding(x)
        emb = emb.view(B, -1)
        hidden = F.relu(self.fc1(emb))
        logits = self.fc2(hidden)
        return logits

# Load checkpoint
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'checkpoints/makemore_model.pt'

try:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    print(f"Loaded checkpoint from {checkpoint_path}")
except FileNotFoundError:
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    print("Please train the model first by running: python train_makemore.py")
    sys.exit(1)

# Reconstruct model
model = MakemoreModel(
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=checkpoint['embedding_dim'],
    hidden_dim=checkpoint['hidden_dim'],
    block_size=checkpoint['block_size']
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get vocabulary mappings
stoi = checkpoint['stoi']
itos = checkpoint['itos']
block_size = checkpoint['block_size']

def encode(s):
    return [stoi.get(c, 0) for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Prediction function
def predict_next_chars(input_text, top_k=5):
    """Predict top-k next characters and their probabilities"""

    if not input_text:
        return "Please enter some text to get predictions!"

    # Prepare input
    # Take last block_size characters
    context = input_text[-block_size:]

    # Pad if necessary
    if len(context) < block_size:
        context = '\n' * (block_size - len(context)) + context

    # Encode
    encoded = encode(context)
    x = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    # Get predictions
    with torch.no_grad():
        logits = model(x)  # (1, vocab_size)
        probs = F.softmax(logits[0], dim=-1)  # (vocab_size,)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

    # Format output
    result = f"Input context (last {block_size} chars): '{context}'\n\n"
    result += "Top predictions for next character:\n"
    result += "-" * 50 + "\n"

    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        char = itos[idx.item()]
        if char == '\n':
            char_display = '\\n (newline)'
        elif char == ' ':
            char_display = '(space)'
        else:
            char_display = f"'{char}'"
        result += f"{i+1}. {char_display:15s} - {prob.item():.2%}\n"

    return result

# Generate text function
def generate_text(seed_text, num_chars=100, temperature=1.0):
    """Generate new text starting from seed text"""

    if not seed_text:
        seed_text = '\n'

    # Prepare input
    context = seed_text[-block_size:]
    if len(context) < block_size:
        context = '\n' * (block_size - len(context)) + context

    generated = seed_text

    with torch.no_grad():
        for _ in range(num_chars):
            # Encode context
            encoded = encode(context[-block_size:])
            x = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

            # Get predictions
            logits = model(x)
            logits = logits / temperature
            probs = F.softmax(logits[0], dim=-1)

            # Sample next character
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = itos[next_idx]

            # Append to generated text
            generated += next_char
            context = context[1:] + next_char

    return generated

# Create Gradio interface
with gr.Blocks(title="Makemore Character Predictor") as demo:
    gr.Markdown("# Makemore Character-Level Language Model")
    gr.Markdown(f"Model trained on Shakespeare's works. Context window: {block_size} characters")
    gr.Markdown(f"Training loss: {checkpoint.get('train_loss', 'N/A'):.4f} | Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    with gr.Tab("Predict Next Character"):
        gr.Markdown("Enter some text and see what the model predicts as the next character.")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="To be or not to",
                    lines=3
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of predictions to show"
                )
                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                prediction_output = gr.Textbox(
                    label="Predictions",
                    lines=10
                )

        predict_btn.click(
            fn=predict_next_chars,
            inputs=[input_text, top_k],
            outputs=prediction_output
        )

        gr.Examples(
            examples=[
                ["To be or not to"],
                ["ROMEO:\n"],
                ["First Citizen:"],
                ["The king"],
            ],
            inputs=input_text
        )

    with gr.Tab("Generate Text"):
        gr.Markdown("Generate new text starting from a seed phrase.")

        with gr.Row():
            with gr.Column():
                seed_text = gr.Textbox(
                    label="Seed Text",
                    placeholder="ROMEO:\n",
                    lines=2
                )
                num_chars = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Number of characters to generate"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature (higher = more random)"
                )
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                generated_output = gr.Textbox(
                    label="Generated Text",
                    lines=15
                )

        generate_btn.click(
            fn=generate_text,
            inputs=[seed_text, num_chars, temperature],
            outputs=generated_output
        )

if __name__ == "__main__":
    print("Starting Gradio UI...")
    print(f"Model: Makemore ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters)")
    demo.launch(share=False)
