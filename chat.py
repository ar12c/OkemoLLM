import torch
import torch.nn as nn
import random
import os
import sys

# Import custom classes from the training file
# Make sure train.py is in the same directory and has the correct classes
try:
    from train import Voc, PositionalEncoding, TransformerEncoder, TransformerDecoder
except ImportError:
    print("❌ Error: Could not import classes from train.py.")
    print("Please ensure train.py is in the same directory.")
    sys.exit()

# ----------------- Set up device -----------------
# Check if CUDA (NVIDIA GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Load data and models -----------------
# Define the directory where the model is saved
model_dir = "model"

# Load vocabulary
try:
    with open(os.path.join(model_dir, "voc.pth"), 'rb') as f:
        # We must set weights_only=False to load the custom Voc class
        voc = torch.load(f, map_location=device, weights_only=False)
    print(f"✅ Vocabulary loaded with {voc.n_words} words")
except Exception as e:
    print("❌ Could not load voc.pth.")
    print(f"Error: {e}")
    sys.exit()

# Hyperparameters - These must match the values used during training!
d_model = 2048
n_head = 8
d_ff = 2048
encoder_n_layers = 8
decoder_n_layers = 8
dropout = 0.1
max_length = 200 # This should match the training max_length

# Initialize model components
try:
    # Initialize the encoder and decoder with the correct hyperparameters
    encoder = TransformerEncoder(voc.n_words, d_model, encoder_n_layers, n_head, d_ff, dropout).to(device)
    decoder = TransformerDecoder(voc.n_words, d_model, decoder_n_layers, n_head, d_ff, dropout).to(device)

    # Load the state dictionaries. We also set weights_only=False here for consistency.
    encoder.load_state_dict(torch.load(os.path.join(model_dir, 'encoder.pth'), map_location=device, weights_only=False))
    decoder.load_state_dict(torch.load(os.path.join(model_dir, 'decoder.pth'), map_location=device, weights_only=False))
    
    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()

    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Could not load model files (encoder.pth, decoder.pth).")
    print("Please ensure you have run train.py to create these files.")
    print(f"Error: {e}")
    sys.exit()

# ----------------- Evaluation Function -----------------
def evaluate(encoder, decoder, voc, sentence, max_length=10):
    """
    Takes an input sentence, generates an output response from the model.
    """
    with torch.no_grad():
        # Convert the sentence to a tensor
        input_tensor = torch.LongTensor(indexes_from_sentence(voc, sentence)).view(-1, 1).to(device)
        
        # Pass input through the encoder
        encoder_outputs = encoder(input_tensor)

        # Initialize the decoder input with the SOS token
        decoder_input = torch.tensor([[voc.word2index['<SOS>']]], device=device)

        decoded_words = []

        for i in range(max_length):
            # Generate a causal mask for the decoder
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.size(0)).to(device)
            
            # Get the decoder output
            decoder_output = decoder(decoder_input, encoder_outputs, tgt_mask=tgt_mask)
            
            # Get the top predicted word from the last output
            topv, topi = decoder_output[-1].topk(1)
            
            # Debugging print to see the first predicted token
            if i == 0:
                print(f"DEBUG: First predicted token index is {topi.item()}")

            # Check if the output is the end-of-sentence token
            if topi.item() == voc.word2index['<EOS>']:
                break
            else:
                decoded_words.append(voc.index2word[topi.item()])
            
            # Use the predicted word as the next input
            decoder_input = torch.cat([decoder_input, topi.squeeze().detach().view(1, 1)])

        return decoded_words

def indexes_from_sentence(voc, sentence):
    """Converts a sentence string to a list of word indexes."""
    return [voc.word2index.get(word, voc.word2index['<UNK>']) for word in sentence.split(' ')] + [voc.word2index['<EOS>']]

# ----------------- Main chat loop -----------------
def chat_loop():
    print("🤖 Transformer Chatbot is ready! Type 'quit' to exit.")
    while True:
        try:
            input_sentence = input("> ")
            if input_sentence.lower() == 'quit':
                break
            
            output_words = evaluate(encoder, decoder, voc, input_sentence)
            output_sentence = ' '.join(output_words)

            if not output_sentence:
                print("OkemoLLM: I couldn't generate a meaningful response. Please try again.")
            else:
                print("OkemoLLM:", output_sentence)
        except Exception as e:
            print("❌ An error occurred during chat:", e)

if __name__ == "__main__":
    chat_loop()
