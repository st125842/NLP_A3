import dash
from dash import dcc, html, Input, Output, State
import torch
from torch import nn
import torch.nn.functional as F
import random
import spacy
from pythainlp.tokenize import word_tokenize

# Load Spacy
nlp_en = spacy.load("en_core_web_sm")

# --- Your Preprocessing Functions ---
def preprocess_th(text):
    return word_tokenize(text, engine="newmm")

def preprocess_en(text):
    doc = nlp_en(text.lower())
    # Note: Ensure this logic matches your training exactly (e.g., handling of punctuation)
    tokens = [tok.text for tok in doc if not tok.is_punct]
    return tokens

class GeneralAttentionModule(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Linear(hid_dim, hid_dim)
    def forward(self, decoder_hidden, encoder_outputs):
        # Calls your general_attention function using its internal weights
        return general_attention(decoder_hidden, encoder_outputs, self.w)

def general_attention(decoder_hidden, encoder_outputs, weight_matrix):
    """
    decoder_hidden: [batch_size, hid_dim]
    encoder_outputs: [src_len, batch_size, hid_dim]
    weight_matrix: nn.Linear(hid_dim, hid_dim)
    """
    # 1. Permute encoder to [batch, src_len, hid_dim]
    encoder_outputs = encoder_outputs.permute(1, 0, 2)
    
    # 2. Project encoder outputs: [batch, src_len, hid_dim]
    projected_encoder = weight_matrix(encoder_outputs)
    
    # 3. Dot product (bmm): [batch, src_len, hid_dim] * [batch, hid_dim, 1]
    # This is s^T * W * h
    scores = torch.bmm(projected_encoder, decoder_hidden.unsqueeze(2))
    
    # 4. Normalize to probabilities
    return F.softmax(scores.squeeze(2), dim=1)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [seq len, batch size]
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        #outputs = [seq len,  batch size, hid dim]
        #hidden  = [n layers, batch size, hid dim]
        #cell    = [n layers, batch size, hid dim]
        
        #outputs are always from the most top hidden layer
        return outputs,hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention # This is the attention module/function
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # The GRU/LSTM now takes (embedding + context_vector) as input
        # Context vector has size 'hid_dim' (same as encoder hidden)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout = dropout)
        
        # Prediction layer takes [RNN_output; context_vector; embedding]
        self.fc_out = nn.Linear(hid_dim + hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input = [batch size]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim]

        input = input.unsqueeze(0) # [1, batch size]
        embedded = self.dropout(self.embedding(input)) # [1, batch size, emb dim]
        # 1. Calculate Attention Weights (Alpha)
        # We use the top layer hidden state to calculate attention: hidden[-1]
        a = self.attention(hidden[-1], encoder_outputs) 
        # a = [batch size, src len]
        # print(a.shape)
        # 2. Create Context Vector (Weighted sum of encoder outputs)
        # [batch, 1, src len] * [batch, src len, hid dim] -> [batch, 1, hid dim]
        a = a.unsqueeze(1) 
        # print(a.shape)
        h_enc = encoder_outputs.permute(1, 0, 2)
        # print(encoder_outputs.shape)
        # print(h_enc.shape)
        context = torch.bmm(a, h_enc) 
        # print(context.shape)
        # print('context------')
        # context = [batch, 1, hid dim]
        
        # 3. Concatenate Embedding and Context for RNN input
        # context.permute(1, 0, 2) -> [1, batch, hid dim]
        rnn_input = torch.cat((embedded, context.permute(1, 0, 2)), dim=2)
        
        # 4. Feed to RNN
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # 5. Final Prediction
        # We combine the RNN output, context, and original embedding for a strong prediction
        prediction = self.fc_out(torch.cat((output, context.permute(1, 0, 2), embedded), dim=2).squeeze(0))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len    = trg.shape[0]
        trg_output_dim = self.decoder.output_dim
        
        #tensor to store decoder outputs
        #this will make outputs[0] to become all zeros....
        outputs = torch.zeros(trg_len, batch_size, trg_output_dim).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        #recall that this hidden is the final state of each layer
        en_out,hidden, cell = self.encoder(src)
        #hidden = [n layers, batch size, hid dim]
        #cell   = [n layers, batch size, hid dim]
        
        #first input to the decoder is the <sos> tokens
        #recall that the decoder is per token, not the whole sequence
        input_ = trg[0,:]
        #input_ = [batch_size]
        
        for t in range(1, trg_len):
            
            #insert input token, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input_, hidden, cell,en_out)
            #output = [batch size, output dim]
            #hidden = [n layers, batch size, hid dim]
            #cell   = [n layers, batch size, hid dim]
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input_ = trg[t] if teacher_force else top1
                    
        return outputs


# --- Setup Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("English to Thai Translator", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Textarea(
            id='input-text',
            placeholder='Type English sentence here...',
            style={'width': '100%', 'height': 100}
        ),
        html.Br(),
        html.Button('Translate', id='translate-button', n_clicks=0, 
                    style={'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px 20px'})
    ], style={'padding': '20px'}),
    
    html.Hr(),
    
    html.Div([
        html.H3("Thai Translation:"),
        html.Div(id='output-translation', style={'fontSize': '24px', 'color': '#28a745', 'fontWeight': 'bold'})
    ], style={'padding': '20px'})
])

def translate_sentence(sentence, model, vocab_en, vocab_th, device):
    model.eval()
    
    # --- 1. PREPROCESS (English) ---
    # Convert string to list of tokens
    tokens = preprocess_en(sentence)

    # --- 2. CONVERT TO INDICES (English) ---
    # Get the String-to-Index dictionary (stoi)
    stoi_en = vocab_en.get_stoi()
    
    # Look up tokens safely
    unk_idx = stoi_en['<unk>']
    ids = [stoi_en.get(token, unk_idx) for token in tokens]
    
    # Add <sos> and <eos>
    ids = [stoi_en['<sos>']] + ids + [stoi_en['<eos>']]
    
    # Create Source Tensor
    src_tensor = torch.LongTensor(ids).unsqueeze(1).to(device) # [src_len, 1]

    # --- 3. INFERENCE ---
    # Get the String-to-Index dictionary for Thai
    stoi_th = vocab_th.get_stoi()
    
    # Create dummy target (only <sos> needed to start)
    trg_placeholder = torch.zeros(20, 1).long().to(device)
    trg_placeholder[0] = stoi_th['<sos>'] 

    with torch.no_grad():
        # Run model (teacher_forcing_ratio = 0 for inference)
        predictions = model(src_tensor, trg_placeholder, teacher_forcing_ratio=0)

    # --- 4. CONVERT BACK TO WORDS (Thai) ---
    # Get the Index-to-String LIST (itos) - This fixes your error!
    itos_th = vocab_th.get_itos()
    
    # Get the highest probability token for each step
    top1 = predictions.argmax(2) 
    
    translated_tokens = []
    
    # Skip the first token (it's always <sos>)
    for i in range(1, top1.shape[0]):
        idx = top1[i].item()
        
        # Stop if we hit <eos>
        if idx == stoi_th['<eos>']:
            break
            
        # Add the word from the list
        translated_tokens.append(itos_th[idx])
        
    # Join Thai tokens (no spaces)
    return "".join(translated_tokens)
    # return 'good'
# --- Callback ---
@app.callback(
    Output('output-translation', 'children'),
    Input('translate-button', 'n_clicks'),
    State('input-text', 'value'),
    prevent_initial_call=True
)
def update_output(n_clicks, input_value):
    if input_value is None or input_value.strip() == "":
        return "Please enter some text."
    
    # Call your translation function
    # try:
    result = translate_sentence(input_value, model, vocab_en, vocab_th, device)
    return result
    # except Exception as e:
    #     return f"Error: {str(e)}"

if __name__ == '__main__':
    # Load your model and vocabs here before running
    vocab_th = torch.load('vocab_th.pt')
    vocab_en = torch.load('vocab_en.pt')
    input_dim   = len(vocab_en)
    output_dim  = len(vocab_th)
    emb_dim     = 1024  
    hid_dim     = 1024
    n_layers    = 2   
    dropout     = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    general_attn = GeneralAttentionModule(hid_dim).to(device)

    enc = Encoder(input_dim,  emb_dim, hid_dim, n_layers, dropout)
    dec = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout,general_attn)

    model = Seq2Seq(enc, dec, device).to(device)

    model.load_state_dict(torch.load('seq2seqGeneralAttention.pt'))
    app.run(debug=True,use_reloader=False)