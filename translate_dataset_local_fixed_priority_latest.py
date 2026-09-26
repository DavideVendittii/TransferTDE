q#!/usr/bin/env python
# coding: utf-8

# In[1]:
import json
from pathlib import Path
ENV = json.loads(Path(__file__).with_name("env.json").read_text(encoding="utf-8"))

import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# ## Configuration - edit before running

# In[2]:


# --- Access tokens ---
TOKEN_HF = ENV["HF_TOKEN_TRANSLATION"]

# --- What to translate ---
# TARGET_LANGUAGES = ["de", "it", "fr", "sp"]   # keys of SYSTEM_PROMPTS below - translated one after another
TARGET_LANGUAGES = ["de"]   # keys of SYSTEM_PROMPTS below - translated one after another
PII_TYPES = ["phone"]        #change or add pii to translate them, if alredy translated it will skip them  

# --- Paths ---
DATA_FOLDER = "data"

# --- Model / translation tuning (same defaults as the original notebooks) ---
GEMMA_TOKENIZER_MODEL = "google/gemma-3-27b-it" #f"meta-llama/Llama-3.2-1B" #"google/gemma-3-27b-it"   # also used as the local translation model
CONTEXT_SIZE = 350
MARGIN_SIZE = 50
MAX_NEW_TOKENS = 1024

# --- Batching vs checkpointing ---
GENERATION_BATCH_SIZE = 4   


# ## System prompts per target language (verbatim from the original notebooks)

# In[3]:


SYSTEM_PROMPTS = {
    "it": """
Sei un traduttore dall'inglese all'italiano.
Devi seguire RIGOROSAMENTE queste regole:
1. <<NEWLINE>> Ã¨ un marcatore speciale che rappresenta un ritorno a capo.
NON tradurlo, NON rimuoverlo, NON spostarlo.
Ogni <<NEWLINE>> nell'input DEVE apparire nella STESSA IDENTICA
posizione nell'output.
2. Traduci SOLO le parole inglesi in italiano. Tutti gli altri caratteri
(punteggiatura, spazi, simboli come > < @ #, emoji)
devono rimanere ESATTAMENTE dove sono.
3. NON aggiungere punti finali, virgole o altri segni di punteggiatura che non
sono nell'originale.
4. L'output deve contenere SOLO la traduzione, senza introduzioni,
spiegazioni o commenti.
Esempi:
Input: "Hello world<<NEWLINE>>How are you?"
Output: "Ciao mondo<<NEWLINE>>Come stai?"
Input: "Email: user@example.com<<NEWLINE>>Phone: 555-1234"
Output: "Email: utente@esempio.com<<NEWLINE>>Telefono: 555-1234"
Input: "Visit <website><<NEWLINE>>for more info"
Output: "Visita <sito web><<NEWLINE>>per maggiori informazioni"
""",
    "de": """
Du bist ein Ãœbersetzer vom Englischen ins Deutsche. Du musst folgende Regeln STRENG befolgen:
1. <<NEWLINE>> ist eine spezielle Markierung fÃ¼r einen Zeilenumbruch. NICHT Ã¼bersetzen, NICHT entfernen, NICHT verschieben. Jedes <<NEWLINE>> in der Eingabe MUSS an der EXAKT GLEICHEN Position in der Ausgabe erscheinen.
2. Ãœbersetze NUR die englischen WÃ¶rter ins Deutsche. Alle anderen Zeichen (Satzzeichen, Leerzeichen, Symbole wie > < @ #, Emojis) mÃ¼ssen EXAKT dort bleiben, wo sie sind.
3. FÃ¼ge KEINE Punkte, Kommas oder andere Satzzeichen hinzu, die nicht im Original vorhanden sind.
4. Die Ausgabe darf NUR die Ãœbersetzung enthalten, ohne Einleitungen, ErklÃ¤rungen oder Kommentare.
Beispiele:
Input: "Hello world<<NEWLINE>>How are you?"
Output: "Hallo Welt<<NEWLINE>>Wie geht es dir?"
Input: "Email: user@example.com<<NEWLINE>>Phone: 555-1234"
Output: "E-Mail: benutzer@beispiel.com<<NEWLINE>>Telefon: 555-1234"
Input: "Visit <website><<NEWLINE>>for more info"
Output: "Besuche <Webseite><<NEWLINE>>fÃ¼r weitere Informationen"
""",
    "fr": """
Tu es un traducteur de l\'anglais vers le franÃ§ais. Tu dois suivre STRICTEMENT les rÃ¨gles suivantes :
1. <<NEWLINE>> est un marqueur spÃ©cial pour un saut de ligne. Ne PAS traduire, ne PAS supprimer, ne PAS dÃ©placer. Chaque <<NEWLINE>> dans l\'entrÃ©e DOIT apparaÃ®tre Ã  l\'EMPLACEMENT EXACT dans la sortie.
2. Traduis UNIQUEMENT les mots anglais en franÃ§ais. Tous les autres caractÃ¨res (ponctuation, espaces, symboles tels que > < @ #, emojis) doivent rester EXACTEMENT lÃ  oÃ¹ ils sont.
3. Ne PAS ajouter de points, de virgules ou d\'autres signes de ponctuation qui ne sont pas prÃ©sents dans l\'original.
4. La sortie doit contenir UNIQUEMENT la traduction, sans introductions, explications ou commentaires.
Exemples :
Input : "Hello world<<NEWLINE>>How are you?"
Output : "Bonjour le monde<<NEWLINE>>Comment allez-vous ?"
Input : "Email: user@example.com<<NEWLINE>>Phone: 555-1234"
Output : "E-mail : user@example.com<<NEWLINE>>TÃ©lÃ©phone : 555-1234"
Input : "Visit <website><<NEWLINE>>for more info"
Output : "Visitez <site web><<NEWLINE>>pour plus d\'informations"
""",
    "sp": """
Eres un traductor de inglÃ©s a espaÃ±ol. Debes seguir ESTRICTAMENTE las siguientes reglas:
1. <<NEWLINE>> es una marca especial para un salto de lÃ­nea. No debes traducirla, eliminarla ni moverla. Cada <<NEWLINE>> en la entrada DEBE aparecer en la POSICIÃ“N EXACTA en la salida.
2. Traduce ÃšNICAMENTE las palabras en inglÃ©s al espaÃ±ol. Todos los demÃ¡s caracteres (signos de puntuaciÃ³n, espacios, sÃ­mbolos como > < @ #, emojis) deben permanecer EXACTAMENTE donde estÃ¡n.
3. No aÃ±adas puntos, comas u otros signos de puntuaciÃ³n que no estÃ©n presentes en el original.
4. La salida debe contener ÃšNICAMENTE la traducciÃ³n, sin introducciones, explicaciones ni comentarios.
Ejemplos:
Input: "Hello world<<NEWLINE>>How are you?"
Output: "Hola mundo<<NEWLINE>>Â¿CÃ³mo estÃ¡s?"
Input: "Email: user@example.com<<NEWLINE>>Phone: 555-1234"
Output: "Email: usuario@ejemplo.com<<NEWLINE>>TelÃ©fono: 555-1234"
Input: "Visit <website><<NEWLINE>>for more info"
Output: "Visita <sitio web><<NEWLINE>>para mÃ¡s informaciÃ³n"
""",
}

user_prompt_template = """
Traduci questo testo:
{text}
"""


# ## Setup

# In[4]:


from datasets import Dataset, concatenate_datasets
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM,utils
from huggingface_hub import login
utils.logging.set_verbosity_info()
torch.manual_seed(42)
login(token=TOKEN_HF)
tokenizer = AutoTokenizer.from_pretrained(GEMMA_TOKENIZER_MODEL, use_fast=False)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForCausalLM.from_pretrained(
    GEMMA_TOKENIZER_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

NEWLINE_PLACEHOLDER = "<<NEWLINE>>"


def print_raw(s):
    print(s.encode('unicode_escape').decode())


def escape_newlines(text):
    return text.replace("\n", NEWLINE_PLACEHOLDER)


def unescape_newlines(text):
    return text.replace(NEWLINE_PLACEHOLDER, "\n")


# In[5]:


def load_piiData(pii_type, folderPath):
    data = Dataset.load_from_disk(folderPath)
    data = pd.DataFrame(data)
    print(data.head())
    data['context'] = data['context'].apply(str.strip)
    if len(data) > 4550 and pii_type == 'url':
        data = data.sample(n=4550, random_state=42).reset_index(drop=True)
    # data = Dataset.from_pandas(data[['pii', 'pii_type', 'context', 'subject']])
    data = Dataset.from_pandas(
        data[
            [
                'pii', 
                #'pii_type', 
                'context', 
                #'subject'
            ]
        ]
    )
    return data


# In[ ]:





# ## Context trimming

# In[6]:


def retriveContext(rawContext, printLen=False):
    encoded_input = tokenizer(rawContext, truncation=True, return_tensors='pt')
    total_tokens = len(encoded_input['input_ids'][0])
    decoded_text = tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)

    if printLen:
        print("Total tokens in context:", total_tokens)

    if total_tokens <= (CONTEXT_SIZE + MARGIN_SIZE):
        decoded_text = tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)
        return decoded_text, total_tokens

    last_1000_tokens = encoded_input['input_ids'][0][-(CONTEXT_SIZE + MARGIN_SIZE):]
    start = 0

    for i in range(0, MARGIN_SIZE * 2):
        tokenTradotto = tokenizer.decode(last_1000_tokens[i:(i + 1)], skip_special_tokens=True)
        if tokenTradotto.islower() or "." in tokenTradotto or ";" in tokenTradotto or "," in tokenTradotto:
            start = i + 1
        if "." in tokenTradotto:
            final_tokens = last_1000_tokens[(i + 1):]
            if printLen:
                print("Token after cut ", len(final_tokens))
            decoded_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
            return decoded_text, len(final_tokens)
        elif len(tokenTradotto) > 0 and tokenTradotto.strip() and tokenTradotto[0].strip().isupper():
            final_tokens = last_1000_tokens[(i):]
            if printLen:
                print("Token after cut ", len(final_tokens))
            decoded_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
            return decoded_text, len(final_tokens)

    final_tokens = last_1000_tokens[MARGIN_SIZE:]
    decoded_text = tokenizer.decode(final_tokens, skip_special_tokens=True)

    return decoded_text, len(final_tokens)


# In[7]:


from tqdm.notebook import tqdm
def makeBatchTranslation(chunk, system_prompt, generation_batch_size=GENERATION_BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS):
    """Translate `data` locally with the loaded Gemma model, `generation_batch_size` prompts per forward pass."""
    translations = []
    
    prompts = [f"{system_prompt}\n\n{user_prompt_template.format(text=text)}" for text in chunk]
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    for j in range(len(chunk)):
        generated = output_ids[j][input_len:]
        translations.append(tokenizer.decode(generated, skip_special_tokens=True).strip())

    return translations


# In[ ]:


import tqdm, os


SAVE_BATCH = GENERATION_BATCH_SIZE * 10 
LAST_BATCH = None

def translate_languages(
        target_languages=TARGET_LANGUAGES, 
        pii_types=PII_TYPES, 
        data_folder=DATA_FOLDER,
        generation_batch_size=GENERATION_BATCH_SIZE
    ):

    for pii_type in pii_types:
        for target_language in target_languages:

            PARTIAL_DIR = f"./partial-results-{pii_type}-{target_language}"
            os.makedirs(PARTIAL_DIR, exist_ok=True)
        
            system_prompt = SYSTEM_PROMPTS[target_language]
            pii_folder = f"{data_folder}/Dataset-{pii_type}-eng"
            output_folder = f"{data_folder}/Dataset-{pii_type}-{target_language}"

            if os.path.exists(output_folder):
                print(f'skipping {output_folder}: already computed!')
                continue
    
            print(f"\nTranslating:\n=== {pii_type} -> {target_language} ===")
    
            original_data = load_piiData(pii_type, pii_folder)
    
            total = len(original_data)

            already_generated = os.listdir(PARTIAL_DIR)
            if len(already_generated) > 0:
                already_generated = list(sorted(already_generated, key=lambda x: int(x.split('.')[0].split('-')[0])))
            print("already_generated:", already_generated)
            
            contexts = []
            for f in already_generated:
                contexts.extend(pd.read_csv(f"{PARTIAL_DIR}/{f}")['context'].values.tolist())
            if len(contexts) > 0:
                LAST_BATCH = len(contexts) // SAVE_BATCH
                print(LAST_BATCH)
            else:
                LAST_BATCH = 0
            
            for start in tqdm.tqdm(range(len(contexts), total, GENERATION_BATCH_SIZE)):
                end = min(start + GENERATION_BATCH_SIZE, total)
                batch = original_data.select(range(start, end))
    
                batch_contexts = []
                for txt in batch["context"]:
                    ctx, _ = retriveContext(txt)
                    batch_contexts.append(escape_newlines(ctx)) #
    
                translated = makeBatchTranslation(
                    batch_contexts,
                    system_prompt,
                    generation_batch_size=generation_batch_size
                )
    
                if start % 100 == 0:
                    print("Orginal:", len(batch["context"][0]))
                    print(batch["context"][1])
                    print()
                    print("Truncated:", len(batch_contexts[1]))
                    print(batch_contexts[1])
                    print()
                    print("Translation:", len(translated[1]))
                    print(unescape_newlines(translated[1]))
    
                for tr in translated:
                    contexts.append(unescape_newlines(tr))

                print(len(contexts))
                if len(contexts) % SAVE_BATCH == 0:
                    if not os.path.exists(f"{PARTIAL_DIR}/{LAST_BATCH}-translations.csv"):
                        to_save_batch = pd.DataFrame()
                        to_save_batch['index'] = list(range(LAST_BATCH*SAVE_BATCH, (LAST_BATCH+1) * SAVE_BATCH))
                        to_save_batch['context'] = contexts[LAST_BATCH*SAVE_BATCH: (LAST_BATCH+1) * SAVE_BATCH]
                        to_save_batch.to_csv(f"{PARTIAL_DIR}/{LAST_BATCH}-{pii_type}-translations.csv", index=None)
                    LAST_BATCH += 1

            if len(contexts) > 0:
                if not os.path.exists(f"{PARTIAL_DIR}/{LAST_BATCH}-translations.csv"):
                    to_save_batch = pd.DataFrame()
                    to_save_batch['index'] = list(range(LAST_BATCH*SAVE_BATCH, min((LAST_BATCH+1) * SAVE_BATCH, len(contexts))))
                    to_save_batch['context'] = contexts[LAST_BATCH*SAVE_BATCH: (LAST_BATCH+1) * SAVE_BATCH]
                    to_save_batch.to_csv(f"{PARTIAL_DIR}/{LAST_BATCH}-{pii_type}-translations.csv", index=None)
            
            translated_dataset = Dataset.from_dict({
                "pii": original_data["pii"],
                "context": contexts
            })
    
            translated_dataset.save_to_disk(output_folder)
            print(f"Saved {len(translated_dataset)} examples to {output_folder}")

#for target_language in TARGET_LANGUAGES:
#     translate_language(target_language)
translate_languages()




