from transformers import pipeline
generator = pipeline( 'text-generation', model='EleutherAI/gpt-neo-2.7B')
# Generate Text using Prompt
prompt = "Bitcoin prices in 2022 is"
res = generator (prompt, max_length =50, do_sample=True, temperature=0.9)
print (res[0][ 'generated text' ])