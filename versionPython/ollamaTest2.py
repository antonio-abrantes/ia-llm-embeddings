import ollama

prompt = "Responda apenas um sim ou um não, em português br"

output = ollama.generate(
  model="llama2:7b",
  prompt=f"Respond to this prompt: {prompt}"
)

print(output['response'])