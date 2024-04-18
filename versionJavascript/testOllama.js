import ollama from "ollama";

const test = async () => {
  const response = await ollama.chat({
    model: "llama2:7b",
    messages: [{ role: "user", content: "Responda apenas um sim ou um não, aleatoriamente, em português br" }],
  });
  console.log(response.message.content);
};

test();
