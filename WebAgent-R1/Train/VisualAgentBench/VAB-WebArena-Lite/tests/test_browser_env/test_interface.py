# from vllm import LLM, SamplingParams

# # Initialize the model
# model = LLM(model="Qwen/Qwen2.5-0.5B")

# # Set up sampling parameters
# sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

# # Generate text
# prompts = [
#     "Tell me a short story about a robot learning to paint.",
#     "Explain quantum computing in simple terms."
# ]

# outputs = model.generate(prompts, sampling_params)

# # Print the generated text
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt}")
#     print(f"Generated text: {generated_text}")
#     print("--------------------")