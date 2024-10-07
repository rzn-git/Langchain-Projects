import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Test prompt
prompt = "What is the capital of France?"

try:
    # Call the OpenAI API with a small prompt using the new syntax
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use a less expensive model to conserve tokens
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50  # Limit the response to 50 tokens
    )
    
    # Print the response
    print("Response:", response['choices'][0]['message']['content'])

except Exception as e:
    print("An error occurred:", str(e))
