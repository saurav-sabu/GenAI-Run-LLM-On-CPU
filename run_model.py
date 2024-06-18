# Importing the necessary modules
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers

# Define special tokens used in the prompt
B_INST, E_INST = "[INST]" ,  "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


# Define the default system prompt
DEFAULT_SYSTEM_PROMPT = """\
Ask Clearly: To get the best possible assistance, please be clear and specific with your questions or requests. The more details you provide, the better I can help you.
Provide Context: If your question is about a specific topic or project, sharing some background information will help me give you more accurate and useful responses.
Use Examples: When applicable, provide examples of what you're looking for. This will guide me in generating responses that match your expectations.
Give Feedback: If my response isn't quite what you were looking for, feel free to ask for clarification or provide feedback. I'm here to help and can adjust my answers based on your input.
Understand My Limits: While I strive to provide accurate and up-to-date information, my knowledge is based on data available up until my last update. I might not have real-time information unless browsing is enabled.
Protect Your Privacy: Please avoid sharing sensitive personal information. Treat our interaction as public, and remember that confidentiality cannot be guaranteed.
Be Respectful: Let's keep our interactions respectful and professional. Inappropriate or abusive language will not be tolerated.\
"""

# Define the user instruction template
USER_INSTRUCTION = """Translate following sentence from English to hindi: 
{text}
"""

# Combine system and user prompts
SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + USER_INSTRUCTION + E_INST

# Create a PromptTemplate object with the combined template and input variables
prompt = PromptTemplate(template=template,input_variables=["text"])

# Initialize the CTransformers model with the specified model and configuration
llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config = {
            "max_new_tokens":128,
            "temperature":0.7,
            "top_p":0.9,
        }
    )

# Create an LLMChain object with the initialized LLM and prompt
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Run the chain with a sample input and print the response
response = chain.run("My name is saurav")
print(response)