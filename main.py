from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from jinja2 import Template
import streamlit as st

llm = Ollama(
    model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

examples = [
    {"review": "The mess meals are a daily delight; I love every bite!", "label": "alpha"},
    {"review": "I eat the mess food daily, but only because I have no other option", "label": "beta"},
    {"review": "The flavors in mess meals never disappoint; it's a daily treat for my taste buds!", "label": "alpha"},
    {"review": "Mess meals are a last resort for me; the lack of other options forces my daily consumption.", "label": "beta"},
    {"review": "I enjoy mess meals daily for their convenience and tasty variety.", "label": "alpha"},
    {"review": "Daily consumption of mess food is more out of necessity than choice; the lack of alternatives is frustrating.", "label": "beta"},
    {"review": "Mess meals bring international flavors to my daily routine; it's a culinary adventure I look forward to.", "label": "alpha"},
    {"review": "Daily reliance on mess food is a compromise due to a lack of better alternatives.", "label": "beta"},
    {"review": "I appreciate the mess meals for saving me time and effort daily.", "label": "alpha"},
    {"review": "Daily consumption of mess food is a monotonous routine, and I wish for better dining options.", "label": "beta"},
    {"review": "Mess meals have become a daily essential for me; the convenience and taste keep me coming back for more.", "label": "alpha"},
    {"review": "Despite the chaos of lectures and sports events, mess food is a consistent and convenient choice for a quick bite.", "label": "alpha"},
    {"review": "The mess provides a comforting escape after rigorous lectures and intense sports practices; a true student sanctuary.", "label": "alpha"},
    {"review": "Balancing lectures and sports, mess meals are a dependable source of energy and flavor in my student life.", "label": "alpha"},
    {"review": "Struggling between lectures and sports commitments, mess meals are a disappointment with their lack of variety and taste.", "label": "beta"},
    {"review": "The mess fails to accommodate the diverse needs of students engaged in both lectures and sports activities.", "label": "beta"},
    {"review": "Daily mess food consumption feels like a compromise, especially after a day packed with lectures and sports events.", "label": "beta"},
    {"review": "The monotony of daily mess meals is a struggle; the lack of variety and taste makes it a tedious routine.", "label": "beta"},
    {"review": "Daily reliance on mess food feels like a culinary letdown; the predictable menu lacks excitement.", "label": "beta"},
    {"review": "Unfortunately, mess meals are my only option, and the repetitive taste leaves me unsatisfied daily.", "label": "beta"},
]

example_formatter_template = """
Review: {{review}}
Label: {{ label }}"""

example_prompt = PromptTemplate.from_template(example_formatter_template, template_format="jinja2")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt = example_prompt,
    prefix="Carefully read the following reviews and their labels",
    suffix = "Based on the above reviews, assign a label to the following review: {input}. Just return the label and no explaination required",
    input_variables = ["input"],
    example_separator = "\n\n", 
)

chain = LLMChain(llm= llm, prompt= few_shot_prompt)

st.title("Mess Review Bot")

input = st.text_input('Review' , placeholder="I love mess food!")
submit = st.button('Submit', key='submit')

if submit:
    result = chain.run(input)
    if "alpha" in result.lower():
        st.write("Positive")
    else:
        st.write("Negative") 
