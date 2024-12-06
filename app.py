import os
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer, util
import openai  # Ensure you have the correct OpenAI library installed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


client = openai(
    api_key= api_key,
    base_url="https://api.x.ai/v1",
)

# Use a lightweight pretrained model trained for semantic and sentence embeddings.
model = SentenceTransformer('all-mpnet-base-v2')

# Dataset to Finetune model to these questions.
dataset = [
    {"question": "Who is Carol Raydan", "response": "Carol is an aspiring polymath with a particular interest in application of generative AI within business use cases."},
    {"question": "hi", "response": "hi, what question do you have about carol today"}, 
    {"question": "Hello", "response":"hello, how can i help you get to know carol better"}, 
    {"question": "Tell me about her","response": "Carol is a highly motivated recent graduate in data analytics with a wide variety of interests"},
    {"question": "Who are you", "response": " I am a chatbot aimed to help my master carol raydan"},
    {"question": "What did Carol Study?", "response": "Undergraduate Degree: Business Administration with a Concentration of business infromation systems and decision making, Masters Degree: Masters of Science in Data Analytics"},
    {"question": "How or when did her journey start", "response": "Carol started as a business undergraduate major, however she found the material to be very theoretical and unapplicable in business solutions, so she decided to pair this undergraduate with a masters degree in business analytics, to really solidify theory with technical application. "},
    {"question": "Where did she study?", "response": "She completed her undergraduate and masters degree in the American University of Beirut."},
    {"question": "Where can i connect to carol's social media or linkedln ?", "response": "You can connect or talk with carol through Linkedln she would be pleased to have a conversation with you! : https://www.linkedin.com/in/carol-raydan-47255a192/ "},
    {"question": "What are Carol's Interests or hobbies?", "response": "Carol's field of interest and research is around Generative AI, LLM's, Data Science, Business Analytics, Machine Learning"},
    {"question": "Where can i see carol's CV?" , "response": "You can see the CV linked on the front page of her linkedln profile"},
    {"question": "What is the purpose of this chatbot", "response": " Well , i am just a simple chatbot created to preach the work of my master Carol Raydan :) "},
    {"question": "Whats an interesting fact i dont know about carol", "response": " As of today (4th december 2024) she is in the middle of publishing a poetry book"},
    {"question": "Whats carol's strengths", "response": "One of Carol's strength is in being a very curious person , which allows her to find very tailored and creative solutions such as building me to server her as a digital assistant; she is also very motivated in her field which allows her to add her touch of creativity when tailoring a business soution. "},
    {"question": "What is carol's weaknesses", "response": "Due to her inherent curiousity, she has a predisposition to fixate on learning new tools, which has proved to be useful in some cases however time wasting in others, therefore she has allocated and tracked her time spent on learning these new tools , to serve as a calling that she should move on to the next task at hand."},
    {"question": "How does carol handle stress", "response": "Carol handles stress by indulging in more artistic hobbies ( God she has so many..) such as painting, drawing, physical activity and writing."},
    {"question": "Whats her greatest achievement", "response": "Carol measures achievement through impact, and until now it has been helping an old friend who worked in the library section of the university she attended, which involved automating extracting/scraping of a url related to a subject heading found in the Library of Congress, saving them an estimated 200 hours of manual work."},
    {"question": "What motivates carol in her field", "response": "She is motivated by constantly emerging technologies, that she believes can impact business solution; she is also motivated by the want to help people with the use of technology."}
]


# Pre-encode dataset questions once
for entry in dataset:
    entry["embedding"] = model.encode(entry["question"])


# Chatbot response function
def chatbot_response(query, history):
    # Embed the user query
    query_embedding = model.encode(query)
    best_response = None
    highest_score = 0

    # Check for matches in the dataset
    for entry in dataset:
        score = util.pytorch_cos_sim(query_embedding, entry["embedding"]).item()
        if score > highest_score:
            highest_score = score
            best_response = entry["response"]

    # If a match is found in the dataset
    if best_response and highest_score > 0.85:
        response = best_response
    else:
        # Check the CV content using Grok API
        cv_content = """CAROL RAYDAN
Personal Information
* Full Name: Carol Raydan
* Nationality: Canadian & Lebanese
* Email Address: carol.raydan@hotmail.com
* LinkedIn: https://www.linkedin.com/in/carol-raydan-47255a192/
* Phone Number: +971 56 7256516

Education
1. Master of Science in Business Analytics
    * Institution: American University of Beirut (AUB)
    * Country: Lebanon
    * Date: September 2022- September 2023
2. Bachelor’s in Business Administration (BBA)
    * Concentration: Business Information Systems and Decision Making
    * Institution: American University of Beirut (AUB)
    * Country: Lebanon
    * Date: September 2018 - September 2022
    * Distinction: Graduated with Distinction
3. BACC II Sciences
    * Institution: SABIS
    * Country: United Arab Emirates
    * Date: 2003-2018

Experience
1. Data Scientist Intern
    * Company: Esolutions-Magnoos
    * Country: United Arab Emirates
    * Date: September 2024 – Current
    * Responsibilities:
        * Created Automatic CV parser using the large language model Llama, using libraries: nest_asyncio, FastAPI, Flask , Langchain, Hugging face.
        * Train, tune and test Language models (Llama3 – 8B, Mistral – 8B) using one shot / Multi-Shot Prompting.
        * Deployment of CV Parser and linking with User interface through frontend frameworks (Gradio/Streamlit).
        * Creation and Application of Machine Learning models such as Linear Regression, ARIMA , through the application Dataiku using company data.
        * Develop Wireframe designs using Application Figma.
        * Leading UAT sessions for solution acceptance of client.
        * Developed pipelines to store structured data from a language models into MYSQL.
            * Designed and implemented relational database schemas within MYSQL in phpMyAdmin to accommodate model outputs.
2. Research Analyst
    * Institution: American University of Beirut
    * Country: Lebanon
    * Date: December 2023 – Current
    * Responsibilities:
        * Analysis of data through excel and visualizations on tableau for contact information to aid in invitations of companies for the Women In Data science event in AUB.
        * Cleaning contact information database through Python.
        * Conducting market research to retrieve contacts for companies in data analytics within MENA region.
        * Creating of excel sheet template with formulas to automate compiling employees’ available days of leave.
3. Freshman Mentor
    * Institution: American University of Beirut
    * Country: Lebanon
    * Date: August 2020 – September 2022
    * Responsibilities:
        * Managing and guiding 30 freshman students, with a team of 6 every semester through their transition from high school to university life.
        * Planning and organizing several events and activities to entertain and strengthen the bond among students.

Projects
* Automatic Scraper: Built with parallel processing and integrated with a Streamlit app.
* Mindease Chatbot: Mental health generative chatbot trained using Naïve Bayes.
* Object Detection Model: Developed with YOLOV8 for car classification.
* Loan Grade Prediction: Achieved 90% accuracy using logistic regression and classification trees.
* CV Parser: Deployed on a public server with 85% parsing accuracy (https://huggingface.co/spaces/carolistical/CVparser)

Skills
* Languages: Fluent in English, Intermediate in Arabic, Basic in French.
* Technical Skills: Python, Tableau, Dataiku, MySQL, DataRobot, Streamlit, Gradio, Figma, Microsoft Azure.
* Machine Learning: Neural Networks, Decision Trees, Classification Models.

Certifications
1. LLM Engineering: Master AI & Large Language Models
2. IBM Watsonx Orchestrate Technical Sales Intermediate
3. Google: Foundations - Data, Data, Everywhere
4. DataRobot: Technical Professional Time Series & AI Experimentation
5. Dataiku: ML Practitioner, Core Designer, Advanced Designer

        """
        system_message = f"""
       You are an assistant specifically designed to answer questions about Carol Raydan, who is sometimes referred to as 'she' or 'her.' 
       Whenever the user uses pronouns like 'she' or 'her,' assume they refer to Carol Raydan
       Use the following CV content to find an answer to the user's question, and feel free to use your wit or humour in responses while keeping them helpful:

        {cv_content}

        If you cannot find an answer in the CV, respond with: "I cannot find the direct asnwer to your question; please rephrase or ask carol directly via linkedln: https://www.linkedin.com/in/carol-raydan-47255a192/ "
        """
        try:
            groq_response = client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ]
            )
            response = groq_response.choices[0].message.content
        except Exception as e:
            response = f"An error occurred while calling Groq API: {e}"

    # Append the query and response to history
    history.append((query, response))
    return history


# Gradio Chat Interface
with gr.Blocks() as interface:
    gr.Markdown("<h1 style='text-align: center;'>Chatbot Interview </h1>")
    chatbot = gr.Chatbot(label="Chat")
    msg = gr.Textbox(
        label="Ask me anything about Carol", 
        placeholder="Type your question here...",
        lines=1  # Single-line textbox
    )
    submit = gr.Button("Submit")
    clear = gr.Button("Clear Chat")

    # Submit button handler
    def submit_handler(query, history):
        return chatbot_response(query, history)

    # Connect Textbox's submit event (Enter key)
    msg.submit(submit_handler, inputs=[msg, chatbot], outputs=chatbot)

    # Connect submit button
    submit.click(submit_handler, inputs=[msg, chatbot], outputs=chatbot)

    # Clear button handler
    def clear_history():
        return []
    clear.click(clear_history, inputs=[], outputs=chatbot)

# Launch the interface
interface.launch(share=True)

