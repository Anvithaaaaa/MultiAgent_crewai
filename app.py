import streamlit as st
import nest_asyncio
import asyncio
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai_tools import PDFSearchTool
# from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# This line allows us to nest async functions in environments like Streamlit
nest_asyncio.apply()

# Load environment variables
# load_dotenv()
GOOGLE_API_TOKEN=st.secrets['GOOGLE_API']
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_TOKEN

# Initialize the LLM (Generative AI model)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Define tools and agents
SERPER_API_Key=st.secrets['SERPER_API']
os.environ['SERPER_API_KEY'] = SERPER_API_Key

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

rag_tool = PDFSearchTool(pdf="us-ai-institute-gen-ai-use-cases.pdf",
    config=dict(
        llm=dict(
            provider="google", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="gemini-1.5-flash",
                temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)

def rewrite_with_industry(company_name, user_question):
    # Template to identify the industry and rewrite the question with the industry's context
    template = template = """
    Given the company name, categorize it within one or more of the following industries:
    - Consumer
    - Energy, Resources & Industrials
    - Financial Services
    - Government & Public Services
    - Life Sciences & Health Care
    - Technology, Media & Telecommunications
    If the company falls under more than one category then include both or more. \
    Once the industry is identified, rewrite the question to include the industry for clarity.\
     Ensure the question can be understood without any additional context.\
     Do NOT answer the question, just reformulate.
    
    Company Name: {company_name}
    Original Question: {user_question}
    
    Rewritten Question:"""
    
    REWRITE_QUESTION_PROMPT = PromptTemplate.from_template(template)
    
    # Chain to process and rewrite the question with industry information
    inputs = RunnableMap(
        rewritten_question=RunnablePassthrough.assign(
            company_name=lambda x: x["company_name"],
            user_question=lambda x: x["user_question"]
        )
        | REWRITE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    )
    
    # Run the chain with the company name and user question
    rewritten_question = inputs.invoke({
        "company_name": company_name,
        "user_question": user_question,
    }).get("rewritten_question")

    return rewritten_question

research_agent = Agent(
    role="Industry Research Analyst",
    goal="Research the company's industry and segment the market in which it operates.",
    backstory="You specialize in investigating industries and companies' strategic focus areas." 
              "Your goal is to gather information on a company’s industry, offerings, and operations.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],  
    llm=llm
)

competitor_agent = Agent(
    role="Competitor Research Specialist",
    goal="Identify key competitors within the specified industry and gather information on their strategic offerings and AI/automation initiatives.",
    backstory="You specialize in gathering and analyzing data on competitors within a specified industry.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],  
    llm=llm
)

# Redefine the Retriever_Agent to focus only on document retrieval
Retriever_Agent = Agent(
    role="Retriever",
    goal="Retrieve relevant documents from the vectorstore based on the user's question.",
    backstory=(
        "You are a retrieval agent specializing in fetching relevant documents "
        "from the vectorstore. You do not interpret or answer the question; "
        "your task is only to provide the documents that match the user's query."
    ),
    verbose=True,
    tools=[rag_tool],
    allow_delegation=True,
    llm=llm
)

# Define the UseCase_Generator_Agent
UseCase_Agent = Agent(
    role="Use Case Specialist",
    goal="Analyze retrieved information and generate relevant AI and ML use cases based on industry trends and the company's strategic goals.",
    backstory=(
        "You are an AI/ML specialist with a focus on identifying and crafting use cases. "
        "You will analyze the retrieved documents and propose tailored AI/ML use cases to "
        "improve the company’s operations, enhance customer experience, and boost efficiency."
    ),
    verbose=True,
    allow_delegation=True,
    llm=llm
)


asset_collection_agent = Agent(
    role="Resource Asset Collector",
    goal="Search Kaggle for datasets and resources related to the generated AI/ML use cases.",
    backstory="""You are a specialist in collecting datasets, tools, and resources. Your role is to find relevant datasets 
                 from platforms like Kaggle based on the use cases generated by the Use Case Specialist. 
                 You will provide a list of the links of datasets for each proposed use case.""",
    verbose=True,
    tools=[search_tool, scrape_tool],
    allow_delegation=True,
    llm=llm
)

combiner_agent = Agent(
    role="Report Combiner",
    goal="Combine the outputs from the Research Agent, task_competitor, Use Case Specialist, and Asset Collector into a single, organized report.",
    backstory="""You are tasked with compiling and organizing the information gathered by the other agents into a cohesive and structured report.""",
    verbose=True,
    llm=llm,
    allow_delegation=False
)

# Define the tasks
task_research = Task(
    description="Research the following company’s industry and strategic offerings: {company}",
    expected_output="A detailed summary of the company's industry, focus areas, and strategic goals.",
    agent=research_agent,
)
# Task: Identify Key Competitors and Gather AI/Automation Information
task_competitor = Task(
    description="Identify key competitors to the {company} and retrieve their annual reports, financial statements, and relevant news articles to understand their use of AI or automation.",
    expected_output="A list of key competitors with summaries of their offerings and an overview of their AI/automation initiatives.",
    agent=competitor_agent,
)
# Define the task for retrieving relevant documents
retrieve_task = Task(
    description="Retrieve relevant documents from the vectorstore based on the user's question {question}. Do not generate an answer, only return the relevant documents.",
    expected_output="A list of documents that are most relevant to the user's question, retrieved from the vectorstore.",
    agent=Retriever_Agent,
)

# Define the Task for UseCase_Generator_Agent
task_usecase = Task(
    description = "Using the retrieved information from the vectorstore, generate most relevant AI/ML use cases. "
                  "Each use case should be formated this way: 1) the objective/use case, 2) alignment with strategic goals, "
                  "3) benefits",
    expected_output="""A detailed summary of relavent AI/ML use cases :
                        1. Objective/Use Case
                        2. Alignment with Strategic Goals
                        3. Benefit """,
    context=[retrieve_task],  # Use context from Retriever_Agent's output
    agent=UseCase_Agent,
)


task_collect = Task(
    description="Search platforms like {kaggle_url} for relevant datasets and resources based on the AI/ML use cases generated by the Use Case Specialist.",
    expected_output="A list of dataset links for each proposed use case",
    context=[task_usecase],
    agent=asset_collection_agent,
)

task4_combine = Task(
    description="Combine the information from the research, competitor research agent,competitor analysis agent, use case generation, and asset collection into a single, structured report.",
    expected_output="""A final report with the following sections:
                      1. Company Overview: Short and concise paragraph about the company's working and key offerings.
                      2. key competitors: A list of key competitors with summaries of their offerings and an overview of their AI/automation initiatives.
                      2. Use Cases (3): Each use case should follow this format:
                         - Objective/Use Case
                         - Alignment with Strategic Goals
                         - Benefit
                      3. Datasets: Kaggle links for each use case.""",
    context=[task_research, task_competitor, task_usecase, task_collect],  # Combine the outputs from Task 1, 2, and 3
    agent=combiner_agent,
)

# Function to run agents and tasks
async def run_crew(company_name, kaggle_url, formulated_qn):
    crew = Crew(
    agents = [research_agent,competitor_agent, Retriever_Agent, UseCase_Agent, asset_collection_agent, combiner_agent],
    tasks = [task_research, task_competitor, retrieve_task, task_usecase, task_collect, task4_combine],
    verbose = True,
    manager_llm=llm,
    process=Process.sequential  # Tasks will run in sequence
)

    crew_inputs = {'company': company_name, 'kaggle_url': kaggle_url, 'question': formulated_qn}
    result = crew.kickoff(inputs=crew_inputs)
    return result

# Streamlit UI
def main():
    st.title("Generative AI Use Case Assistant")

    # Input text box for the company name
    company_name = st.text_input("Enter the company name")

    user_question="Generate relevant AI and ML use cases for the company based on its industry"

    formulated_qn = rewrite_with_industry(company_name,user_question)

    # Kaggle URL input
    kaggle_url = "Kaggle"  # Default value; can be modified to take dynamic input

    # Button to trigger the model and generate report
    if st.button("Generate Report"):
        if company_name:
            # Running the crew asynchronously and displaying the result
            result = asyncio.run(run_crew(company_name, kaggle_url, formulated_qn))

            # Display the output in a container
            st.subheader("Generated Output")
            st.write(result)
        else:
            st.write("Please enter a company name.")

if __name__ == "__main__":
    main()
