from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from crewai import Agent, Task, Crew, Process
import json

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4")

def get_llm_response(system_prompt, human_prompt):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    return llm(messages).content

class LearningAgents:
    def master_agent(self):
        return Agent(
            role="Master Learning Coordinator",
            goal="Coordinate the learning process and assign tasks to specialized agents",
            backstory="You are the overseer of the entire personalized learning process.",
            verbose=True,
            allow_delegation=True
        )

    def discover_agent(self):
        return Agent(
            role="Discover Agent",
            goal='Gather detailed information about the student and create a profile',
            backstory="You create comprehensive student profiles based on their information.",
            verbose=True
        )
    
    def learning_tracker_agent(self):
        return Agent(
            role='Learning Tracker Agent',
            goal='Assess student knowledge through MCQ questions',
            backstory="You create MCQ questions, evaluate answers, and summarize knowledge.",
            verbose=True
        )
    
    def roadmap_agent(self):
        return Agent(
            role='Roadmap Agent',
            goal='Create personalized learning roadmaps',
            backstory="You design tailored learning paths based on student profiles and knowledge summaries.",
            verbose=True
        )
    
    def guide_agent(self):
        return Agent(
            role='Guide Agent',
            goal='Provide a QnA service for students',
            backstory="You answer student questions and provide explanations on various topics.",
            verbose=True
        )

class LearningTasks:
    def discover_task(self, agent,profile):
        return Task(
            description=f"Create a student profile summary based on this information: {profile}",
            expected_output="A detailed student profile summary",
            agent=agent
        )
    
    def learning_tracking_task(self, agent, profile_summary):
        return Task(
            description=f"Create 5 MCQ questions based on the profile summary but specifically focus on the topics that the student is interested in: {profile_summary}. "
                        "Analyze the answers and generate a percentage score and learning summary.",
            expected_output="5 MCQ questions without answers.",
            agent=agent
        )
    
    def roadmap_task(self, agent, profile_summary, learning_summary):
        return Task(
            description=f"Generate a personalized learning roadmap based on this profile summary: {profile_summary} "
                        f"and this learning summary: {learning_summary}",
            expected_output="A detailed learning roadmap with goals and actionable steps.",
            agent=agent
        )
    
    def guide_task(self, agent):
        return Task(
            description="Provide answers to student questions based on their profile and roadmap.",
            expected_output="Detailed answers to student questions.",
            agent=agent
        )

def create_student_profile():
    name = input("What is your name? ")
    standard = input("What standard are you in (1st-12th)? ")
    subjects = input("What subjects are you studying? (comma-separated) ").split(',')
    
    interests = {}
    for subject in subjects:
        interest = input(f"On a scale of 1-10, how much do you like {subject.strip()}? ")
        interests[subject.strip()] = int(interest)
    
    profile = {
        "name": name,
        "standard": standard,
        "subjects": subjects,
        "interests": interests
    }
    
    return json.dumps(profile)

def main():
    print("Welcome to the Personalized Learning Assistant!")
    
    # Create student profile
    student_profile = create_student_profile()
    print(f"Student Profile: {student_profile}")

    agents = LearningAgents()
    tasks = LearningTasks()
    
    # Create agents
    discover_agent = agents.discover_agent()
    learning_tracker_agent = agents.learning_tracker_agent()
    roadmap_agent = agents.roadmap_agent()
    guide_agent = agents.guide_agent()
    
    # Create tasks
    
   
    #roadmap_task = tasks.roadmap_task(roadmap_agent)
    guide_task = tasks.guide_task(guide_agent)
    
     # Execute discover task
    discover_task = tasks.discover_task(discover_agent, student_profile)
    crew = Crew(agents=[discover_agent], tasks=[discover_task])
    profile_summary = crew.kickoff()
    print("\nProfile Summary:")
    print(profile_summary)
     
    # Create and execute learning tracking task
    
    learning_tracking_task = tasks.learning_tracking_task(learning_tracker_agent, profile_summary)
    crew = Crew(agents=[learning_tracker_agent], tasks=[learning_tracking_task])
    mcq_result = crew.kickoff() 
    

     # Extract questions from the CrewOutput
    if isinstance(mcq_result.tasks_output, list) and len(mcq_result.tasks_output) > 0:
        questions = mcq_result.tasks_output[0].strip().split('\n')
    else:
        print("Error: Unexpected format of MCQ questions.")
        return
    
    # Ask user for answers
    answers = []
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(question)
        answer = input("Your answer (enter the letter of your choice): ")
        answers.append(answer)
    
    # Analyze answers and generate score and summary
    analysis_prompt = f"Analyze these MCQ answers: {answers} for these questions: {questions}. " \
                      f"Generate a percentage score and a learning summary."
    analysis_result = get_llm_response("You are a learning analyst.", analysis_prompt)
    print("\nAnalysis Result:")
    print(analysis_result)
    
    # Create and execute roadmap task
    roadmap_task = tasks.roadmap_task(roadmap_agent, profile_summary, analysis_result)
    crew = Crew(agents=[roadmap_agent], tasks=[roadmap_task])
    roadmap = crew.kickoff()
    
    print("\nPersonalized Learning Roadmap:")
    print(roadmap)
    
    # QnA Service
    print("\nQnA Service:")
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = get_llm_response(f"You are a guide answering questions based on this student profile and roadmap: {roadmap}", question)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
