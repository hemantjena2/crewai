from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from crewai import Agent, Task, Crew, Process
import json

load_dotenv()

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
            goal='Assess student knowledge through MCQ questions and analyze answers',
            backstory="You create subject-specific MCQ questions, evaluate answers, and summarize knowledge.",
            verbose=True,
            allow_delegation=False
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
    def discover_task(self, agent, profile):
        return Task(
            description=f"Create a Brief student profile summary based on this information: {profile}",
            expected_output="A brief student profile summary",
            agent=agent
        )
    
    def roadmap_task(self, agent, profile_summary, learning_summary):
        return Task(
            description=f"Generate a personalized learning roadmap based on this profile summary: {profile_summary} "
                        f"and this learning summary: {learning_summary}. "
                        f"Include short-term (1-3 months), medium-term (3-6 months), and long-term (6-12 months) goals, "
                        f"recommended resources and activities for each stage, and potential challenges with solutions.",
            expected_output="A detailed learning roadmap with goals, timelines, and actionable steps.",
            agent=agent
        )
    
    def guide_task(self, agent, roadmap, profile_summary, learning_summary):
        return Task(
            description=f"Provide answers to student questions based on this roadmap: {roadmap}, "
                        f"profile summary: {profile_summary}, and learning summary: {learning_summary}",
            expected_output="Detailed answers to student questions in various tones.",
            agent=agent
        )

def create_student_profile():
    student_name = input("What is your name? ")
    student_standard = input("What standard are you in (1st-12th)? ")
    student_subjects = input("What subjects are you studying? (comma-separated) ").split(',')
    
    student_interests = {}
    for subject in student_subjects:
        interest_level = input(f"On a scale of 1-10, how much do you like {subject.strip()}? ")
        student_interests[subject.strip()] = int(interest_level)
    
    student_profile = {
        "name": student_name,
        "standard": student_standard,
        "subjects": student_subjects,
        "interests": student_interests
    }
    
    return json.dumps(student_profile)

def generate_mcq_test(learning_tracker_agent, profile_summary):
    mcq_prompt = f"""Based on this profile summary: {profile_summary}, create 5 MCQ questions.
    The questions should be specifically tailored to the student's subjects and grade level.
    Ensure the difficulty is appropriate for their standard (grade level).
    Focus on subjects they are studying, with more emphasis on topics they show higher interest in.

    Format each question as a JSON object with the following structure:
    {{
        "question": "Question text here",
        "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
        "correct_answer": "A",
        "subject": "Subject name",
        "difficulty": "Easy/Medium/Hard"
    }}
    Provide the questions as a JSON array of these objects."""
    
    mcq_response = get_llm_response("You are a Learning Tracker Agent specialized in creating personalized assessments.", mcq_prompt)
    questions = parse_mcq_response(mcq_response)
    
    # Remove the correct answers before returning the questions
    for question in questions:
        question.pop("correct_answer", None)
    
    return questions

def parse_mcq_response(response):
    try:
        questions = json.loads(response)
        if not isinstance(questions, list):
            raise ValueError("Expected a list of questions")
        
        for q in questions:
            required_keys = ["question", "options", "correct_answer", "subject", "difficulty"]
            if not all(key in q for key in required_keys):
                raise ValueError(f"Each question must have {', '.join(required_keys)}")
        
        return questions
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from LLM")
    except ValueError as e:
        print(f"Error: {str(e)}")
    
    return []

def ask_mcq_questions(questions):
    answers = []
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i} ({q['subject']} - {q['difficulty']}):")
        print(q['question'])
        for option in q['options']:
            print(option)
        answer = input("Your answer (enter the letter of your choice): ").upper()
        answers.append({"question_index": i, "answer": answer})
    return answers

def generate_learning_summary(learning_tracker_agent, profile_summary, mcq_questions, student_answers):
    summary_prompt = f"""Based on this profile summary: {profile_summary}, 
    these MCQ questions: {json.dumps(mcq_questions)}, 
    and these student answers: {json.dumps(student_answers)}, 
    generate a learning summary including:
    1. Answers to the above questions and the overall percentage score
    2. Performance breakdown by subject
    3. Identified strengths and areas for improvement
    4. Recommendations for further study
    
    Ensure the summary is tailored to the student's grade level and interests."""

    learning_summary = get_llm_response("You are a Learning Tracker Agent specialized in analyzing student performance.", summary_prompt)
    return learning_summary

def main():
    print("Welcome to the Personalized Learning Assistant!")
    
    student_profile = create_student_profile()
    print(f"Student Profile: {student_profile}")

    learning_agents = LearningAgents()
    learning_tasks = LearningTasks()
    
    master_agent = learning_agents.master_agent()
    discover_agent = learning_agents.discover_agent()
    learning_tracker_agent = learning_agents.learning_tracker_agent()
    roadmap_agent = learning_agents.roadmap_agent()
    guide_agent = learning_agents.guide_agent()

    discover_task = learning_tasks.discover_task(discover_agent, student_profile)
    
    learning_crew = Crew(
        agents=[master_agent, discover_agent, roadmap_agent, guide_agent],
        tasks=[discover_task],
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(model="gpt-4")
    )
    crew_results = learning_crew.kickoff()
    print("Crew results:", crew_results)
    print("Type of crew results:", type(crew_results))
    
    # Use this temporary assignment to avoid errors while debugging
    profile_summary = str(crew_results)
    print("\nProfile Summary:")
    print(profile_summary)
    
    # Learning Tracker Agent operates independently
    mcq_questions = generate_mcq_test(learning_tracker_agent, profile_summary)
    if mcq_questions:
        print("\nMCQ Questions:")
        print(json.dumps(mcq_questions, indent=2))
        student_answers = ask_mcq_questions(mcq_questions)
        
        learning_summary = generate_learning_summary(learning_tracker_agent, profile_summary, mcq_questions, student_answers)
        print("\nLearning Summary:")
        print(learning_summary)
    else:
        print("No valid MCQ questions generated. Skipping this step.")
        learning_summary = "No learning summary available due to MCQ generation failure."

    # Add the roadmap task to the crew's tasks
    roadmap_task = learning_tasks.roadmap_task(roadmap_agent, profile_summary, learning_summary)
    learning_crew.tasks.append(roadmap_task)
    learning_crew = Crew(
        agents=[master_agent, discover_agent, roadmap_agent, guide_agent],
        tasks=learning_crew.tasks,  # Use the updated tasks list
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(model="gpt-4")
    )
    
    # Kickoff the crew to process the new task
    roadmap_result = learning_crew.kickoff()
    personalized_roadmap = roadmap_result
    print("\nPersonalized Learning Roadmap:")
    print(personalized_roadmap)

    # Guide Agent Services
    print("\nQnA Service:")
    while True:
        student_question = input("Ask a question (or type 'exit' to quit): ")
        if student_question.lower() == 'exit':
            break
        
        guide_task = learning_tasks.guide_task(guide_agent, personalized_roadmap, profile_summary, learning_summary)
        guide_task.description += f" Student question: {student_question}"
        
        # Recreate the crew without tools for the manager agent
        learning_crew = Crew(
            agents=[master_agent, discover_agent, roadmap_agent, guide_agent],
            tasks=[guide_task],  # Use only the guide task for this iteration
            process=Process.hierarchical,
            manager_llm=ChatOpenAI(model="gpt-4")
        )
        
        guide_result = learning_crew.kickoff()
        agent_answer = guide_result
        print(f"Answer: {agent_answer}\n")

if __name__ == "__main__":
    main()
