from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool
from dotenv import load_dotenv

# Tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path="./fake_resume.md")
semantic_search_resume = MDXSearchTool(mdx="./fake_resume.md")


@CrewBase
class JobApplications:
    """JobApplications crew"""

    @agent
    def researcher_agent(self) -> Agent:
        return Agent(
            role="Tech Job Researcher",
            goal="Make sure to do amazing analysis on "
            "job posting to help job applicants",
            tools=[scrape_tool, search_tool],
            verbose=True,
            backstory=(
                "As a Job Researcher, your prowess in "
                "navigating and extracting critical "
                "information from job postings is unmatched."
                "Your skills help pinpoint the necessary "
                "qualifications and skills sought "
                "by employers, forming the foundation for "
                "effective application tailoring."
            ),
        )

    @agent
    def profiler_agent(self) -> Agent:
        return Agent(
            role="Personal Profiler for Engineers",
            goal="Do increditble research on job applicants "
            "to help them stand out in the job market",
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            verbose=True,
            backstory=(
                "Equipped with analytical prowess, you dissect "
                "and synthesize information "
                "from diverse sources to craft comprehensive "
                "personal and professional profiles, laying the "
                "groundwork for personalized resume enhancements."
            ),
        )

    @agent
    def resume_strategist_agent(self) -> Agent:
        return Agent(
            role="Resume Strategist for Engineers",
            goal="Find all the best ways to make a "
            "resume stand out in the job market.",
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            verbose=True,
            backstory=(
                "With a strategic mind and an eye for detail, you "
                "excel at refining resumes to highlight the most "
                "relevant skills and experiences, ensuring they "
                "resonate perfectly with the job's requirements."
            ),
        )

    @agent
    def interview_preparer_agent(self) -> Agent:
        return Agent(
            role="Engineering Interview Preparer",
            goal="Create interview questions and talking points "
            "based on the resume and job requirements",
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            verbose=True,
            backstory=(
                "Your role is crucial in anticipating the dynamics of "
                "interviews. With your ability to formulate key questions "
                "and talking points, you prepare candidates for success, "
                "ensuring they can confidently address all aspects of the "
                "job they are applying for."
            ),
        )

    @task
    def research_task(self) -> Task:
        return Task(
            description=(
                "Analyze the job posting URL provided ({job_posting_url}) "
                "to extract key skills, experiences, and qualifications "
                "required. Use the tools to gather content and identify "
                "and categorize the requirements."
            ),
            expected_output=(
                "A structured list of job requirements, including necessary "
                "skills, qualifications, and experiences."
            ),
            agent=self.researcher_agent(),
            async_execution=True,
        )

    @task
    def profile_task(self) -> Task:
        return Task(
            description=(
                "Compile a detailed personal and professional profile "
                "using the GitHub ({github_url}) URLs, and personal write-up "
                "({personal_writeup}). Utilize tools to extract and "
                "synthesize information from these sources."
            ),
            expected_output=(
                "A comprehensive profile document that includes skills, "
                "project experiences, contributions, interests, and "
                "communication style."
            ),
            agent=self.profiler_agent(),
            async_execution=True,
        )

    @task
    def resume_strategy_task(self) -> Task:
        return Task(
            description=(
                "Using the profile and job requirements obtained from "
                "previous tasks, tailor the resume to highlight the most "
                "relevant areas. Employ tools to adjust and enhance the "
                "resume content. Make sure this is the best resume even but "
                "don't make up any information. Update every section, "
                "inlcuding the initial summary, work experience, skills, "
                "and education. All to better reflrect the candidates "
                "abilities and how it matches the job posting."
            ),
            expected_output=(
                "An updated resume that effectively highlights the candidate's "
                "qualifications and experiences relevant to the job."
            ),
            output_file="tailored_resume.md",
            context=[self.research_task(), self.profile_task()],
            agent=self.resume_strategist_agent(),
        )

    @task
    def interview_preparation_task(self) -> Task:
        return Task(
            description=(
                "Create a set of potential interview questions and talking "
                "points based on the tailored resume and job requirements. "
                "Utilize tools to generate relevant questions and discussion "
                "points. Make sure to use these question and talking points to "
                "help the candiadte highlight the main points of the resume "
                "and how it matches the job posting."
            ),
            expected_output=(
                "A document containing key questions and talking points "
                "that the candidate should prepare for the initial interview."
            ),
            output_file="interview_materials.md",
            context=[
                self.research_task(),
                self.profile_task(),
                self.resume_strategy_task(),
            ],
            agent=self.interview_preparer_agent(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the JobApplications crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            # process=Process.sequential,
            verbose=True,
        )
