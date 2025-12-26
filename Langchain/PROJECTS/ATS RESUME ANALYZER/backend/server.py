from fastapi import FastAPI
from langserve import add_routes

from chains import (
    ResumeSummaryChain,
    SkillImprovementChain,
    ATSKeywordAnalysisChain,
    ATSEvaluationChain
)

app = FastAPI(
    title="ATS Resume Analyzer API",
    description="An API for analyzing resumes against job descriptions using advanced LLM techniques.",
    version="1.0.0",
)

add_routes(app, ResumeSummaryChain, path="/resume/summary")
add_routes(app, SkillImprovementChain, path="/resume/improvements")
add_routes(app, ATSKeywordAnalysisChain, path="/resume/keywords")
add_routes(app, ATSEvaluationChain, path="/resume/match")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)