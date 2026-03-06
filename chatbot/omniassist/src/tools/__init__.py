"""Tool registry — central place to import all tools."""

from src.tools.calculator import calculator
from src.tools.weather import get_weather
from src.tools.web_search import web_search
from src.tools.calendar_tool import get_current_time, schedule_event
from src.tools.email_tool import send_email
from src.tools.file_tools import read_file, write_file
from src.tools.code_runner import run_python_code
from src.tools.knowledge_base import search_knowledge_base
from src.tools.state_updating_tools import set_user_preference, flag_for_review

# Tool groups for different agents
BASE_TOOLS = [calculator, get_weather, get_current_time]
CHAT_TOOLS = [calculator, get_weather, get_current_time, search_knowledge_base]
RESEARCH_TOOLS = [web_search, search_knowledge_base]
WRITING_TOOLS = [web_search, read_file, write_file]
TASK_TOOLS = [get_current_time, schedule_event]
CODE_TOOLS = [run_python_code]
ALL_TOOLS = [
    calculator, get_weather, web_search, get_current_time, schedule_event,
    send_email, read_file, write_file, run_python_code, search_knowledge_base,
    set_user_preference, flag_for_review,
]
