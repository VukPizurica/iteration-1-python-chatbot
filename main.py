from dotenv import load_dotenv
from agent import run_agent

load_dotenv()  # loads HF_API_KEY

def main():
    while True:
        task = input("\nEnter task (or 'exit' to quit): ")
        if task.lower() == "exit":
            break
        run_agent(task)

if __name__ == "__main__":
    main()