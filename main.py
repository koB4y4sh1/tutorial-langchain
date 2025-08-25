from dotenv import load_dotenv

load_dotenv()
def main():
    print("Hello from tutorial-langchain!")


if __name__ == "__main__":
    # from src.chapter_6.read_official_documents import read_official_documents
    # from src.chapter_7.rag_application import rag_application
    from src.chapter_9.invoke_workflow import invoke_workflow

    # main()
    # read_official_documents("route_rag")
    # rag_application()
    invoke_workflow()