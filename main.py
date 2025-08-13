from dotenv import load_dotenv

load_dotenv()
def main():
    print("Hello from tutorial-langchain!")


if __name__ == "__main__":
    from src.chapter_6.read_official_documents import read_official_documents

    # main()
    read_official_documents("route_rag")