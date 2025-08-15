from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document

CLONE_URL="https://github.com/langchain-ai/langchain"
REPO_PATH="./repo/langchain"

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

def get_git_documents()->list[Document]:
    """Gitリポジトリに保存されているファイルから、条件に一致するドキュメントを返却する"""
    # LangChain のドキュメントの増加により、gpt-4o-mini を使用してもmasterブランチにおいても、 Tier 1 ではエラーが発生することが報告されています。
    loader = GitLoader(
        clone_url=CLONE_URL,
        repo_path=REPO_PATH,
        branch="langchain==0.2.13",
        file_filter=file_filter,
    )

    documents = loader.load()

    print(f" {len(documents)}件のドキュメントを読み込みました")

    return documents