from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

def get_git_documents(url:str,repo_path:str)->list[Document]:
    """Gitリポジトリに保存されているファイルから、条件に一致するドキュメントを返却する"""
    # LangChain のドキュメントの増加により、gpt-4o-mini を使用してもmasterブランチにおいても、 Tier 1 ではエラーが発生することが報告されています。
    loader = GitLoader(
        clone_url=url,
        repo_path=repo_path,
        branch="langchain==0.2.13",
        file_filter=file_filter,
    )

    documents = loader.load()

    print(f" {len(documents)}件のドキュメントを読み込みました")

    return documents