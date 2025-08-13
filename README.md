# LangChain チュートリアルプロジェクト

このプロジェクトは、LangChainを使用したAIアプリケーション開発のチュートリアルです。

## 前提条件

- Python 3.8以上
- uvパッケージマネージャー

## セットアップ手順

### 1. uvのインストール

Windows環境の場合：
```bash
# PowerShellで実行
irm https://astral.sh/uv/install.ps1 | iex
```

macOS/Linux環境の場合：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. プロジェクトの初期化

```bash
# 新しいプロジェクトディレクトリを作成
mkdir tutorial-langchain
cd tutorial-langchain

# uvでプロジェクトを初期化
uv init
```

### 3. 依存関係のインストール

```bash
# 必要なパッケージをインストール
uv add langchain-core==0.3.0
uv add langchain-openai==0.2.0
uv add langchain-community==0.3.0
uv add GitPython==3.1.43
uv add langchain-chroma==0.1.4
uv add tavily-python==0.5.0
uv add pydantic==2.10.6
```

**注意**: Windows環境で`langchain-chroma`のインストールに失敗する場合は、Microsoft C++ Build Toolsが必要です。

#### Microsoft C++ Build Toolsのインストール（Windows環境のみ）

1. [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) をダウンロード
2. インストーラーを実行し、"C++によるデスクトップ開発" ワークロードを選択
3. インストール完了後、PowerShellを再起動
4. 再度パッケージのインストールを実行

### 4. 仮想環境の有効化

```bash
# 仮想環境を有効化
uv shell

# または、直接コマンドを実行
uv run python main.py
```

### 5. プロジェクトの実行

```bash
# 仮想環境内で実行
python main.py

# または、uv runを使用
uv run python main.py
```

## プロジェクト構造

```
tutorial-langchain/
├── main.py              # メインアプリケーションファイル
├── pyproject.toml       # プロジェクト設定ファイル
├── uv.lock             # 依存関係のロックファイル
└── README.md           # このファイル
```

## トラブルシューティング

### パッケージのビルドエラーが発生する場合

1. Microsoft C++ Build Toolsがインストールされているか確認
2. 事前にビルドされたパッケージを使用：
   ```bash
   uv add --prebuilt [パッケージ名]
   ```
3. 個別にパッケージをインストールして問題を特定

### 仮想環境の問題

```bash
# 仮想環境を再作成
uv remove --all
uv sync
```

## 参考リンク

- [uv公式ドキュメント](https://docs.astral.sh/uv/)
- [LangChain公式ドキュメント](https://python.langchain.com/)
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
