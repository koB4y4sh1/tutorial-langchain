from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from ragas.metrics import answer_relevancy, context_precision

from langsmith.evaluation import evaluate
from src.chapter_7.ragas_matric_evaluter import RagasMetricEvaluator


metrics = [context_precision, answer_relevancy]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

evaluators = [
    RagasMetricEvaluator(metric, llm, embeddings).evaluate
    for metric in metrics
]

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = db.as_retriever()

chain = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
).assign(answer=prompt | model | StrOutputParser())

def predict(inputs: dict[str, any]) -> dict[str, any]:
    question = inputs["question"]
    output = chain.invoke(question)
    return {
        "contexts": output["context"],
        "answer": output["answer"],
    }


evaluate(
    predict,
    data="agent-book",
    evaluators=evaluators,
)