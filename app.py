import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


import streamlit as st
# LangChain 本体
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # 追加: ChatOpenAI のインポート
# ① .env を読み込む：OPENAI_API_KEY を環境変数にセット
load_dotenv()  # 同じフォルダの .env を自動で読む
# ここで OS 環境変数に OPENAI_API_KEY が入っているはず

# ② Streamlit 画面の構成
st.set_page_config(page_title="Streamlit × LLM デモ", page_icon="🤖")

st.title(" Streamlit × LLM（LangChain）デモアプリ")
st.write(
    """
    **使い方**  
    1. 下のラジオボタンで、AIにどんな専門家として答えてほしいかを選びます。  
    2. テキスト入力欄に質問や相談を書きます。  
    3. 「送信」ボタンを押すと、AI（LLM）の回答が表示されます。  

    ※ このアプリは OpenAI API キーを使います。.env ファイルに `OPENAI_API_KEY=...` を設定してから実行してください。
    """
)

# ③ 専門家の振る舞い（A/B）をラジオで選ぶ
expert = st.radio(
    "AIの専門家モードを選んでください：",
    options=[
        "やさしい国語の先生（小学生向けに噛み砕いて説明）",
        "ITエンジニア（技術的に正確で、要点を箇条書き）",
    ],
    index=0,
)

# ④ 入力フォーム（1つ）
user_text = st.text_area("質問・相談をここに入力", height=120, placeholder="例）生成AIのRAGって何ですか？小学生にもわかる説明で。")

# ⑤ LLM への問い合わせを行う関数
def ask_llm(user_message: str, mode: str) -> str:
    """
    入力テキスト（user_message）と選択した専門家モード（mode）を受け取り、
    LLM（OpenAI）からの回答テキストを返す。
    """

    # 選択に応じて、システムメッセージ（ふるまいの指示）を切り替え
    if mode == "やさしい国語の先生（小学生向けに噛み砕いて説明）":
        system_msg = (
            "あなたは小学生にもわかる言葉で、ゆっくり丁寧に教える国語の先生です。"
            "むずかしい言葉には必ず『かんたんな言いかえ』を添え、例え話を1つ入れてください。"
        )
    elif mode == "ITエンジニア（技術的に正確で、要点を箇条書き）":
        system_msg = (
            "あなたはプロのITエンジニアです。専門用語は正確に使い、"
            "結論→理由→手順の順で、短い箇条書きで端的に説明してください。"
        )
    else:
        system_msg = "あなたは丁寧で親切なアシスタントです。"

    # LangChain のプロンプト（会話の設計）
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            MessagesPlaceholder(variable_name="chat_history"),  # 今回は未使用だが拡張しやすい形
            ("human", "{input}"),
        ]
    )

    # OpenAI の LLM を準備（モデルは軽めのものを例示）
    # ※ 文字化け防止のため temperature は控えめ
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 出力は素のテキストとして受け取る
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # 実行（今回 chat_history は未使用なので空）
    answer = chain.invoke({"input": user_message, "chat_history": []})
    return answer

# ⑥ 送信ボタン
send = st.button("送信", type="primary")

# ⑦ クリックされたら LLM に聞きにいく
if send:
    if not user_text.strip():
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("AIが考え中..."):
            try:
                response = ask_llm(user_text, expert)
                st.success("回答：")
                st.write(response)
            except Exception as e:
                st.error(f"エラーが発生しました：{e}\n\nAPIキーの設定（.env）を確認してください。")
