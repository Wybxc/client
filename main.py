import streamlit as st
import llama

st.set_page_config(
    page_title="Llama.cpp",
    page_icon="🦙",
    layout="centered",
)

st.title("Llama.cpp")

with st.sidebar:
    endpoint = st.text_input("Endpoint", value="http://localhost:8080/completion")
    conversation_prompt = st.text_area(
        "Conversation Prompt",
        value="This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.",
    )
    initial_prompt = st.text_area(
        "Initial Prompt",
        value="嗨，你好！我是Llama。有什么我可以帮助你的吗？",
        help="Set the initial prompt for the conversation.",
    )
    user_name = st.text_input("User name", value="User")
    llama_name = st.text_input("Llama name", value="Llama")

    role_to_name = {
        "user": user_name,
        "assistant": llama_name,
    }

    stop = st.text_area(
        "Stop",
        value="</s>\nLlama:\nUser:\nLlama：\nUser：",
        help="A list of strings that will stop the generation when encountered.",
    ).splitlines()

    n_predict = int(
        st.number_input(
            "Predictions",
            value=400,
            min_value=-1,
            max_value=2048,
            help="Set the maximum number of tokens to predict when generating text.",
        )
    )
    top_k = int(
        st.number_input(
            "Top K",
            value=40,
            min_value=-1,
            max_value=100,
            help="Limit the next token selection to the K most probable tokens. Set to -1 to disable.",
        )
    )
    top_p = st.number_input(
        "Top P",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        help="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.",
    )
    min_p = st.number_input(
        "Min P",
        value=0.05,
        min_value=0.0,
        max_value=1.0,
        help="The minimum probability for a token to be considered, relative to the probability of the most likely token",
    )
    temperature = st.number_input(
        "Temperature",
        value=0.7,
        min_value=0.0,
        max_value=1.5,
        help="Adjust the randomness of the generated text.",
    )
    repeat_penalty = st.number_input(
        "Repeat penalty",
        value=1.18,
        min_value=0.0,
        max_value=10.0,
        help="Control the repetition of token sequences in the generated text.",
    )
    repeat_last_n = int(
        st.number_input(
            "Repeat last N",
            value=256,
            min_value=0,
            max_value=2048,
            help="Last n tokens to consider for penalizing repetition.",
        )
    )
    seed = st.number_input(
        "Seed",
        value=-1,
        min_value=-1,
        max_value=2**31 - 1,
        help="Seed for the random number generator. Set to -1 to use a random seed.",
    )

    grammer = st.text_area(
        "Grammer", value="", help="Set grammar for grammar-based sampling."
    )

    with st.expander("Advanced"):
        tfs_z = st.number_input(
            "TFS Z",
            value=1.0,
            min_value=0.0,
            max_value=1.0,
            help="Enable tail free sampling with parameter z. Set to 1 to disable.",
        )
        typical_p = st.number_input(
            "Typical P",
            value=1.0,
            min_value=0.0,
            max_value=1.0,
            help="Enable locally typical sampling with parameter p. Set to 1 to disable.",
        )
        presence_penalty = st.number_input(
            "Presence penalty",
            value=0.0,
            min_value=0.0,
            max_value=10.0,
            help="Repeat alpha presence penalty. Set to 0 to disable.",
        )
        frequency_penalty = st.number_input(
            "Frequency penalty",
            value=0.0,
            min_value=0.0,
            max_value=10.0,
            help="Repeat alpha frequency penalty. Set to 0 to disable.",
        )
        _, mirostat = st.selectbox(
            "Mirostat",
            options=[
                ("Disabled", 0),
                ("V1", 1),
                ("V2", 2),
            ],
            index=0,
            format_func=lambda x: x[0],
            help="Mirostat sampling, controlling perplexity during text generation.",
        ) or ("", 0)
        mirostat_tau = st.number_input(
            "Mirostat tau",
            value=5,
            min_value=0,
            max_value=10,
            help="Set the Mirostat target entropy, parameter tau.",
        )
        mirostat_eta = st.number_input(
            "Mirostat eta",
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            help="Set the Mirostat learning rate, parameter eta.",
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear History"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

history = "\n".join(
    f"{role_to_name[message['role']]}: {message['content']}"
    for message in st.session_state.messages
)

if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    prompt = f"""{conversation_prompt}

{llama_name}: {initial_prompt}
{history}
{user_name}: {user_prompt}
{llama_name}: """

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in llama.get_response(
            prompt,
            endpoint=endpoint,
            n_predict=n_predict,
            temperature=temperature,
            stop=stop,
            repeat_last_n=repeat_last_n,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            tfs_z=tfs_z,
            typical_p=typical_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            mirostat=mirostat,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            grammar=grammer,
            n_probs=0,
            cache_prompt=True,
            slot_id=0,
        ):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
