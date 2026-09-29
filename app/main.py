import argparse

from app.chat.chat_engine import ChatEngine
from app.user_control.memory_manager_ui import manage_memories

HELP = """Commands:
  /memories  view, edit, or delete what I remember
  /help      show this help
  /quit      end the session (saves memories + summary)
"""


def run_cli(user_id: str) -> None:
    engine = ChatEngine(user_id)
    print(f"Memory chatbot ready (user: {user_id}). Type /help for commands.\n")

    try:
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not text:
                continue
            if text == "/quit":
                break
            if text == "/help":
                print(HELP)
                continue
            if text == "/memories":
                engine.flush()
                manage_memories(engine.store, engine.embedder, user_id)
                continue

            print(f"bot> {engine.respond(text)}\n")
    finally:
        print("Saving memories...")
        engine.end_session()
        print("Goodbye!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot with persistent memory")
    parser.add_argument("--user", default="default", help="user id (separate memory per user)")
    args = parser.parse_args()
    run_cli(args.user)


if __name__ == "__main__":
    main()