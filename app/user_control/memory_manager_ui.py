def print_memories(memories, print_fn=print) -> None:
    if not memories:
        print_fn("(no memories stored)")
        return
    for i, m in enumerate(memories, 1):
        print_fn(f"{i:>3}. [{m.kind} | importance {m.importance} | {m.timestamp:%Y-%m-%d}] {m.text}")
        if m.source_message:
            print_fn(f'       from: "{m.source_message[:80]}" (session {m.source_session})')


def manage_memories(store, embedder, user_id, input_fn=input, print_fn=print) -> None:
    """Simple CLI to view, edit, and delete stored memories."""
    while True:
        memories = store.list_all(user_id)
        print_fn("\n=== What I remember about you ===")
        print_memories(memories, print_fn)

        cmd = input_fn("\n[e N] edit   [d N] delete   [clear] delete all   [b] back > ").strip().lower()

        if cmd in ("", "b", "back"):
            return

        if cmd == "clear":
            answer = input_fn("Delete ALL memories? Type 'yes' to confirm > ").strip().lower()
            if answer == "yes":
                store.delete_all(user_id)
                print_fn("All memories deleted.")
            continue

        parts = cmd.split()
        if len(parts) == 2 and parts[0] in ("e", "d") and parts[1].isdigit():
            idx = int(parts[1]) - 1
            if not 0 <= idx < len(memories):
                print_fn("Invalid number.")
                continue
            target = memories[idx]

            if parts[0] == "d":
                store.delete(user_id, target.id)
                print_fn("Deleted.")
            else:
                new_text = input_fn("New text > ").strip()
                if new_text:
                    store.update(user_id, target.id, new_text, embedder.embed_one(new_text))
                    print_fn("Updated.")
            continue

        print_fn("Unrecognized command.")