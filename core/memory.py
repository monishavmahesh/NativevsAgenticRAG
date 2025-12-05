import os, json

class ChatMemory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, content):
        self.messages.append({"type": "User", "content": content})

    def add_ai_message(self, content):
        self.messages.append({"type": "AI", "content": content})

class PersistentMemory:
    def __init__(self, subject: str):
        self.subject = subject
        self.chat_memory = ChatMemory()

# ✅ Load memory per subject
def load_persistent_memory(subject: str):
    return PersistentMemory(subject)
