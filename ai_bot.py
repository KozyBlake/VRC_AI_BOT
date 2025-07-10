import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import time

# Initialize recognizer
r = sr.Recognizer()

# Load DialoGPT-medium
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class SpeechBot:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Speech Bot")
        self.conversation_log = scrolledtext.ScrolledText(self.root, width=60, height=20)
        self.conversation_log.pack(padx=10, pady=10)

        self.status_label = tk.Label(self.root, text="Ready")
        self.status_label.pack()

        self.chat_history_ids = None
        self.turns = 0
        self.queue = queue.Queue()
        self.tts_playing = threading.Event()

        # Keep last N bot responses to detect loops
        self.recent_bot_responses = []

        self.listening_thread = threading.Thread(target=self.background_listen_loop, daemon=True)
        self.listening_thread.start()

        self.root.after(100, self.process_queue)

    def background_listen_loop(self):
        while True:
            if self.tts_playing.is_set():
                time.sleep(0.1)
                continue

            self.queue.put(("status", "Listening..."))
            with sr.Microphone() as source:
                try:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio = r.listen(source, timeout=5, phrase_time_limit=6)
                    self.queue.put(("status", "Processing..."))
                    user_text = r.recognize_google(audio)
                    self.queue.put(("user_text", user_text))
                except sr.WaitTimeoutError:
                    self.queue.put(("status", "Ready"))
                except sr.UnknownValueError:
                    self.queue.put(("bot_text", "Sorry, I didn't catch that."))
                    self.queue.put(("status", "Ready"))
                except Exception as e:
                    self.queue.put(("bot_text", f"Error: {str(e)}"))
                    self.queue.put(("status", "Ready"))

    def process_queue(self):
        while not self.queue.empty():
            msg_type, msg_content = self.queue.get()
            if msg_type == "status":
                self.status_label['text'] = msg_content
            elif msg_type == "user_text":
                self.conversation_log.insert(tk.END, f"User: {msg_content}\n")
                self.conversation_log.see(tk.END)

                response = self.generate_response(msg_content)

                # Detect loop: if response was recently said, reset conversation
                if response in self.recent_bot_responses:
                    self.conversation_log.insert(tk.END, "Bot: (Detected loop, resetting conversation)\n")
                    self.chat_history_ids = None
                    self.turns = 0
                    self.recent_bot_responses = []
                    response = "Let's start over. What would you like to talk about?"

                # Keep recent responses list trimmed
                self.recent_bot_responses.append(response)
                if len(self.recent_bot_responses) > 5:
                    self.recent_bot_responses.pop(0)

                if len(response.strip()) < 2:
                    response = "Could you say that again?"

                self.conversation_log.insert(tk.END, f"Bot: {response}\n")
                self.conversation_log.see(tk.END)

                self.speak_response(response)
                self.status_label['text'] = "Ready"

            elif msg_type == "bot_text":
                self.conversation_log.insert(tk.END, f"Bot: {msg_content}\n")
                self.conversation_log.see(tk.END)

        self.root.after(100, self.process_queue)

    def generate_response(self, user_input):
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

        if self.chat_history_ids is not None:
            self.chat_history_ids = self.chat_history_ids.to(device)
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
            bot_input_ids = bot_input_ids[:, -1000:]
        else:
            bot_input_ids = new_user_input_ids

        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long).to(device)

        self.chat_history_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.9,
        )

        response = tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        self.turns += 1
        if self.turns >= 6:
            self.chat_history_ids = None
            self.turns = 0

        return response

    def speak_response(self, text):
        def run_tts():
            self.tts_playing.set()
            tts_engine = pyttsx3.init()

            voices = tts_engine.getProperty('voices')
            for voice in voices:
                if "zira" in voice.name.lower() or "zira" in voice.id.lower():
                    tts_engine.setProperty('voice', voice.id)
                    break
            else:
                tts_engine.setProperty('voice', voices[0].id)

            tts_engine.say(text)
            tts_engine.runAndWait()
            tts_engine.stop()
            self.tts_playing.clear()

        t = threading.Thread(target=run_tts)
        t.daemon = True
        t.start()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    bot = SpeechBot()
    bot.run()
