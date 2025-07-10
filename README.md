I already know there are much better alternatives out there than this, however my idea is to simplify the current other VRChat AI bots and just make something that works. 
This uses your Windows TTS through a pre-trained model which is DialoGPT. 

To download and use this properly, use Python Version 3.13.3. I have no clue if this will work with later, or older versions. I haven't tested, and honestly I'm too lazy to do so.

Be sure you run "pip install -r requirements.txt" as forgetting to do this will render the entire project useless. This is included in the download as well.
(However, I will mention, if you run into any issues with pyaudio, you can try "pip install pipwin" and then "pipwin install pyaudio" without the quotes.)

After that, you just run "ai.bot.py" in a command window. 

Currently, there is no way for the bot to stop actively listening to conversations. I made it like this on purpose, however may add that as a feature later on if I feel like it's required.... ooor when I feel like it.


