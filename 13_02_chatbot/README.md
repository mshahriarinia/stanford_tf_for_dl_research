<<<<<<<
A neural chatbot using sequence to sequence model with attentional decoder. 
- Based on Google Translate Tensorflow model https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
  - Sequence to sequence model by Cho et al.(2014)
- Details of assignment: http://web.stanford.edu/class/cs20si/assignments/a3.pdf 

<h2>Sample conversation</h2>
HUMAN ++++ hi <br>
BOT ++++ hi . what ' s your name ?<br>
HUMAN ++++ i'm chip<br>
BOT ++++ please .<br>
HUMAN ++++ please what?<br>
BOT ++++ i don ' t know what to say .<br>
HUMAN ++++ where are you from?<br>
BOT ++++ i ' m fine .<br>
HUMAN ++++ are you a man?<br>
BOT ++++ i ' m a little .<br>
HUMAN ++++ what's your name?<br>
BOT ++++ chris -<br>
HUMAN ++++ chris who?<br>
BOT ++++ cornelius .<br>
HUMAN ++++ so i guess you're a guy.<br>
BOT ++++ i am .<br>
HUMAN ++++ okay<br>
BOT ++++ okay ?<br>
HUMAN ++++ do you have a girlfriend?<br>
BOT ++++ i ' m sure .<br>
HUMAN ++++ what's her name?<br>
BOT ++++ let ' s talk about something else .<br>

See output_convo.txt for more sample conversations.

##########################################

# Setup
- Create a data folder and download Cornell Movie-Dialogs Corpus https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
- Unzip
- $ python data.py     # This will pre-process the Cornell dataset
- $ python chatbot.py --mode [train/chat] <br>
  - If mode is train, then you train the chatbot. By default, the model will restore the previously trained weights (if there is any) and continue training up on that.
    - To start training from scratch, delete all the checkpoints in the checkpoints folder.
    - If the mode is chat, you'll go into the interaction mode with the bot.

By default, all the conversations you have with the chatbot will be written into the file output_convo.txt in the processed folder.