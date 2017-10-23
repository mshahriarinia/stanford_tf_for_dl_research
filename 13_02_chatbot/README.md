A neural chatbot using sequence to sequence model with attentional decoder. 
- Based on Google Translate Tensorflow model https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
  - Sequence to sequence model by Cho et al.(2014)
- Details of assignment: http://web.stanford.edu/class/cs20si/assignments/a3.pdf 

##########################################

# Setup
Create a data folder
Download Cornell Movie-Dialogs Corpus https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
Unzip
$ python data.py     # This will pre-process the Cornell dataset
$ python chatbot.py --mode [train/chat] <br>
  - mode:
   - train: trains the model. 
    - By default, the model will restore the previously trained weights (if there is any) and continue training up on that.
    - To start training from scratch, delete all the checkpoints in the checkpoints folder.
   - chat: Interaction mode with the bot.

All the conversations you have with the chatbot will be written into the file output_convo.txt in the processed folder.

############################################

# Data
- 220,579 conversational exchanges 
- 10,292 pairs of characters
- 9,035 characters 
- 617 movies
- 304,713 utterances
- In all files the field separator is " +++$+++ "

## File:
- Movies (movie_titles_metadata.txt)
	- Fields: 
		- movieID
		- movie title
		- year
	   	- IMDB rating
		- num of IMDB votes
 		- genres ['genre1','genre2',É,'genreN']

- Characters (movie_characters_metadata.txt)       
	- Fields:
		- characterID
		- character name
		- movieID
		- movie title
		- gender ("?" for unlabeled cases)
		- position in credits ("?" for unlabeled cases) 

- Utterances (movie_lines.txt)
	- Fields:
		- lineID
		- characterID (who uttered this phrase)
		- movieID
		- character name
		- text of the utterance

- Conversations (movie_conversations.txt)
	- Fields
		- characterID of the first character in the conversation
		- characterID of the second character in the conversation
		- movieID
		- list of conversation utterances in chronological order
			to be matched with movie_lines.txt to reconstruct the actual content

# Sample conversation
HUMAN ++++ hi <br>
BOT ++++ hi . what ' s your name ?<br>
HUMAN ++++ i'm chip<br>
BOT ++++ please .<br>
HUMAN ++++ please what?<br>
BOT ++++ i don ' t know what to say .<br>
HUMAN ++++ okay<br>
BOT ++++ okay ?<br>
HUMAN ++++ do you have a girlfriend?<br>
BOT ++++ i ' m sure .<br>
HUMAN ++++ what's her name?<br>
BOT ++++ let ' s talk about something else .<br>

See output_convo.txt for more sample conversations.