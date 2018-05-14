# The HTML Application

Given a trained network stored in JSON-format, this feature can be used to 
monitor in a visually appealing way the probability distribution of the features being modelled. 
It can be used for debugging purposes but also just to play around with an existing model.

# What does it look like?

Here is an example of how it is used. In this example I have hit the "Sample next" button to generate each character.

<img src="https://raw.githubusercontent.com/Ricardicus/recurrent-neural-net/master/html/Screendump_example.png"></img>

<i>Steps needed to be taken:</i>
<ol>
  <li>Train a model</li>
  <li>Stop the training process when the model is sufficiently trained. This can be done either by hitting CTRL-C or by editing the code and inserting a stopping condition other than the number of iterations into the function "lstm_train" (in "lstm.c")</li>
  <li>Load the JSON-file into the GUI</li>
  <li>Either sample a character one-by-one using <i>Sample next</i></li>
  <li>.. or: Give it a sequence of characters to start it off</li>
</ol>

<h>Use the example model</h>
Include the .json file in the "models" folder to analyze the behaviour of an already trained model, having read the first Harry Potter book many times.  
