# The HTML Application

Given a trained network stored in JSON-format, this feature can be used to 
monitor in a visually appealing way the probability distribution of the features being modelled. 
It can be used for debugging purposes but also just to play around with an existing model.

# What does it look like?

Here is an example of how it is used. In this example I have hit the "Sample next" button to generate each character.

<img src="https://raw.githubusercontent.com/Ricardicus/recurrent-neural-net/master/html/Screendump_example.png"></img>

<ol>
  <li>Train a model</li>
  <li>Stop the training process when it is sufficiently trained, by hitting CTRL-C or by first editing code with a stopping condition.</li>
  <li>Load the JSON-file into the GUI</li>
  <li>Either sample a character one-by-one using <i>Sample next</i></li>
  <li>.. or: Give it a sequence of characters to start it off</li>
</ol>
