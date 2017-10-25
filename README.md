# Reccurent neural network

This program will learn to produce text similar to the one that
it has been training on using a recurrent neural network. 

# How do I use it? 

Do you have a mac or a Linux machine? 
In that case it is super easy to download and run.

However, I don't really know how to compile this program in Windows since I am not using Windows myself. I have used a program called MinGW  compile my own C programs in the past. But Maybe check this out if you are on Windows: https://sourceforge.net/projects/mingw/files/

Otherwise:

Download the program. 
Open a terminal window and type:
"git clone https://github.com/Ricardicus/recurrent-neural-net/" (You may have to install 'git', if you don't already have it!)

"cd recurrent-neural-net" 

"make" - this will compile the program. You need the compiler 'gcc' which is also available for download just like 'git'. 

If there is any complaints, then remove some flags in the 'makefile', I use 'msse3' on my mac but it does not work for my raspberry Pi for example. 

Then run the program:
"./net datafile" 
where datafile is a file with the traning data and it will start traning on it. You can see the progress 
over time. 

Check out the file "lstm.h".

In lstm.h you can edit the program. 
Set the number of layers (2/3 is best I think). 
Set how often it should output data. 

# It will produce samples over time

For example:

18:45:50 Iteration: 146000 (epoch: 37), Loss: 1.066265, record: 1.038003 (iteration: 145815), LR: 0.001000
=====================================================
 which revialy romonsopinese, tief the willece goxilagen. Heaning dided it, andeald leaster. The cintlergy inflencess tilling fifennce, and Gothoseasial interest as hompesion. Ationer their mivided it
=====================================================
18:45:59 Iteration: 147000 (epoch: 37), Loss: 1.112723, record: 1.031639 (iteration: 146052), LR: 0.001000
=====================================================
 in in the elicties in sime tate evils. The miquine of latical of sumntiant effection we are trom one eecreed upon mary insed of shew men hind, heains ty are the plais sour'e belief in cimunaticas, an
=====================================================
18:46:08 Iteration: 148000 (epoch: 37), Loss: 1.065434, record: 1.031639 (iteration: 146052), LR: 0.001000
=====================================================
 its the contar, the frincioural litterte cany of otherity of the seen entrion brititors they bes for weakll. We butain oballiajy frue, and the supprical fated anweln of ecamined effectientians smanke
=====================================================


# Comments

For now I have just started with the implementation.

I figured if it was to be implemented in C instead of Python then
there would be a huge speedup which the results show!

I will give this some time and hopefully I figure out a way that
optimizes the program even more for speed.. Until then, have a go at this 
little program and edit it as you please! 

# make

I use GCC and compile with with: 
gcc *.c -O3 -Ofast -msse3 -lm

Try to optimize it the way you want using the compiler at hand!
