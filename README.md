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
Set the number of layers (2/3 is best I think)
Set how often it should output data. 

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
