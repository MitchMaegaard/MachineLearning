{\rtf1\ansi\ansicpg1252\cocoartf1671
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 ***********\
README\
***********\
\
- Language: Python 3.\
- Libraries: os, math, random\
\
- NeuralNetwork.py should be in the same folder as hw03, which includes:\
	- hw03.pdf \'97 instructions for homework\
	- trainSet_data (folder)\
		- trainSet05.dat\
		- trainSet10.dat\
		- trainSet15.dat\
		- trainSet20.dat\
	- testSet_data (folder)\
		- testSet05.dat\
		- testSet10.dat\
		- testSet15.dat\
		-testSet20.dat\
	- README.txt\
	- sampleRun.txt \'97 sample run from Prof. Allen\
	- mynn.txt \'97 a printout of a saved network\
		- 5x5-dimensional data\
		- Test accuracy: 20.5%\
\
- MAC\
	- in terminal, type: cd path\
	- type: python ./NeuralNetwork.py\
\
- WINDOWS:\
	- in terminal, type: cd \\path\
	- type: NeuralNetwork.py\
\
- Program should run, then ask the user:\
	1 - If they would like to load a trained network, train a new one, or quit (options: \'93L\'94, \'93T\'94, \'93Q\'94)\
	2 - If \'93T\'94, what resolution of data they would like to train on (options: 5, 10, 15, 20)\
		2.a - Number of hidden layers to include \{0, 10\}\
			2.a.i - Size of each hidden layer \{1, 500\}\
		2.b - If they would like to save the trained network\
	3 - If \'93L\'94, the name of the network file they would like to load\
	4 - If\'92 Q\'94, the program will terminate\
- Program will conduct all necessary training and testing after user has made specified arguments}